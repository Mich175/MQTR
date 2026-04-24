"""
models/gtr_tracker.py
Main GTR tracker – integrates YOLO detector, ST Re-ID extractor,
GTR Transformer, and multi-query associator with Trio-Weight.

Completely independent of detectron2.
"""

import torch
import numpy as np
import cv2
from typing import List, Optional, Dict
from collections import deque

from detector.yolo_detector import YOLODetector, Detection
from tracker.transformer import GTRTransformer
from tracker.st_reid_extractor import STReIDExtractor
from tracker.association import MultiQueryAssociator, TrackInstance


class GTRTracker:
    """
    Full GTR tracking pipeline:

    Frame → YOLO detection → ST Re-ID extraction → sliding window →
    Multi-query GTR association with Trio-Weight → track ID assignment
    """

    def __init__(
        self,
        # Detector
        yolo_model:           str,
        conf_thresh:          float,
        target_classes:       Optional[List[int]],

        # Transformer
        d_model:              int,
        n_heads:              int,
        n_enc:                int,
        n_dec:                int,

        # Sliding window
        window_size:          int,
        min_track_len:        int,

        # Association
        overlap_thresh:       float,
        asso_thresh:          float,
        with_iou:             bool,
        decay_time:           float,
        max_center_dist:      float,
        query_strategy:       str,
        n_queries:            int,
        motion_sigma:         float,

        # Trio-Weight 开关
        use_temporal_weight:    bool,
        use_confidence_weight:  bool,
        use_motion_weight:      bool,

        # Re-ID
        n_frames:             int,

        device:               str,
    ):
        self.window_size   = window_size
        self.min_track_len = min_track_len
        self.device        = device

        # ── Components ────────────────────────────────────────────────────────
        self.detector = YOLODetector(
            model_name=yolo_model,
            device=device,
            conf_thresh=conf_thresh,
            target_classes=target_classes,
        )

        self.reid = STReIDExtractor(
            d_model=d_model,
            n_frames=n_frames,
            freeze_vit=False,
            device=device,
        )

        self.transformer = GTRTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_enc=n_enc,
            n_dec=n_dec,
        ).to(device)

        self.associator = MultiQueryAssociator(
            overlap_thresh        = overlap_thresh,
            asso_thresh           = asso_thresh,
            with_iou              = with_iou,
            decay_time            = decay_time,
            max_center_dist       = max_center_dist,
            query_strategy        = query_strategy,
            n_queries             = n_queries,
            motion_sigma          = motion_sigma,
            use_temporal_weight   = use_temporal_weight,
            use_confidence_weight = use_confidence_weight,
            use_motion_weight     = use_motion_weight,
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.window:        deque = deque(maxlen=window_size)
        self.id_count:      int   = 0
        self.frame_id:      int   = 0
        self.all_instances: List[TrackInstance] = []

    @classmethod
    def from_config(cls, cfg: dict) -> "GTRTracker":
        """
        Build GTRTracker from config dict.

        Usage:
            import yaml
            cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
            tracker = GTRTracker.from_config(cfg)
        """
        return cls(
            yolo_model            = cfg["detector"]["model_name"],
            conf_thresh           = cfg["detector"]["conf_thresh"],
            target_classes        = cfg["detector"]["target_classes"],
            d_model               = cfg["transformer"]["d_model"],
            n_heads               = cfg["transformer"]["n_heads"],
            n_enc                 = cfg["transformer"]["n_enc"],
            n_dec                 = cfg["transformer"]["n_dec"],
            window_size           = cfg["tracker"]["window_size"],
            min_track_len         = cfg["tracker"]["min_track_len"],
            overlap_thresh        = cfg["association"]["overlap_thresh"],
            asso_thresh           = cfg["association"]["asso_thresh"],
            with_iou              = cfg["association"]["with_iou"],
            decay_time            = cfg["association"]["decay_time"],
            max_center_dist       = cfg["association"]["max_center_dist"],
            query_strategy        = cfg["association"]["query_strategy"],
            n_queries             = cfg["association"]["n_queries"],
            motion_sigma          = cfg["association"]["motion_sigma"],
            use_temporal_weight   = cfg["association"]["use_temporal_weight"],
            use_confidence_weight = cfg["association"]["use_confidence_weight"],
            use_motion_weight     = cfg["association"]["use_motion_weight"],
            n_frames              = cfg["reid"]["n_frames"],
            device                = cfg["eval"]["device"],
        )

    def reset(self):
        """Reset tracker state for a new video."""
        self.window.clear()
        self.id_count      = 0
        self.frame_id      = 0
        self.all_instances = []
        self.reid.reset_history()   # 清空 ST Re-ID 历史缓存

    def load_reid_checkpoint(self, checkpoint_path: str):
        """
        Load trained Re-ID weights.

        Usage:
            tracker.load_reid_checkpoint("checkpoints/best_reid.pth")
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.reid.load_state_dict(ckpt["model"])
        print(f"✅ Re-ID checkpoint loaded from {checkpoint_path}")

    @torch.no_grad()
    def update(self, frame: np.ndarray) -> List[Dict]:
        """
        Process one frame and return tracked objects.

        Args:
            frame: HxWx3 numpy array (BGR)

        Returns:
            list of dicts: track_id, box, score, class_id, frame_id
        """
        # Step 1: Detect
        detection = self.detector.detect(frame, self.frame_id)

        # Step 2: Extract ST Re-ID features
        current_ids = (list(self.window)[-1].track_ids
                       if len(self.window) > 0 else None)
        reid_feats  = self.reid.extract(
            frame, detection.boxes, track_ids=current_ids)

        # Step 3: Build TrackInstance
        inst = TrackInstance(
            boxes=detection.boxes,
            scores=detection.scores,
            class_ids=detection.class_ids,
            reid_features=reid_feats,
            frame_id=self.frame_id,
            track_ids=None,
        )

        # Step 4: First frame → assign new IDs directly
        if self.frame_id == 0:
            n = len(inst.boxes)
            inst.track_ids = torch.arange(
                1, n + 1, dtype=torch.long, device=self.device)
            self.id_count = n + 1

        self.window.append(inst)
        self.all_instances.append(inst)

        # Step 5: Multi-query association with Trio-Weight
        if self.frame_id > 0:
            window_list = list(self.window)
            window_list, self.id_count = self.associator.associate(
                transformer=self.transformer,
                instances=window_list,
                id_count=self.id_count,
            )
            self.window.clear()
            for w in window_list:
                self.window.append(w)

        # Step 6: Build output
        current = list(self.window)[-1]
        results = []
        if current.track_ids is not None:
            for i in range(len(current.boxes)):
                results.append({
                    "track_id": current.track_ids[i].item(),
                    "box":      current.boxes[i].cpu().tolist(),
                    "score":    current.scores[i].item(),
                    "class_id": current.class_ids[i].item(),
                    "frame_id": self.frame_id,
                })

        self.frame_id += 1
        return results

    def track_video(
        self,
        video_path:      str,
        output_path:     Optional[str] = None,
        show:            bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> List[List[Dict]]:
        """
        Track all objects in a video.

        Args:
            video_path:      input video path
            output_path:     save annotated video (optional)
            show:            display live (optional)
            checkpoint_path: load Re-ID weights before tracking
        """
        self.reset()

        if checkpoint_path:
            self.load_reid_checkpoint(checkpoint_path)

        cap    = cv2.VideoCapture(video_path)
        writer = None

        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS)
            W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (W, H),
            )

        all_results = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.update(frame)
            all_results.append(results)
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")

            if output_path or show:
                vis = self._draw(frame, results)
                if writer:
                    writer.write(vis)
                if show:
                    cv2.imshow("GTR Tracker", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        print(f"✅ Done! {frame_count} frames processed.")
        return all_results

    def _draw(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and track IDs."""
        vis    = frame.copy()
        colors = {}
        for r in results:
            tid = r["track_id"]
            if tid not in colors:
                np.random.seed(tid)
                colors[tid] = tuple(np.random.randint(50, 255, 3).tolist())
            x1, y1, x2, y2 = map(int, r["box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), colors[tid], 2)
            cv2.putText(vis, f"ID:{tid} {r['score']:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[tid], 2)
        return vis