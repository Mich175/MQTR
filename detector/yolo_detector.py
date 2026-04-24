"""
detector/yolo_detector.py
YOLO detector wrapper – replaces detectron2's CenterNet2 detector.
Supports YOLOv8, YOLOv9, YOLOv11 via ultralytics.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from ultralytics import YOLO


@dataclass
class Detection:
    """Single frame detection results."""
    boxes:      torch.Tensor   # (N, 4) xyxy format
    scores:     torch.Tensor   # (N,)
    class_ids:  torch.Tensor   # (N,)
    frame_id:   int


class YOLODetector:
    """
    Wraps ultralytics YOLO for use with GTR tracker.
    Completely replaces detectron2 + CenterNet2.
    """

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        device: str = "cuda",
        conf_thresh: float = 0.3,
        nms_thresh: float = 0.45,
        target_classes: Optional[List[int]] = None,
        img_size: int = 1280,
    ):
        """
        Args:
            model_name:     YOLO model ('yolov8x.pt', 'yolov8n.pt', 'yolo11x.pt' ...)
            device:         'cuda' or 'cpu'
            conf_thresh:    detection confidence threshold
            nms_thresh:     NMS IoU threshold
            target_classes: list of COCO class ids to track (None = all)
            img_size:       inference image size
        """
        self.device       = device
        self.conf_thresh  = conf_thresh
        self.nms_thresh   = nms_thresh
        self.target_classes = target_classes
        self.img_size     = img_size

        self.model = YOLO(model_name)
        self.model.to(device)
        print(f"[YOLODetector] Loaded {model_name} on {device}")

    @torch.no_grad()
    def detect(self, frame: np.ndarray, frame_id: int = 0) -> Detection:
        """
        Run detection on a single frame.

        Args:
            frame:    HxWx3 numpy array (BGR or RGB)
            frame_id: frame index

        Returns:
            Detection dataclass
        """
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.nms_thresh,
            imgsz=self.img_size,
            classes=self.target_classes,
            verbose=False,
        )[0]

        boxes     = results.boxes.xyxy   # (N, 4)
        scores    = results.boxes.conf   # (N,)
        class_ids = results.boxes.cls.long()  # (N,)

        return Detection(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            frame_id=frame_id,
        )

    @torch.no_grad()
    def detect_batch(
        self, frames: List[np.ndarray], start_id: int = 0
    ) -> List[Detection]:
        """
        Run detection on a list of frames (batch inference).

        Args:
            frames:   list of HxWx3 numpy arrays
            start_id: starting frame_id

        Returns:
            list of Detection
        """
        results = self.model(
            frames,
            conf=self.conf_thresh,
            iou=self.nms_thresh,
            imgsz=self.img_size,
            classes=self.target_classes,
            verbose=False,
            stream=True,
        )

        detections = []
        for frame_id, r in enumerate(results):
            detections.append(Detection(
                boxes=r.boxes.xyxy,
                scores=r.boxes.conf,
                class_ids=r.boxes.cls.long(),
                frame_id=start_id + frame_id,
            ))
        return detections
