"""
utils/metrics.py
MOT evaluation metrics: MOTA, HOTA, IDF1.

Writes tracker results in MOT format and
calls TrackEval for official evaluation.
"""

import os
import numpy as np
from pathlib import Path


def write_mot_results(results, output_dir: str, seq_name: str):
    """
    Write tracking results in MOT challenge format.

    Format: frame, id, x, y, w, h, conf, -1, -1, -1

    Args:
        results:    list of per-frame results from tracker
        output_dir: directory to save results
        seq_name:   sequence name (e.g. MOT17-02-FRCNN)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{seq_name}.txt")

    lines = []
    for frame_results in results:
        for r in frame_results:
            x1, y1, x2, y2 = r["box"]
            w = x2 - x1
            h = y2 - y1
            line = (f"{r['frame_id'] + 1},{r['track_id']},"
                    f"{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                    f"{r['score']:.4f},-1,-1,-1")
            lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[Metrics] Saved results → {output_path}")
    return output_path


def compute_basic_metrics(gt_path: str, pred_path: str):
    """
    Compute basic MOT metrics without TrackEval.
    Returns MOTA approximation.

    For official HOTA/IDF1, use TrackEval separately.
    """
    # Load predictions
    pred = {}
    with open(pred_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), \
                          float(parts[4]), float(parts[5])
            if frame_id not in pred:
                pred[frame_id] = []
            pred[frame_id].append({
                "track_id": track_id,
                "box": [x, y, x + w, y + h]
            })

    # Load ground truth
    gt = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            if conf < 0.5:
                continue
            x, y, w, h = float(parts[2]), float(parts[3]), \
                          float(parts[4]), float(parts[5])
            if frame_id not in gt:
                gt[frame_id] = []
            gt[frame_id].append([x, y, x + w, y + h])

    # Simple MOTA calculation
    total_gt   = sum(len(v) for v in gt.values())
    total_fp   = 0
    total_fn   = 0
    total_id_sw = 0

    iou_threshold = 0.5
    prev_matches  = {}   # gt_idx → track_id

    for frame_id in sorted(gt.keys()):
        gt_boxes   = gt.get(frame_id, [])
        pred_boxes = pred.get(frame_id, [])

        if not gt_boxes:
            total_fp += len(pred_boxes)
            continue

        if not pred_boxes:
            total_fn += len(gt_boxes)
            continue

        # Compute IoU matrix
        gt_arr   = np.array(gt_boxes)
        pred_arr = np.array([p["box"] for p in pred_boxes])

        iou_matrix = _iou_matrix(gt_arr, pred_arr)

        # Greedy matching
        matched_gt   = set()
        matched_pred = set()
        current_matches = {}

        for _ in range(min(len(gt_boxes), len(pred_boxes))):
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < iou_threshold:
                break
            gi, pi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched_gt.add(gi)
            matched_pred.add(pi)
            current_matches[gi] = pred_boxes[pi]["track_id"]
            iou_matrix[gi, :] = -1
            iou_matrix[:, pi] = -1

        # Count metrics
        total_fn += len(gt_boxes) - len(matched_gt)
        total_fp += len(pred_boxes) - len(matched_pred)

        # Count ID switches
        for gi, tid in current_matches.items():
            if gi in prev_matches and prev_matches[gi] != tid:
                total_id_sw += 1

        prev_matches = current_matches

    # MOTA = 1 - (FN + FP + IDSW) / GT
    mota = 1 - (total_fn + total_fp + total_id_sw) / max(total_gt, 1)

    return {
        "MOTA":  round(mota * 100, 2),
        "FP":    total_fp,
        "FN":    total_fn,
        "IDSW":  total_id_sw,
        "GT":    total_gt,
    }


def _iou_matrix(gt_boxes, pred_boxes):
    """Compute IoU matrix between gt and pred boxes."""
    iou = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gb in enumerate(gt_boxes):
        for j, pb in enumerate(pred_boxes):
            iou[i, j] = _iou(gb, pb)
    return iou


def _iou(box_a, box_b):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter

    return inter / (union + 1e-8)
