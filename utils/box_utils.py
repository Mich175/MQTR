"""
utils/box_utils.py
Bounding box utility functions.
"""

import torch
import numpy as np


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Args:
        boxes_a: (N, 4) xyxy
        boxes_b: (M, 4) xyxy

    Returns:
        iou: (N, M)
    """
    ax1, ay1, ax2, ay2 = boxes_a.unbind(1)
    bx1, by1, bx2, by2 = boxes_b.unbind(1)

    inter_x1 = torch.max(ax1[:, None], bx1[None, :])
    inter_y1 = torch.max(ay1[:, None], by1[None, :])
    inter_x2 = torch.min(ax2[:, None], bx2[None, :])
    inter_y2 = torch.min(ay2[:, None], by2[None, :])

    inter = (inter_x2 - inter_x1).clamp(min=0) * \
            (inter_y2 - inter_y1).clamp(min=0)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter

    return inter / (union + 1e-8)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def box_center(boxes: torch.Tensor) -> torch.Tensor:
    """Get center points of boxes. Input: (N, 4) xyxy."""
    return (boxes[:, :2] + boxes[:, 2:]) / 2


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Get area of boxes. Input: (N, 4) xyxy."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
