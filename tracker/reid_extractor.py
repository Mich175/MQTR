"""
tracker/reid_extractor.py
Lightweight Re-ID feature extractor using RoI-aligned crops from YOLO detections.
Replaces detectron2's roi_heads reid feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import roi_align
import numpy as np
from typing import List
from detector.yolo_detector import Detection


class ReidExtractor(nn.Module):
    """
    Extracts Re-ID features from detected bounding box crops.

    Architecture: lightweight CNN → global avg pool → L2-normalized embedding

    Can be replaced with any stronger Re-ID backbone
    (e.g., OSNet, TransReID) without changing the interface.
    """

    def __init__(
        self,
        d_model:   int = 256,
        crop_size: int = 128,
        device:    str = "cuda",
    ):
        super().__init__()
        self.d_model   = d_model
        self.crop_size = crop_size
        self.device    = device

        # Simple CNN backbone for reid features
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to(device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((crop_size, crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(
        self,
        frame:     np.ndarray,      # HxWx3 BGR
        detection: "Detection",
    ) -> torch.Tensor:              # (N, d_model)
        """
        Extract Re-ID features for all detections in a frame.

        Args:
            frame:     raw frame (HxWx3 numpy)
            detection: Detection object with boxes

        Returns:
            features: (N, d_model) L2-normalized
        """
        if len(detection.boxes) == 0:
            return torch.zeros(
                (0, self.d_model), device=self.device)

        frame_rgb = frame[:, :, ::-1].copy()  # BGR -> RGB
        H, W = frame_rgb.shape[:2]

        crops = []
        for box in detection.boxes.cpu():
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(W, int(x2)); y2 = min(H, int(y2))
            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            else:
                crop = frame_rgb[y1:y2, x1:x2]
            crops.append(self.transform(crop))

        batch = torch.stack(crops).to(self.device)   # (N, 3, H, W)
        feats = self.backbone(batch).squeeze(-1).squeeze(-1)  # (N, d_model)
        feats = F.normalize(feats, dim=1)             # L2 normalize
        return feats