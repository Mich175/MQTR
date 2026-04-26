"""
tracker/st_reid_extractor.py

Spatial-Temporal Transformer for Re-ID feature extraction.

Architecture:
  1. Patch Embedding  : split crop into 16x16 patches
  2. Spatial Encoder  : ViT-Small (pretrained) - captures appearance
  3. Temporal Encoder : cross-frame attention   - captures motion context
  4. ST Fusion        : merge spatial + temporal
  5. Output           : 256-dim L2-normalized Re-ID feature

Design choices:
  - ViT-Small (patch16, ImageNet pretrained) as spatial backbone
    → strong appearance features with minimal data
  - Temporal encoder uses [CLS] tokens from each frame
    → lightweight, avoids O(T*N_patch^2) cost
  - History window: 4 frames (current + 3 past)
  - Output dim: 256 to match GTR Transformer d_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from typing import List, Optional, Deque
from collections import deque

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("[WARNING] timm not installed. Run: pip install timm")


# ── Temporal Transformer ───────────────────────────────────────────────────────

class TemporalAttentionLayer(nn.Module):
    """
    Single layer of temporal self-attention.
    Operates on [CLS] tokens across T frames: (B, T, d)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        x = self.norm1(x + self.drop(self.attn(x, x, x)[0]))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class TemporalEncoder(nn.Module):
    """
    Temporal Transformer: models motion context across frames.

    Input:  CLS tokens from T frames → (B, T, d_spatial)
    Output: temporally-enriched tokens → (B, T, d_model)
    """

    def __init__(
        self,
        d_spatial: int = 384,   # ViT-Small output dim
        d_model:   int = 256,
        n_heads:   int = 4,
        n_layers:  int = 2,
        max_frames: int = 4,
        dropout:   float = 0.1,
    ):
        super().__init__()

        # Project spatial features to d_model
        self.input_proj = nn.Linear(d_spatial, d_model)

        # Learnable temporal position embeddings
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, max_frames, d_model))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        # Temporal self-attention layers
        self.layers = nn.ModuleList([
            TemporalAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_tokens: (B, T, d_spatial)  CLS token per frame

        Returns:
            (B, T, d_model)  temporally enriched features
        """
        T = cls_tokens.shape[1]
        x = self.input_proj(cls_tokens)           # (B, T, d_model)
        x = x + self.temporal_pos[:, :T, :]       # add temporal pos embed
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ── ST Fusion ──────────────────────────────────────────────────────────────────

class STFusion(nn.Module):
    """
    Fuse spatial and temporal features.

    Strategy: gated fusion
      - Spatial branch: appearance (what does the object look like?)
      - Temporal branch: motion context (how is it moving?)
      - Gate: learned weighting between the two
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        # Gate: sigmoid output controls spatial vs temporal blend
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(
        self,
        spatial:  torch.Tensor,   # (B, d_model)
        temporal: torch.Tensor,   # (B, d_model)
    ) -> torch.Tensor:            # (B, d_model)
        combined = torch.cat([spatial, temporal], dim=-1)   # (B, 2d)
        gate     = self.gate(combined)                       # (B, d)
        fused    = gate * spatial + (1 - gate) * temporal   # gated blend
        fused    = self.out_proj(combined) + fused           # residual
        return self.norm(fused)


# ── Main ST Re-ID Extractor ────────────────────────────────────────────────────

class STReIDExtractor(nn.Module):
    """
    Spatial-Temporal Re-ID feature extractor.

    Replaces the simple CNN in reid_extractor.py with a
    ViT-based spatial encoder + temporal attention across frames.

    Args:
        d_model:     output feature dimension (256 to match GTR)
        n_frames:    number of frames in temporal window (current + history)
        vit_name:    timm ViT model name
        freeze_vit:  freeze ViT backbone (True during early training)
        device:      'cuda' or 'cpu'
    """

    # ViT-Small: good balance of speed vs quality
    # d_spatial = 384 (ViT-Small hidden dim)
    VIT_NAME     = "vit_small_patch16_224"
    D_SPATIAL    = 384

    def __init__(
        self,
        d_model:    int  = 256,
        n_frames:   int  = 4,      # temporal window size
        freeze_vit: bool = True,   # freeze during warmup
        device:     str  = "cuda",
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("Please install timm: pip install timm")

        self.d_model  = d_model
        self.n_frames = n_frames
        self.device   = device

        # ── Spatial: ViT-Small (ImageNet pretrained) ──────────────────────────
        self.vit = timm.create_model(
            self.VIT_NAME,
            pretrained=True,
            num_classes=0,      # remove classification head
        )
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False
            print("[STReID] ViT backbone frozen (freeze_vit=True)")

        # Project ViT output to d_model
        self.spatial_proj = nn.Linear(self.D_SPATIAL, d_model)

        # ── Temporal: attention across frames ─────────────────────────────────
        self.temporal_encoder = TemporalEncoder(
            d_spatial=self.D_SPATIAL,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
            max_frames=n_frames,
        )

        # ── ST Fusion ──────────────────────────────────────────────────────────
        self.fusion = STFusion(d_model=d_model)

        # ── Per-object frame history buffer ────────────────────────────────────
        # Maps track_id -> deque of past CLS tokens
        self._history: dict = {}

        self.to(device)

        # Image preprocessing for ViT
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    def unfreeze_vit(self, n_last_blocks: int = 4):
        """
        Gradually unfreeze last n blocks of ViT for fine-tuning.
        Call this after warmup training is stable.
        """
        blocks = list(self.vit.blocks)
        for block in blocks[-n_last_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"[STReID] Unfroze last {n_last_blocks} ViT blocks")

    def _extract_spatial(
        self, crops: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract per-frame spatial features using ViT.

        Args:
            crops: (N, 3, 224, 224)

        Returns:
            cls_tokens: (N, D_SPATIAL)
        """
        return self.vit(crops)   # (N, D_SPATIAL)

    def _get_history(
        self,
        track_ids: Optional[torch.Tensor],
        cls_tokens: torch.Tensor,   # (N, D_SPATIAL)
    ) -> torch.Tensor:              # (N, T, D_SPATIAL)
        """
        For each object, retrieve its past CLS tokens and
        update history buffer.

        If track_ids is None (first frame), return just current token.
        """
        N = cls_tokens.shape[0]
        T = self.n_frames

        result = []
        for i in range(N):
            tok = cls_tokens[i]  # (D_SPATIAL,)

            if track_ids is not None:
                tid = track_ids[i].item()
                if tid not in self._history:
                    self._history[tid] = deque(maxlen=T)
                self._history[tid].append(tok.detach())
                history = list(self._history[tid])
            else:
                history = [tok.detach()]

            # Pad with zeros if not enough history
            while len(history) < T:
                history.insert(0, torch.zeros_like(tok))

            seq = torch.stack(history[-T:], dim=0)  # (T, D_SPATIAL)
            result.append(seq)

        return torch.stack(result, dim=0)  # (N, T, D_SPATIAL)

    def reset_history(self):
        """Clear history buffer (call between videos)."""
        self._history.clear()

    @torch.no_grad()
    def extract(
        self,
        frame:     np.ndarray,        # HxWx3 BGR
        boxes:     torch.Tensor,      # (N, 4) xyxy
        track_ids: Optional[torch.Tensor] = None,  # (N,) or None
    ) -> torch.Tensor:                # (N, d_model)
        """
        Extract ST Re-ID features for all detected objects.

        Args:
            frame:     raw frame BGR numpy
            boxes:     detection boxes xyxy
            track_ids: existing track IDs (for history lookup)

        Returns:
            features: (N, d_model) L2-normalized ST Re-ID features
        """
        N = len(boxes)
        if N == 0:
            return torch.zeros((0, self.d_model), device=self.device)

        # ── Step 1: Crop & preprocess ──────────────────────────────────────────
        frame_rgb = frame[:, :, ::-1].copy()   # BGR → RGB
        H, W = frame_rgb.shape[:2]

        crops = []
        for box in boxes.cpu():
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(W, int(x2)); y2 = min(H, int(y2))
            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                crop = frame_rgb[y1:y2, x1:x2]
            crops.append(self.transform(crop))

        batch = torch.stack(crops).to(self.device)  # (N, 3, 224, 224)

        # ── Step 2: Spatial features (ViT) ────────────────────────────────────
        cls_tokens = self._extract_spatial(batch)   # (N, D_SPATIAL)

        # ── Step 3: Build temporal sequences ──────────────────────────────────
        temporal_seq = self._get_history(
            track_ids, cls_tokens)                  # (N, T, D_SPATIAL)

        # ── Step 4: Temporal encoding ──────────────────────────────────────────
        temporal_out = self.temporal_encoder(
            temporal_seq)                           # (N, T, d_model)
        temporal_feat = temporal_out[:, -1, :]      # take current frame token

        # ── Step 5: Spatial projection ─────────────────────────────────────────
        spatial_feat = self.spatial_proj(cls_tokens)  # (N, d_model)

        # ── Step 6: ST Fusion ──────────────────────────────────────────────────
        fused = self.fusion(spatial_feat, temporal_feat)  # (N, d_model)

        # ── Step 7: L2 normalize ───────────────────────────────────────────────
        features = F.normalize(fused, dim=1)        # (N, d_model)

        return features


# ── Training loss for Re-ID ────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Batch-hard triplet loss for Re-ID training.
    For each anchor, find hardest positive and hardest negative.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,   # (N, d)
        labels:   torch.Tensor,   # (N,) track IDs
    ) -> torch.Tensor:
        N = features.shape[0]

        # Pairwise L2 distance
        dist = torch.cdist(features, features, p=2)  # (N, N)

        # Masks
        labels_eq  = labels[:, None] == labels[None, :]  # (N, N)
        labels_neq = ~labels_eq

        # For each anchor: hardest positive (max dist among same ID)
        pos_dist = (dist * labels_eq.float()).max(dim=1)[0]

        # For each anchor: hardest negative (min dist among diff ID)
        neg_dist = dist.clone()
        neg_dist[labels_eq] = float('inf')
        neg_dist = neg_dist.min(dim=1)[0]

        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        return loss


class STReIDLoss(nn.Module):
    """
    Combined loss for ST Re-ID training:
      - Triplet loss (metric learning)
      - Cross-entropy ID loss (classification)
    """

    def __init__(self, n_classes: int, d_model: int = 256, margin: float = 0.3):
        super().__init__()
        self.triplet = TripletLoss(margin=margin)
        self.classifier = nn.Linear(d_model, n_classes)
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(
        self,
        features: torch.Tensor,   # (N, d_model)
        labels:   torch.Tensor,   # (N,) track IDs as class indices
    ) -> torch.Tensor:
        trip_loss = self.triplet(features, labels)
        logits    = self.classifier(features)
        ce_loss   = self.ce(logits, labels)
        return trip_loss + ce_loss
