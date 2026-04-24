"""
association.py
Multi-Query Association with Trio-Weight Score Aggregation.

Key improvements over original GTR:

  1. Multi-Query Tracking (first_mid_last strategy):
     Original GTR uses only the LATEST frame as Query.
     We select THREE representative frames:
       - First frame:  long-term memory
       - Middle frame: medium-term reference
       - Last frame:   most recent observation
     → More robust to missed detections
     → Covers short / medium / long-term temporal context

  2. Trio-Weight Association:
     Original GTR treats all frames equally with no weighting.
     We apply THREE complementary weights:

       a) Temporal weight  w_t = (k+1)/T
          → Newer frames get higher weight
          → Recent observations are more reliable

       b) Confidence weight  w_c = sigmoid(score)
          → High-confidence detections get higher weight
          → Reliable detections drive association

       c) Motion consistency weight  w_m = exp(-Δv²/σ²)
          → Smooth motion gets higher weight
          → Erratic/jumping targets get lower weight

     Final score = f_asso × w_t × w_c × w_m

     → More accurate identity assignment
     → Fewer ID switches
     → Better handling of occlusion and similar appearances
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrackInstance:
    """Represents one frame's worth of tracked objects."""
    boxes:         torch.Tensor           # (n, 4) xyxy
    scores:        torch.Tensor           # (n,)   detection confidence
    class_ids:     torch.Tensor           # (n,)
    reid_features: torch.Tensor           # (n, d)
    frame_id:      int
    track_ids:     Optional[torch.Tensor] = None  # (n,)

    def __len__(self):
        return len(self.boxes)


class MultiQueryAssociator:
    """
    Multi-Query Association with Trio-Weight Score Aggregation.

    Improvement 1 — Multi-Query (first_mid_last):
      Original GTR: k = window_len-1 (latest frame only)
      Ours: k ∈ {first, middle, last} (3 representative frames)

    Improvement 2 — Trio-Weight:
      Final score = f_asso × w_t × w_c × w_m
      w_t: temporal weight    (k+1)/T
      w_c: confidence weight  sigmoid(score)
      w_m: motion weight      exp(-Δv²/σ²)
    """

    def __init__(
        self,
        overlap_thresh:        float = 0.5,
        asso_thresh:           float = 0.3,
        with_iou:              bool  = True,
        decay_time:            float = 0.9,
        max_center_dist:       float = 4.0,
        not_mult_thresh:       bool  = False,
        query_strategy:        str   = "first_mid_last",
        n_queries:             int   = 3,
        use_temporal_weight:   bool  = True,
        use_confidence_weight: bool  = True,
        use_motion_weight:     bool  = True,
        temporal_decay:        float = 0.9,
        conf_weight_scale:     float = 1.0,
        motion_sigma:          float = 50.0,
    ):
        self.overlap_thresh        = overlap_thresh
        self.asso_thresh           = asso_thresh
        self.with_iou              = with_iou
        self.decay_time            = decay_time
        self.max_center_dist       = max_center_dist
        self.not_mult_thresh       = not_mult_thresh
        self.query_strategy        = query_strategy
        self.n_queries             = n_queries
        self.use_temporal_weight   = use_temporal_weight
        self.use_confidence_weight = use_confidence_weight
        self.use_motion_weight     = use_motion_weight
        self.temporal_decay        = temporal_decay
        self.conf_weight_scale     = conf_weight_scale
        self.motion_sigma          = motion_sigma

    # ── Improvement 1: Multi-Query Frame Selection ────────────────────────────

    def select_query_frames(self, window_len: int) -> List[int]:
        """
        Select which frames act as Queries.

        GTR original: always uses frame k = window_len-1 only
        Our improvement: select 3 representative frames

        'first_mid_last': first, middle, last frame (our innovation)
          T=32 → [0, 16, 31]
          T=5  → [0, 2, 4]
          T=2  → [0, 1]
          T=1  → [0]

        'all':    every frame (highest recall, slowest)
        'last_k': only the most recent n_queries frames
        'uniform': evenly sampled frames
        """
        if self.query_strategy == "first_mid_last":
            if window_len == 1:
                return [0]
            if window_len == 2:
                return [0, 1]
            first  = 0
            middle = window_len // 2
            last   = window_len - 1
            return list(dict.fromkeys([first, middle, last]))

        elif self.query_strategy == "all":
            return list(range(window_len))

        elif self.query_strategy == "last_k":
            start = max(0, window_len - self.n_queries)
            return list(range(start, window_len))

        elif self.query_strategy == "uniform":
            indices = np.linspace(0, window_len - 1,
                                  self.n_queries, dtype=int)
            return list(dict.fromkeys(indices.tolist()))

        else:
            raise ValueError(f"Unknown strategy: {self.query_strategy}")

    # ── Improvement 2a: Temporal Weight ──────────────────────────────────────

    def compute_temporal_weight(self, k: int, T: int) -> float:
        """
        Temporal weight: newer frames get higher weight.

        w_t = (k+1) / T
          k=0 (oldest)   → w_t = 1/T   (lowest)
          k=T-1 (newest) → w_t = 1.0   (highest)

        Rationale: recent observations are more reliable
        for identity assignment than old ones.
        """
        if not self.use_temporal_weight:
            return 1.0
        return (k + 1) / T

    # ── Improvement 2b: Confidence Weight ────────────────────────────────────

    def compute_confidence_weight(
        self, scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Confidence weight: high-score detections get higher weight.

        w_c = sigmoid(score × scale)
          score=0.9 → w_c ≈ 0.71  (high weight)
          score=0.3 → w_c ≈ 0.57  (lower weight)

        Rationale: detections with high confidence scores
        are more reliable for association than low-confidence ones.
        """
        if not self.use_confidence_weight:
            return torch.ones(len(scores), device=scores.device)
        return torch.sigmoid(scores * self.conf_weight_scale)

    # ── Improvement 2c: Motion Consistency Weight ─────────────────────────────

    def compute_motion_weight(
        self,
        instances: List[TrackInstance],
        k: int,
    ) -> torch.Tensor:
        """
        Motion consistency weight: w_m = exp(-Δv² / σ²)

        Measures how smoothly the target has been moving.
          Smooth motion  → Δv small → w_m ≈ 1.0  (high weight)
          Erratic motion → Δv large → w_m ≈ 0.0  (low weight)

        Rationale:
          If a target has been moving consistently
          (e.g. walking steadily), the association is
          more reliable than if it suddenly jumped or
          changed direction drastically (e.g. occlusion).

          This is especially useful for DanceTrack where
          dancers make rapid, unpredictable movements.

        Formula:
          v1 = center(k-1) - center(k-2)   previous velocity
          v2 = center(k)   - center(k-1)   current velocity
          Δv = ||v2 - v1||²                velocity change
          w_m = exp(-Δv / σ²)
        """
        n_k    = len(instances[k].boxes)
        device = instances[k].boxes.device

        if not self.use_motion_weight:
            return torch.ones(n_k, device=device)

        # Need at least 2 previous frames
        if k < 2:
            return torch.ones(n_k, device=device)

        prev1 = instances[k - 1]
        prev2 = instances[k - 2]

        if len(prev1.boxes) == 0 or len(prev2.boxes) == 0:
            return torch.ones(n_k, device=device)

        def center(inst):
            b = inst.boxes
            return (b[:, :2] + b[:, 2:]) / 2   # (n, 2)

        c_curr  = center(instances[k])   # (n_k, 2)
        c_prev1 = center(prev1)          # (n_p1, 2)
        c_prev2 = center(prev2)          # (n_p2, 2)

        # Use minimum count to avoid size mismatch
        n = min(n_k, len(c_prev1), len(c_prev2))
        if n == 0:
            return torch.ones(n_k, device=device)

        # Velocity vectors
        v1 = c_prev1[:n] - c_prev2[:n]   # previous velocity (n, 2)
        v2 = c_curr[:n]  - c_prev1[:n]   # current velocity  (n, 2)

        # Velocity change magnitude squared
        delta_v = ((v2 - v1) ** 2).sum(dim=1)   # (n,)

        # Gaussian decay: smooth motion → w_m close to 1
        w_m = torch.exp(
            -delta_v / (self.motion_sigma ** 2))  # (n,)

        # Pad to n_k if needed
        if n < n_k:
            pad = torch.ones(n_k - n, device=device)
            w_m = torch.cat([w_m, pad])

        return w_m

    # ── IoU Helper ────────────────────────────────────────────────────────────

    def _compute_iou(self, boxes_a, boxes_b):
        ax1, ay1, ax2, ay2 = boxes_a.unbind(1)
        bx1, by1, bx2, by2 = boxes_b.unbind(1)
        ix1 = torch.max(ax1[:, None], bx1[None, :])
        iy1 = torch.max(ay1[:, None], by1[None, :])
        ix2 = torch.min(ax2[:, None], bx2[None, :])
        iy2 = torch.min(ay2[:, None], by2[None, :])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-8)

    # ── Single Query Score ────────────────────────────────────────────────────

    def _single_query_score(
        self, transformer, instances, k, n_t, N, T
    ) -> Dict:
        """
        Compute association scores for one query frame k.

        Pipeline:
          all_features → GTR Transformer → asso_scores
          → time decay → track aggregation
          → IoU boost → distance filter
        """
        all_features = torch.cat(
            [x.reid_features for x in instances], dim=0
        ).unsqueeze(0)                            # (1, N, d)

        k_start   = sum(n_t[:k])
        k_end     = sum(n_t[:k + 1])
        k_inds    = list(range(k_start, k_end))
        nonk_inds = [i for i in range(N) if i not in k_inds]

        query_features = all_features[:, k_inds, :]  # (1, n_k, d)

        # GTR Transformer forward
        asso_logits = transformer(all_features, query_features)
        asso_scores = torch.sigmoid(asso_logits)
        asso_nonk   = asso_scores[:, nonk_inds]      # (n_k, Np)

        # Time decay: older frames contribute less
        if self.decay_time > 0 and len(nonk_inds) > 0:
            dts = torch.cat([
                instances[t].reid_features.new_full(
                    (len(instances[t]),), T - t - 2)
                for t in range(T) if t != k
            ], dim=0)
            asso_nonk = asso_nonk * (self.decay_time ** dts[None, :])

        # Get existing track IDs from non-query frames
        ids = torch.cat([
            x.track_ids for t, x in enumerate(instances)
            if t != k and x.track_ids is not None
        ], dim=0)

        unique_ids = torch.unique(ids)
        M          = len(unique_ids)
        id_inds    = (unique_ids[None, :] == ids[:, None]).float()

        # Aggregate per track: (n_k, Np) × (Np, M) → (n_k, M)
        traj_score = torch.mm(asso_nonk, id_inds)

        # IoU boost: spatial overlap bonus
        if self.with_iou and M > 0 and len(nonk_inds) > 0:
            last_inds  = (id_inds * torch.arange(
                len(nonk_inds), device=id_inds.device)[:, None]
            ).max(dim=0)[1]
            nonk_boxes = torch.cat([
                instances[t].boxes for t in range(T) if t != k], dim=0)
            k_boxes    = instances[k].boxes
            last_boxes = nonk_boxes[last_inds]
            last_ious  = self._compute_iou(k_boxes, last_boxes)
            traj_score = torch.max(traj_score, last_ious)

        # Center distance filter
        if self.max_center_dist > 0 and M > 0 and len(nonk_inds) > 0:
            k_boxes    = instances[k].boxes
            nonk_boxes = torch.cat([
                instances[t].boxes for t in range(T) if t != k], dim=0)
            k_ct  = (k_boxes[:, :2] + k_boxes[:, 2:]) / 2
            k_s   = ((k_boxes[:, 2:] - k_boxes[:, :2]) ** 2).sum(1)
            nk_ct = (nonk_boxes[:, :2] + nonk_boxes[:, 2:]) / 2
            dist  = ((k_ct[:, None] - nk_ct[None, :]) ** 2).sum(2)
            norm_dist   = dist / (k_s[:, None] + 1e-8)
            valid       = norm_dist < self.max_center_dist
            valid_track = torch.mm(
                valid.float(), id_inds).clamp(max=1).bool()
            traj_score[~valid_track] = 0

        return {
            "k":          k,
            "k_inds":     k_inds,
            "n_k":        len(k_inds),
            "traj_score": traj_score,
            "unique_ids": unique_ids,
            "ids":        ids,
            "id_inds":    id_inds,
        }

    # ── Main Association Entry Point ──────────────────────────────────────────

    def associate(
        self,
        transformer,
        instances:  List[TrackInstance],
        id_count:   int,
    ) -> Tuple[List[TrackInstance], int]:
        """
        Multi-Query Association with Trio-Weight Voting.

        For each query frame k in {first, middle, last}:
          1. Compute traj_score via GTR Transformer
          2. Apply Trio-Weight:
               w_t = (k+1)/T          temporal weight
               w_c = sigmoid(score)   confidence weight
               w_m = exp(-Δv²/σ²)    motion weight
          3. weighted_score = traj_score × w_t × w_c × w_m
          4. Hungarian matching
          5. Accumulate weighted votes

        Final assignment:
          Each object gets track_id with highest total vote score
          Unmatched objects → new ID
        """
        T   = len(instances)
        n_t = [len(x.boxes) for x in instances]
        N   = sum(n_t)

        if N == 0:
            return instances, id_count

        query_frames = self.select_query_frames(T)

        # vote_table[global_obj_idx][track_id] = accumulated_score
        vote_table: Dict[int, Dict[int, float]] = {}

        for k in query_frames:
            if n_t[k] == 0:
                continue
            nonk_has_ids = all(
                instances[t].track_ids is not None
                for t in range(T) if t != k)
            if not nonk_has_ids:
                continue

            result     = self._single_query_score(
                transformer, instances, k, n_t, N, T)
            traj_score = result["traj_score"]
            unique_ids = result["unique_ids"]
            k_inds     = result["k_inds"]
            id_inds    = result["id_inds"]

            if traj_score.numel() == 0 or len(unique_ids) == 0:
                continue

            # ── Trio-Weight ───────────────────────────────────────────────────
            # w_t: temporal weight → newer frames matter more
            temporal_w = self.compute_temporal_weight(k, T)

            # w_c: confidence weight → reliable detections matter more
            conf_w = self.compute_confidence_weight(
                instances[k].scores)                        # (n_k,)

            # w_m: motion weight → smooth motion matters more
            motion_w = self.compute_motion_weight(
                instances, k)                               # (n_k,)

            # Trio-Weight combination
            combined_w = temporal_w * conf_w * motion_w    # (n_k,)

            weighted_score = traj_score * combined_w[:, None]  # (n_k, M)

            # Hungarian matching
            match_i, match_j = linear_sum_assignment(
                -weighted_score.cpu().numpy())

            for i, j in zip(match_i, match_j):
                thresh = (self.overlap_thresh * id_inds[:, j].sum().item()
                          if not self.not_mult_thresh
                          else self.overlap_thresh)
                if weighted_score[i, j].item() > thresh:
                    global_idx = k_inds[i]
                    tid        = unique_ids[j].item()
                    vote_score = weighted_score[i, j].item()
                    if global_idx not in vote_table:
                        vote_table[global_idx] = {}
                    vote_table[global_idx][tid] = \
                        vote_table[global_idx].get(tid, 0.0) + vote_score

        # ── Final ID assignment via weighted voting ────────────────────────────
        final_id_map:  Dict[int, int] = {}
        used_track_ids = set()

        scored = []
        for global_idx, votes in vote_table.items():
            best_tid   = max(votes, key=votes.get)
            best_score = votes[best_tid]
            scored.append((best_score, global_idx, best_tid))
        scored.sort(reverse=True)

        for score, global_idx, tid in scored:
            if tid not in used_track_ids:
                final_id_map[global_idx] = tid
                used_track_ids.add(tid)

        # Assign IDs
        cumsum = 0
        for t, inst in enumerate(instances):
            track_ids = []
            for local_i in range(n_t[t]):
                global_i = cumsum + local_i
                if global_i in final_id_map:
                    track_ids.append(final_id_map[global_i])
                elif (inst.track_ids is not None and
                      local_i < len(inst.track_ids) and
                      inst.track_ids[local_i].item() >= 0):
                    track_ids.append(inst.track_ids[local_i].item())
                else:
                    id_count += 1
                    track_ids.append(id_count)

            device = inst.reid_features.device
            instances[t].track_ids = torch.tensor(
                track_ids, dtype=torch.long, device=device)
            cumsum += n_t[t]

        return instances, id_count