"""
association.py
Multi-Query Association with Trio-Weight Score Aggregation.
支持两种模式:
  1. 余弦相似度模式 (use_transformer=False, 默认)
  2. GTR Transformer 模式 (use_transformer=True, 需要训练好的权重)
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrackInstance:
    boxes:         torch.Tensor
    scores:        torch.Tensor
    class_ids:     torch.Tensor
    reid_features: torch.Tensor
    frame_id:      int
    track_ids:     Optional[torch.Tensor] = None

    def __len__(self):
        return len(self.boxes)


class MultiQueryAssociator:

    def __init__(
        self,
        overlap_thresh:        float = 0.5,
        asso_thresh:           float = 0.4,
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
        use_transformer:       bool  = False,  # True = Transformer, False = 余弦相似度
        debug:                 bool  = False,
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
        self.use_transformer       = use_transformer
        self.debug                 = debug

    # ------------------------------------------------------------------ #
    #  Query frame selection
    # ------------------------------------------------------------------ #

    def select_query_frames(self, window_len: int) -> List[int]:
        if self.query_strategy == "first_mid_last":
            if window_len == 1:
                return [0]
            if window_len == 2:
                return [0, 1]
            return list(dict.fromkeys([0, window_len // 2, window_len - 1]))
        elif self.query_strategy == "all":
            return list(range(window_len))
        elif self.query_strategy == "last_k":
            return list(range(max(0, window_len - self.n_queries), window_len))
        elif self.query_strategy == "uniform":
            indices = np.linspace(0, window_len - 1, self.n_queries, dtype=int)
            return list(dict.fromkeys(indices.tolist()))
        else:
            raise ValueError(f"Unknown strategy: {self.query_strategy}")

    # ------------------------------------------------------------------ #
    #  Trio-Weight helpers
    # ------------------------------------------------------------------ #

    def compute_temporal_weight(self, k: int, T: int) -> float:
        if not self.use_temporal_weight:
            return 1.0
        return (k + 1) / T

    def compute_confidence_weight(self, scores: torch.Tensor) -> torch.Tensor:
        if not self.use_confidence_weight:
            return torch.ones(len(scores), device=scores.device)
        return torch.sigmoid(scores * self.conf_weight_scale)

    def compute_motion_weight(
        self, instances: List[TrackInstance], k: int
    ) -> torch.Tensor:
        n_k    = len(instances[k].boxes)
        device = instances[k].boxes.device

        if not self.use_motion_weight or k < 2:
            return torch.ones(n_k, device=device)

        prev1 = instances[k - 1]
        prev2 = instances[k - 2]

        if len(prev1.boxes) == 0 or len(prev2.boxes) == 0:
            return torch.ones(n_k, device=device)

        def center(inst):
            b = inst.boxes
            return (b[:, :2] + b[:, 2:]) / 2

        c_curr  = center(instances[k])
        c_prev1 = center(prev1)
        c_prev2 = center(prev2)

        n = min(n_k, len(c_prev1), len(c_prev2))
        if n == 0:
            return torch.ones(n_k, device=device)

        v1      = c_prev1[:n] - c_prev2[:n]
        v2      = c_curr[:n]  - c_prev1[:n]
        delta_v = ((v2 - v1) ** 2).sum(dim=1)
        w_m     = torch.exp(-delta_v / (self.motion_sigma ** 2))

        if n < n_k:
            w_m = torch.cat([w_m, torch.ones(n_k - n, device=device)])

        return w_m

    # ------------------------------------------------------------------ #
    #  IoU
    # ------------------------------------------------------------------ #

    def _compute_iou(
        self, boxes_a: torch.Tensor, boxes_b: torch.Tensor
    ) -> torch.Tensor:
        ax1, ay1, ax2, ay2 = boxes_a.unbind(1)
        bx1, by1, bx2, by2 = boxes_b.unbind(1)
        ix1   = torch.max(ax1[:, None], bx1[None, :])
        iy1   = torch.max(ay1[:, None], by1[None, :])
        ix2   = torch.min(ax2[:, None], bx2[None, :])
        iy2   = torch.min(ay2[:, None], by2[None, :])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-8)

    def _get_last_box_per_track(
        self,
        ref_ids:       torch.Tensor,
        unique_ids:    torch.Tensor,
        ref_boxes_all: torch.Tensor,
    ) -> torch.Tensor:
        """每个 track 取最后一次出现的 box（修复原来 arange×id_inds 的 bug）。"""
        last_boxes = []
        for uid in unique_ids:
            mask     = (ref_ids == uid).nonzero(as_tuple=True)[0]
            last_idx = mask[-1].item()
            last_boxes.append(ref_boxes_all[last_idx])
        return torch.stack(last_boxes, dim=0)  # (M, 4)

    # ------------------------------------------------------------------ #
    #  Score computation – Transformer 模式
    # ------------------------------------------------------------------ #

    def _score_with_transformer(
        self,
        transformer,
        instances:  List[TrackInstance],
        last_t:     int,
        ref_frames: List[int],
        n_t:        List[int],
        T:          int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """用 GTR Transformer 计算关联分数。返回 (traj_score, unique_ids)。"""
        all_feats_cat = torch.cat(
            [x.reid_features for x in instances], dim=0
        ).unsqueeze(0)  # (1, N, d)

        k_start = sum(n_t[:last_t])
        k_end   = sum(n_t[:last_t + 1])
        k_inds  = list(range(k_start, k_end))

        if not k_inds:
            return None, None

        query_feats = all_feats_cat[:, k_inds, :]       # (1, n_curr, d)
        asso_logits = transformer(all_feats_cat, query_feats)  # (n_curr, N)
        asso_scores = torch.sigmoid(asso_logits)

        # 只取参考帧的列
        ref_inds = []
        for t in ref_frames:
            start = sum(n_t[:t])
            ref_inds.extend(range(start, start + n_t[t]))

        asso_ref = asso_scores[:, ref_inds]  # (n_curr, n_ref)

        # 时间衰减
        if self.decay_time > 0:
            dts = torch.cat([
                instances[t].reid_features.new_full(
                    (len(instances[t]),), T - t - 2)
                for t in ref_frames
            ], dim=0)
            asso_ref = asso_ref * (self.decay_time ** dts[None, :])

        ref_ids    = torch.cat([instances[t].track_ids for t in ref_frames], dim=0)
        unique_ids = torch.unique(ref_ids)

        if len(unique_ids) == 0:
            return None, None

        id_inds    = (unique_ids[None, :] == ref_ids[:, None]).float()
        traj_score = torch.mm(asso_ref, id_inds)  # (n_curr, M)

        return traj_score, unique_ids

    # ------------------------------------------------------------------ #
    #  Score computation – 余弦相似度模式
    # ------------------------------------------------------------------ #

    def _score_with_cosine(
        self,
        instances:  List[TrackInstance],
        last_t:     int,
        ref_frames: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """用余弦相似度计算关联分数。返回 (traj_score, unique_ids)。"""
        curr_feats = instances[last_t].reid_features
        ref_feats  = torch.cat([
            instances[t].reid_features for t in ref_frames
        ], dim=0)

        if self.debug:
            print(f"[DEBUG] curr_feats mean abs: {curr_feats.abs().mean().item():.4f}")
            print(f"[DEBUG] ref_feats  mean abs: {ref_feats.abs().mean().item():.4f}")

        curr_norm  = torch.nn.functional.normalize(curr_feats, dim=1)
        ref_norm   = torch.nn.functional.normalize(ref_feats,  dim=1)
        sim        = torch.mm(curr_norm, ref_norm.t())  # (n_curr, n_ref)

        if self.debug:
            print(f"[DEBUG] sim range: [{sim.min().item():.3f}, {sim.max().item():.3f}]")

        ref_ids    = torch.cat([instances[t].track_ids for t in ref_frames], dim=0)
        unique_ids = torch.unique(ref_ids)
        id_inds    = (unique_ids[None, :] == ref_ids[:, None]).float()
        traj_score = torch.mm(sim, id_inds)  # (n_curr, M)

        return traj_score, unique_ids

    # ------------------------------------------------------------------ #
    #  Main associate
    # ------------------------------------------------------------------ #

    def associate(
        self,
        transformer,
        instances: List[TrackInstance],
        id_count:  int,
    ) -> Tuple[List[TrackInstance], int]:

        T   = len(instances)
        n_t = [len(x.boxes) for x in instances]
        N   = sum(n_t)

        if N == 0:
            return instances, id_count

        last_t = T - 1

        if n_t[last_t] == 0:
            return instances, id_count

        ref_frames = [
            t for t in range(T - 1)
            if instances[t].track_ids is not None
            and len(instances[t].boxes) > 0
        ]

        if not ref_frames:
            n = n_t[last_t]
            instances[last_t].track_ids = torch.arange(
                id_count + 1, id_count + n + 1,
                dtype=torch.long,
                device=instances[last_t].boxes.device,
            )
            id_count += n
            return instances, id_count

        # ---- 计算关联分数 ----
        if self.use_transformer and transformer is not None:
            traj_score, unique_ids = self._score_with_transformer(
                transformer, instances, last_t, ref_frames, n_t, T)
            # Transformer 失败时回退到余弦
            if traj_score is None:
                if self.debug:
                    print("[DEBUG] Transformer failed, falling back to cosine.")
                traj_score, unique_ids = self._score_with_cosine(
                    instances, last_t, ref_frames)
        else:
            traj_score, unique_ids = self._score_with_cosine(
                instances, last_t, ref_frames)

        # ---- IoU boost ----
        if self.with_iou:
            try:
                ref_boxes_all = torch.cat([
                    instances[t].boxes for t in ref_frames
                ], dim=0)
                ref_ids_all = torch.cat([
                    instances[t].track_ids for t in ref_frames
                ], dim=0)
                curr_boxes = instances[last_t].boxes
                last_boxes = self._get_last_box_per_track(
                    ref_ids_all, unique_ids, ref_boxes_all)
                ious       = self._compute_iou(curr_boxes, last_boxes)
                traj_score = torch.max(traj_score, ious)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] IoU boost failed: {e}")

        # ---- Trio-Weight ----
        temporal_w = self.compute_temporal_weight(last_t, T)
        conf_w     = self.compute_confidence_weight(instances[last_t].scores)
        motion_w   = self.compute_motion_weight(instances, last_t)
        combined_w = temporal_w * conf_w * motion_w
        traj_score = traj_score * combined_w[:, None]

        if self.debug:
            print(f"[DEBUG] traj_score range: "
                  f"[{traj_score.min().item():.3f}, {traj_score.max().item():.3f}]")

        # ---- Hungarian matching ----
        score_np         = traj_score.cpu().numpy()
        match_i, match_j = linear_sum_assignment(-score_np)

        assigned = {}
        used_ids = set()

        for i, j in zip(match_i, match_j):
            if score_np[i, j] > self.asso_thresh:
                tid = unique_ids[j].item()
                if tid not in used_ids:
                    assigned[i] = tid
                    used_ids.add(tid)

        if self.debug:
            print(f"[DEBUG] matched {len(assigned)}/{n_t[last_t]} detections, "
                  f"thresh={self.asso_thresh}")

        # ---- 分配 ID ----
        track_ids = []
        for i in range(n_t[last_t]):
            if i in assigned:
                track_ids.append(assigned[i])
            else:
                id_count += 1
                track_ids.append(id_count)

        instances[last_t].track_ids = torch.tensor(
            track_ids, dtype=torch.long,
            device=instances[last_t].boxes.device)

        return instances, id_count