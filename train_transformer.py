"""
train_transformer.py
训练 GTRTransformer 用于多目标跟踪关联。
"""

import os
import sys
import argparse
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tracker.transformer import GTRTransformer
from tracker.st_reid_extractor import STReIDExtractor


def load_mot_gt(gt_path):
    data = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            tid   = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf  = int(parts[6])
            cls   = int(parts[7]) if len(parts) > 7 else 1
            if conf != 1 or cls != 1:
                continue
            data[frame].append([x, y, x + w, y + h, tid])
    return data


def extract_features(args, device):
    import cv2
    reid = STReIDExtractor(
        d_model   = args.d_model,
        n_frames  = args.n_frames,
        freeze_vit= False,
        device    = str(device),
    )
    ckpt  = torch.load(args.reid_ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    reid.load_state_dict(state)
    reid.eval()
    print(f"[Extract] Loaded ReID from {args.reid_ckpt}")

    for seq in sorted(os.listdir(args.data_root)):
        gt_path = os.path.join(args.data_root, seq, 'gt', 'gt.txt')
        img_dir = os.path.join(args.data_root, seq, 'img1')
        out_dir = os.path.join(args.feat_dir, seq)
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(gt_path):
            continue
        gt = load_mot_gt(gt_path)
        reid.reset_history()
        for frame_id in sorted(gt.keys()):
            out_path = os.path.join(out_dir, f"{frame_id:06d}.pt")
            if os.path.exists(out_path):
                continue
            img_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
            if not os.path.exists(img_path):
                continue
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            dets  = gt[frame_id]
            boxes = torch.tensor([[d[0], d[1], d[2], d[3]] for d in dets], dtype=torch.float32)
            if len(boxes) == 0:
                torch.save(torch.zeros(0, args.d_model), out_path)
                continue
            with torch.no_grad():
                feats = reid.extract(frame, boxes, track_ids=None)
            torch.save(feats.cpu(), out_path)
        print(f"[Extract] {seq} done")


class MOTWindowDataset(Dataset):
    def __init__(self, data_root, feat_dir, window_size=5, val_seq=None, is_val=False):
        self.feat_dir    = feat_dir
        self.window_size = window_size
        self.samples     = []
        for seq in sorted(os.listdir(data_root)):
            if is_val and seq != val_seq:
                continue
            if not is_val and seq == val_seq:
                continue
            gt_path = os.path.join(data_root, seq, 'gt', 'gt.txt')
            if not os.path.exists(gt_path):
                continue
            gt     = load_mot_gt(gt_path)
            frames = sorted(gt.keys())
            if len(frames) < window_size:
                continue
            for i in range(len(frames) - window_size + 1):
                win  = frames[i: i + window_size]
                tids = {d[4] for f in win for d in gt[f]}
                if len(tids) >= 2:
                    self.samples.append((seq, win, gt))
        print(f"[Dataset] {'val' if is_val else 'train'}: {len(self.samples)} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, win_frames, gt = self.samples[idx]
        all_feats, all_tids, frame_sizes = [], [], []
        for f in win_frames:
            feat_path = os.path.join(self.feat_dir, seq, f"{f:06d}.pt")
            dets = gt[f]
            n    = len(dets)
            if os.path.exists(feat_path):
                feats = torch.load(feat_path, weights_only=False)
                n     = min(len(feats), n)
                feats = feats[:n]
            else:
                feats = torch.zeros(n, 256)
            tids = torch.tensor([d[4] for d in dets[:n]], dtype=torch.long)
            all_feats.append(feats)
            all_tids.append(tids)
            frame_sizes.append(n)
        return {"feats": all_feats, "tids": all_tids, "frame_sizes": frame_sizes}


def collate_fn(batch):
    return batch


def asso_loss(logits, target, pos_weight=10.0):
    weight = torch.ones_like(target)
    weight[target == 1] = pos_weight
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def run_epoch(model, loader, optimizer, device, args, train=True):
    model.train() if train else model.eval()
    total_loss, total_acc, n = 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            for sample in batch:
                feats, tids, frame_sizes = sample["feats"], sample["tids"], sample["frame_sizes"]
                T   = len(feats)
                n_t = frame_sizes
                all_feats = torch.cat(feats, dim=0).unsqueeze(0).to(device)
                all_tids  = torch.cat(tids,  dim=0).to(device)
                if all_feats.shape[1] == 0:
                    continue
                last_t  = T - 1
                k_start = sum(n_t[:last_t])
                k_end   = sum(n_t[:last_t + 1])
                k_inds  = list(range(k_start, k_end))
                if not k_inds:
                    continue
                q_feats    = all_feats[:, k_inds, :]
                tids_query = all_tids[k_inds]

                logits = model(all_feats, q_feats)
                target = (tids_query[:, None] == all_tids[None, :]).float()
                loss   = asso_loss(logits, target, args.pos_weight)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # 屏蔽 query 帧自身，只在参考帧里计算 top-3 acc
                ref_mask = torch.ones(all_feats.shape[1], dtype=torch.bool, device=device)
                for idx in k_inds:
                    ref_mask[idx] = False
                masked_logits = logits.clone()
                masked_logits[:, ~ref_mask] = -1e9

                topk      = min(3, masked_logits.shape[1])
                topk_inds = masked_logits.topk(topk, dim=1).indices
                correct   = 0
                for qi in range(len(tids_query)):
                    tid       = tids_query[qi].item()
                    pred_tids = all_tids[topk_inds[qi]]
                    if tid in pred_tids:
                        correct += 1
                acc = correct / max(len(tids_query), 1)

                total_loss += loss.item()
                total_acc  += acc
                n          += 1

    return total_loss / max(n, 1), total_acc / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    default="/root/MQTR/data/MOT17/train")
    parser.add_argument("--feat_dir",     default="/root/MQTR/mot17_feats")
    parser.add_argument("--reid_ckpt",    default="/root/MQTR/checkpoints/best_reid.pth")
    parser.add_argument("--save_dir",     default="/root/MQTR/checkpoints")
    parser.add_argument("--resume",       default="")
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--n_frames",     type=int,   default=4)
    parser.add_argument("--n_heads",      type=int,   default=8)
    parser.add_argument("--n_enc",        type=int,   default=6)
    parser.add_argument("--n_dec",        type=int,   default=6)
    parser.add_argument("--dim_ff",       type=int,   default=1024)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--window_size",  type=int,   default=5)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight",   type=float, default=10.0)
    parser.add_argument("--val_seq",      default="MOT17-02-SDP")
    parser.add_argument("--extract_only", action="store_true")
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] device: {device}")

    flag = os.path.join(args.feat_dir, ".done")
    if not os.path.exists(flag):
        print("[Info] Extracting features...")
        os.makedirs(args.feat_dir, exist_ok=True)
        extract_features(args, device)
        open(flag, 'w').close()
    else:
        print("[Info] Features already extracted.")

    if args.extract_only:
        return

    train_ds = MOTWindowDataset(args.data_root, args.feat_dir, args.window_size,
                                val_seq=args.val_seq, is_val=False)
    val_ds   = MOTWindowDataset(args.data_root, args.feat_dir, args.window_size,
                                val_seq=args.val_seq, is_val=True)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)

    model = GTRTransformer(
        d_model=args.d_model, n_heads=args.n_heads,
        n_enc=args.n_enc, n_dec=args.n_dec,
        dim_ff=args.dim_ff, dropout=args.dropout,
    ).to(device)

    start_epoch, best_acc = 1, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        print(f"[Info] Resumed from epoch {start_epoch - 1}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, device, args, train=True)
        scheduler.step()
        print(f"[Epoch {epoch:03d}] loss={tr_loss:.4f}  top3_acc={tr_acc:.3f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            vl_loss, vl_acc = run_epoch(model, val_loader, None, device, args, train=False)
            print(f"           val_loss={vl_loss:.4f}  val_top3_acc={vl_acc:.3f}")
            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save({"epoch": epoch, "model": model.state_dict(), "best_acc": best_acc},
                           os.path.join(args.save_dir, "best_transformer.pth"))
                print(f"           Best saved (top3_acc={best_acc:.3f})")
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_acc": best_acc},
                       os.path.join(args.save_dir, f"transformer_epoch{epoch:03d}.pth"))

    print(f"[Done] best val top3_acc: {best_acc:.3f}")


if __name__ == "__main__":
    main()