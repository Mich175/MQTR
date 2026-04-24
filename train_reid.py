"""
train_reid.py
Training script for the Spatial-Temporal Re-ID extractor.

Usage:
  python train_reid.py
  python train_reid.py --config configs/default.yaml
  python train_reid.py --dataset mot17
  python train_reid.py --dataset both
  python train_reid.py --resume checkpoints/last_reid.pth
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracker.st_reid_extractor import STReIDExtractor, STReIDLoss
from data.mot_dataset import ReIDDataset


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, loss_fn, loader, optimizer, device, epoch):
    model.train()
    loss_fn.train()

    total_loss = 0.0
    n_batches  = len(loader)

    for i, (crops, labels) in enumerate(loader):
        B, T, C, H, W = crops.shape
        crops  = crops.to(device)
        labels = labels.to(device)

        crops_flat    = crops.view(B * T, C, H, W)
        cls_tokens    = model.vit(crops_flat)
        cls_tokens    = cls_tokens.view(B, T, -1)

        temporal_out  = model.temporal_encoder(cls_tokens)
        temporal_feat = temporal_out[:, -1, :]
        spatial_feat  = model.spatial_proj(cls_tokens[:, -1, :])

        fused    = model.fusion(spatial_feat, temporal_feat)
        features = F.normalize(fused, dim=1)

        loss = loss_fn(features, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(loss_fn.parameters()), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 50 == 0:
            print(f"  Epoch {epoch} [{i+1}/{n_batches}]  "
                  f"loss: {total_loss / (i+1):.4f}")

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--dataset", default="both",
                        choices=["mot17", "dancetrack", "both"])
    parser.add_argument("--resume",  default=None)
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = cfg["train"]["device"]

    # 强制转换为 float 防止 yaml 读成字符串
    lr           = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])

    print(f"\n{'='*50}")
    print(f"GTR-PyTorch Re-ID Training")
    print(f"{'='*50}")
    print(f"  Config:   {args.config}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Epochs:   {cfg['train']['epochs']}")
    print(f"  Batch:    {cfg['train']['batch_size']}")
    print(f"  LR:       {lr}")
    print(f"  Device:   {device}")
    print(f"{'='*50}\n")

    if args.dataset == "mot17":
        cfg["data"]["dancetrack_root"] = None
    elif args.dataset == "dancetrack":
        cfg["data"]["mot17_root"] = None

    roots = []
    if cfg["data"].get("mot17_root"):
        roots.append(cfg["data"]["mot17_root"])
    if cfg["data"].get("dancetrack_root"):
        roots.append(cfg["data"]["dancetrack_root"])

    dataset = ReIDDataset(
        roots=roots,
        n_frames=cfg["data"]["n_frames"],
        min_len=cfg["data"]["min_track_len"],
        img_size=cfg["data"]["img_size"],
        augment=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    print(f"[Data] {dataset.n_classes} identities, "
          f"{len(loader)} batches/epoch\n")

    model = STReIDExtractor(
        d_model=cfg["reid"]["d_model"],
        n_frames=cfg["reid"]["n_frames"],
        freeze_vit=cfg["reid"]["freeze_vit"],
        device=device,
    )

    loss_fn = STReIDLoss(
        n_classes=dataset.n_classes,
        d_model=cfg["reid"]["d_model"],
    ).to(device)

    # 断点续训
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])

        # finetune 时数据集不同，不加载 loss_fn
        if ckpt.get("dataset") == args.dataset:
            loss_fn.load_state_dict(ckpt["loss_fn"])
            start_epoch = ckpt["epoch"] + 1
            print(f"[Resume] 从 epoch {ckpt['epoch']} 继续训练\n")
        else:
            print(f"[Finetune] 新数据集，只加载模型权重\n")
            start_epoch = 1

    # 优化器
    optimizer = optim.AdamW([
        {"params": model.temporal_encoder.parameters(),
         "lr": lr},
        {"params": model.spatial_proj.parameters(),
         "lr": lr},
        {"params": model.fusion.parameters(),
         "lr": lr},
        {"params": loss_fn.parameters(),
         "lr": lr},
        {"params": model.vit.parameters(),
         "lr": lr * 0.1},
    ], weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"],
        eta_min=1e-6,
    )
    for _ in range(start_epoch - 1):
        scheduler.step()

    save_dir  = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):

        if epoch == cfg["train"]["unfreeze_epoch"]:
            model.unfreeze_vit(
                n_last_blocks=cfg["train"]["unfreeze_blocks"])
            print(f"\n[Epoch {epoch}] ViT 最后 "
                  f"{cfg['train']['unfreeze_blocks']} 个 block 解冻！\n")

        avg_loss = train_one_epoch(
            model, loss_fn, loader, optimizer, device, epoch)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}  "
              f"loss: {avg_loss:.4f}  lr: {lr_now:.2e}")

        ckpt = {
            "epoch":   epoch,
            "model":   model.state_dict(),
            "loss_fn": loss_fn.state_dict(),
            "config":  cfg,
            "dataset": args.dataset,
        }

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, f"{save_dir}/best_reid.pth")
            print(f"  ✅ Best checkpoint 保存！loss={best_loss:.4f}")

        torch.save(ckpt, f"{save_dir}/last_reid.pth")

        if epoch % 5 == 0:
            torch.save(ckpt, f"{save_dir}/reid_epoch{epoch}.pth")

    print(f"\n✅ 训练完成！Best loss: {best_loss:.4f}")
    print(f"   Checkpoint: {save_dir}/best_reid.pth")


if __name__ == "__main__":
    main()