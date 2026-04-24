"""
eval.py
Evaluation on MOT17 / DanceTrack.

Usage:
  python eval.py --dataset mot17
  python eval.py --dataset dancetrack
  python eval.py --dataset mot17 --split test
  python eval.py --dataset mot17 --checkpoint checkpoints/best_reid.pth
"""

import argparse
import yaml
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mot_dataset import build_tracking_dataset
from models.gtr_tracker import GTRTracker
from utils.metrics import write_mot_results, compute_basic_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--dataset",    default="mot17",
                        choices=["mot17", "dancetrack"])
    parser.add_argument("--split",      default="val",
                        choices=["val", "test", "train"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to Re-ID checkpoint")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    print(f"\n{'='*50}")
    print(f"GTR-PyTorch Evaluation")
    print(f"{'='*50}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Split:    {args.split}")
    print(f"  Strategy: {cfg['association']['query_strategy']}")
    print(f"  Window:   {cfg['tracker']['window_size']}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*50}\n")

    # 初始화 tracker
    tracker = GTRTracker.from_config(cfg)

    # 加载 Re-ID checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        tracker.load_reid_checkpoint(args.checkpoint)

    # 加载数据集
    tracking_data = build_tracking_dataset(
        cfg, dataset_name=args.dataset, split=args.split)

    output_dir = cfg["eval"]["output_dir"]
    all_motas  = []

    for seq in tracking_data.sequences:
        tracker.reset()   # ← reset 同时清空 Re-ID history
        all_results = []

        print(f"  Tracking {seq.seq_name} ({len(seq)} frames)...")

        for i in range(len(seq)):
            frame = seq.get_frame(i)
            if frame is None:
                all_results.append([])
                continue
            results = tracker.update(frame)
            all_results.append(results)

        # 保存结果
        result_path = write_mot_results(
            all_results,
            os.path.join(output_dir, args.dataset),
            seq.seq_name,
        )

        # 计算 MOTA
        if seq.gt_path.exists() and args.split != "test":
            metrics = compute_basic_metrics(
                str(seq.gt_path), result_path)
            all_motas.append(metrics["MOTA"])
            print(f"    MOTA: {metrics['MOTA']:.1f}  "
                  f"FP: {metrics['FP']}  "
                  f"FN: {metrics['FN']}  "
                  f"IDSW: {metrics['IDSW']}")

    print(f"\n{'='*50}")
    if all_motas:
        avg = sum(all_motas) / len(all_motas)
        print(f"  Avg MOTA: {avg:.1f}")
    print(f"  Results → {output_dir}/{args.dataset}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()