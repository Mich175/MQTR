"""
track.py
Single video inference script.

Usage:
  python track.py --video input.mp4
  python track.py --video input.mp4 --output output.mp4
  python track.py --video input.mp4 --show
  python track.py --video input.mp4 --checkpoint checkpoints/best_reid.pth
  python track.py --config configs/default.yaml --video input.mp4
"""

import argparse
import yaml
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gtr_tracker import GTRTracker


def main():
    parser = argparse.ArgumentParser(description="GTR-PyTorch Inference")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--video",      required=True)
    parser.add_argument("--output",     default=None)
    parser.add_argument("--show",       action="store_true")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to Re-ID checkpoint")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    print(f"\n{'='*50}")
    print(f"GTR-PyTorch Inference")
    print(f"{'='*50}")
    print(f"  Config:     {args.config}")
    print(f"  Video:      {args.video}")
    print(f"  Output:     {args.output}")
    print(f"  Strategy:   {cfg['association']['query_strategy']}")
    print(f"  Window:     {cfg['tracker']['window_size']}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*50}\n")

    tracker = GTRTracker.from_config(cfg)

    results = tracker.track_video(
        video_path=args.video,
        output_path=args.output,
        show=args.show,
        checkpoint_path=args.checkpoint,
    )

    total = sum(len(r) for r in results)
    print(f"\n✅ Done! {len(results)} frames, {total} detections total")


if __name__ == "__main__":
    main()