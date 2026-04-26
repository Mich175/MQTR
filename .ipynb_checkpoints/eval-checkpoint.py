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

    print(f"\n{'='*60}")
    print(f"GTR-PyTorch Evaluation")
    print(f"{'='*60}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Split:      {args.split}")
    print(f"  Strategy:   {cfg['association']['query_strategy']}")
    print(f"  Window:     {cfg['tracker']['window_size']}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")

    # 初始化 tracker
    tracker = GTRTracker.from_config(cfg)

    # 加载 Re-ID checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        tracker.load_reid_checkpoint(args.checkpoint)
        print(f"[Eval] Loaded checkpoint: {args.checkpoint}\n")

    # 加载数据集
    tracking_data = build_tracking_dataset(
        cfg, dataset_name=args.dataset, split=args.split)

    output_dir = cfg["eval"]["output_dir"]

    # GT 根目录
    gt_root = cfg["data"]["mot17_root"]  # data/MOT17

    all_metrics = []

    for seq in tracking_data.sequences:
        tracker.reset()
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

        # 计算指标
        if seq.gt_path.exists() and args.split != "test":

            try:
                import trackeval
                from trackeval import Evaluator
                from trackeval.datasets import MotChallenge2DBox
                from trackeval.metrics import HOTA, CLEAR, Identity
                import numpy as np

                # GT 文件夹指向 data/MOT17/train
                gt_folder = os.path.join(gt_root, "train")

                dataset_config = {
                    'GT_FOLDER':          gt_folder,
                    'TRACKERS_FOLDER': output_dir,
                    'OUTPUT_FOLDER':      None,
                    'TRACKERS_TO_EVAL':   None,
                    'CLASSES_TO_EVAL':    ['pedestrian'],
                    'BENCHMARK':          'MOT17',
                    'SPLIT_TO_EVAL':      'train',
                    'INPUT_AS_ZIP':       False,
                    'PRINT_CONFIG':       False,
                    'DO_PREPROC':         False,
                    'TRACKER_SUB_FOLDER': '',
                    'OUTPUT_SUB_FOLDER':  '',
                    'SEQMAP_FILE':        None,
                    'SEQ_INFO':           {seq.seq_name: None},
                    'GT_LOC_FORMAT':      '{gt_folder}/{seq}/gt/gt.txt',
                    'SKIP_SPLIT_FOL':     True,
                }

                eval_config = {
                    'USE_PARALLEL':         False,
                    'NUM_PARALLEL_CORES':   1,
                    'BREAK_ON_ERROR':       False,
                    'PRINT_RESULTS':        False,
                    'PRINT_ONLY_COMBINED':  False,
                    'PRINT_CONFIG':         False,
                    'TIME_PROGRESS':        False,
                    'OUTPUT_SUMMARY':       False,
                    'OUTPUT_EMPTY_CLASSES': False,
                    'OUTPUT_DETAILED':      False,
                    'PLOT_CURVES':          False,
                }

                evaluator    = Evaluator(eval_config)
                dataset      = MotChallenge2DBox(dataset_config)
                metrics_list = [HOTA(), CLEAR(), Identity()]
                results, _   = evaluator.evaluate([dataset], metrics_list)

                tracker_name = list(results['MotChallenge2DBox'].keys())[0]
                res = results['MotChallenge2DBox'][tracker_name]\
                              [seq.seq_name]['pedestrian']

                hota = float(np.mean(res['HOTA']['HOTA'])) * 100
                assa = float(np.mean(res['HOTA']['AssA'])) * 100
                deta = float(np.mean(res['HOTA']['DetA'])) * 100
                mota = float(res['CLEAR']['MOTA'])          * 100
                idf1 = float(res['Identity']['IDF1'])       * 100
                idsw = int(res['CLEAR']['IDSW'])

                m = {
                    'HOTA': round(hota, 2),
                    'AssA': round(assa, 2),
                    'DetA': round(deta, 2),
                    'MOTA': round(mota, 2),
                    'IDF1': round(idf1, 2),
                    'IDSW': idsw,
                }
                all_metrics.append(m)

                print(f"    HOTA: {m['HOTA']:.1f}  "
                      f"IDF1: {m['IDF1']:.1f}  "
                      f"MOTA: {m['MOTA']:.1f}  "
                      f"AssA: {m['AssA']:.1f}  "
                      f"DetA: {m['DetA']:.1f}  "
                      f"IDSW: {m['IDSW']}")

            except Exception as e:
                print(f"    [TrackEval unavailable: {e}] "
                      f"falling back to basic metrics")
                m = compute_basic_metrics(
                    str(seq.gt_path), result_path)
                all_metrics.append(m)
                print(f"    MOTA: {m['MOTA']:.1f}  "
                      f"FP: {m['FP']}  "
                      f"FN: {m['FN']}  "
                      f"IDSW: {m['IDSW']}")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    if all_metrics:
        keys = ['HOTA', 'IDF1', 'MOTA', 'AssA', 'DetA', 'IDSW']
        for k in keys:
            if k in all_metrics[0]:
                vals = [m[k] for m in all_metrics]
                avg  = sum(vals) / len(vals)
                print(f"  Avg {k:>4}: {avg:.2f}")
    print(f"\n  Results → {output_dir}/{args.dataset}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()