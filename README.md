# GTR-PyTorch

A clean PyTorch reimplementation of **Global Tracking Transformers (GTR, CVPR 2022)**,
completely removing the detectron2 dependency and replacing the detector with YOLO.

## Key Improvements over Original GTR

| | Original GTR | GTR-PyTorch |
|---|---|---|
| Framework | detectron2 + CenterNet2 | Pure PyTorch |
| Detector | CenterNet2 (locked) | Any YOLO (v8/v9/v11) |
| Query mode | Single frame only | **Multi-frame queries** |
| Installation | Complex (detectron2 + submodules) | `pip install -r requirements.txt` |
| Windows support | ❌ | ✅ |
| Online inference | ❌ (TODO in original) | ✅ |

## Architecture

```
Frame
  ↓
YOLODetector          # detection (replaceable)
  ↓
ReidExtractor         # CNN crop → L2 normalized features
  ↓
Sliding Window        # buffer last N frames
  ↓
GTRTransformer        # Encoder: self-attn across all objects
                      # Decoder: multi-query cross-attn
  ↓
MultiQueryAssociator  # weighted voting + Hungarian matching
  ↓
Track IDs
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Track persons in a video
python track.py --video input.mp4 --output output.mp4 --classes 0

# Use different YOLO model
python track.py --video input.mp4 --yolo yolov8n.pt --output output.mp4

# Multi-query strategies
python track.py --video input.mp4 --query-strategy all       # all frames as queries
python track.py --video input.mp4 --query-strategy last_k --n-queries 3
python track.py --video input.mp4 --query-strategy uniform --n-queries 4

# Adjust window size
python track.py --video input.mp4 --window 8
```

## Multi-Query Strategy

The key innovation over the original GTR:

- **`all`**: every frame in the window acts as a Query (most thorough)
- **`last_k`**: only the most recent k frames (balance of speed/accuracy)
- **`uniform`**: uniformly sampled frames across the window

Scores from multiple queries are aggregated via **time-weighted voting**:
newer frames get higher weights, resolving conflicts via score magnitude.

## Project Structure

```
gtr-pytorch/
├── detector/
│   └── yolo_detector.py      # YOLO wrapper
├── tracker/
│   ├── transformer.py        # GTR Transformer (pure PyTorch)
│   ├── reid_extractor.py     # Re-ID feature extractor
│   └── association.py        # Multi-query association + Hungarian matching
├── models/
│   └── gtr_tracker.py        # Main tracker
├── track.py                  # Inference entry point
└── requirements.txt
```
