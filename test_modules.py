import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 测试1：检测器
print("测试 YOLODetector...")
from detector.yolo_detector import YOLODetector
import numpy as np
detector = YOLODetector(model_name='weights/yolo11x.pt', device='cuda')
frame = np.zeros((480, 640, 3), dtype=np.uint8)
det = detector.detect(frame, frame_id=0)
print(f"✅ Detection OK: boxes={det.boxes.shape}")

# 测试2：Transformer
print("\n测试 GTRTransformer...")
from tracker.transformer import GTRTransformer
model = GTRTransformer(d_model=256, n_heads=8, n_enc=2, n_dec=2).cuda()
all_feat = torch.randn(1, 20, 256).cuda()
q_feat   = torch.randn(1, 5, 256).cuda()
out = model(all_feat, q_feat)
print(f"✅ Transformer OK: output={out.shape}")

# 测试3：ST Re-ID
print("\n测试 STReIDExtractor...")
from tracker.st_reid_extractor import STReIDExtractor
reid = STReIDExtractor(d_model=256, n_frames=4, device='cuda')
boxes = torch.tensor([[10,10,100,100],[50,50,150,150]], dtype=torch.float32).cuda()
feats = reid.extract(frame, boxes)
print(f"✅ ReID OK: features={feats.shape}")

print("\n🎉 所有模块测试通过！")