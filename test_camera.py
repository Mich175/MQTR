import cv2
import yaml
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gtr_tracker import GTRTracker

# 统一从配置文件读取
cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))

tracker = GTRTracker.from_config(cfg)

cap = cv2.VideoCapture(0)
print("按 Q 或 ESC 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = tracker.update(frame)
        for r in results:
            x1, y1, x2, y2 = map(int, r['box'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{r['track_id']}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    cv2.imshow('GTR Tracker', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("退出成功！")