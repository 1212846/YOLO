import cv2
import time
import numpy as np
import ctypes
from ultralytics import YOLO

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
            pass

user32 = ctypes.windll.user32
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)
print(f"检测到真实分辨率: {screen_w} x {screen_h}")

cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

cv2.resizeWindow("YOLO Detection", screen_w, screen_h)
cv2.moveWindow("YOLO Detection", 0, 0)

cv2.setWindowProperty("YOLO Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("摄像头已打开")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    results = model(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if (x2 - x1) * (y2 - y1) < (w * h * 0.01):
            continue
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = box.conf[0]

        box_w = x2 - x1
        box_h = y2 - y1

        corner_len = max(15, min(box_w, box_h) // 4)

        cv2.line(frame, (x1, y1 + corner_len), (x1, y1), (255, 0, 0), 3)
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 0, 0), 3)

        cv2.line(frame, (x2 - corner_len, y1), (x2, y1), (255, 0, 0), 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (255, 0, 0), 3)

        cv2.line(frame, (x1, y2 - corner_len), (x1, y2), (255, 0, 0), 3)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (255, 0, 0), 3)

        cv2.line(frame, (x2 - corner_len, y2), (x2, y2), (255, 0, 0), 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 0, 0), 3)

        cx, cy = x1 + box_w // 2, y1 + box_h // 2
        cv2.line(frame, (cx - 5, cy), (cx + 5, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 5), (cx, cy + 5), (0, 255, 0), 1)
        
        cv2.line(frame, (w // 2, h // 2), (cx, cy), (0, 255, 0), 1, cv2.LINE_AA)

        label = f"{class_name} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 0), 2)
    cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    black_canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    x_offset = (screen_w - new_w) // 2
    y_offset = (screen_h - new_h) // 2
    black_canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

    cv2.imshow("YOLO Detection", black_canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()