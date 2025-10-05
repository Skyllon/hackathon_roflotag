import cv2
import torch
from ultralytics import YOLO

# Load YOLO11n-pose model
model = YOLO("../models/yolo11n-pose.pt")

input_video_path = "input.mp4"
output_video_path = "output.mp4"

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Не удалось открыть видеофайл")
    exit()

# Get initial params of input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Write proccessed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

CONFIDENCE_THRESHOLD = 0.5

cv2.namedWindow("YOLO-Pose", cv2.WINDOW_NORMAL) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        if result.keypoints is not None:
            keypoints_data = result.keypoints

            if hasattr(keypoints_data, 'xy') and hasattr(keypoints_data, 'conf'):
                try:
                    keypoints = keypoints_data.xy.cpu().numpy()
                    conf = keypoints_data.conf.cpu().numpy() 
                except AttributeError:
                    print("Keypoints data is not in expected format. Skipping this frame.")
                    continue
            else:
                print("Keypoints do not have expected attributes. Skipping this frame.")
                continue

            for kp, cf in zip(keypoints, conf):
                if len(kp) != 17:
                    continue

                valid_kp = [
                    (int(x), int(y)) if c > CONFIDENCE_THRESHOLD else None
                    for (x, y), c in zip(kp, cf)
                ]

                for point in valid_kp:
                    if point:
                        cv2.circle(frame, point, 5, (0, 255, 0), -1)

                skeleton = [
                    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12),
                    (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (0, 1), (1, 2), (2, 3), (3, 4)
                ]

                for (i, j) in skeleton:
                    if valid_kp[i] and valid_kp[j]:
                        cv2.line(frame, valid_kp[i], valid_kp[j], (255, 0, 0), 2)

    out.write(frame)

    cv2.imshow("YOLO-Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Обработанное видео сохранено в {output_video_path}")
