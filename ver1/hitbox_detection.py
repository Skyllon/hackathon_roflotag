import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO11n-pose model
model = YOLO("../models/yolo11n-pose.pt")

# Catch camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("YOLO-HitboxWithID", cv2.WINDOW_NORMAL)

persons_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    new_persons = [] 

    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Is person already exists in list
            found_index = None

            for idx, prev_center in enumerate(persons_list):
                if np.linalg.norm(np.array(center) - np.array(prev_center)) < 50:
                    found_index = idx
                    break

            if found_index is None:
                new_persons.append(center)
            else:
                new_persons.append(persons_list[found_index])

    persons_list = new_persons

    for idx, (x1, y1, x2, y2) in enumerate(result.boxes.xyxy.cpu().numpy()):
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.putText(frame, f"ID: {idx+1}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO-Pose", frame)

    if cv2.waitKey(1) & 255 == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
