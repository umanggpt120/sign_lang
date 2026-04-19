import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import numpy as np


base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # Thumb
    (0,5),(5,6),(6,7),(7,8),          # Index
    (0,9),(9,10),(10,11),(11,12),     # Middle
    (0,13),(13,14),(14,15),(15,16),   # Ring
    (0,17),(17,18),(18,19),(19,20),   # Pinky
    (5,9),(9,13),(13,17)              # Palm
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks):

            # show left/right label
            label = result.handedness[i][0].category_name
            x0, y0 = int(hand[0].x * w), int(hand[0].y * h)
            cv2.putText(frame, label, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # draw points
            for lm in hand:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0,255,0), -1)

            # draw connections
            for s, e in HAND_CONNECTIONS:
                x1, y1 = int(hand[s].x * w), int(hand[s].y * h)
                x2, y2 = int(hand[e].x * w), int(hand[e].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    count = len(result.hand_landmarks) if result.hand_landmarks else 0
    cv2.putText(frame, f"Hands: {count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0) if count else (0,0,255), 2)

    cv2.imshow("Hand Detection - Phase 1", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()