import cv2
import os
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import numpy as np

# Load model
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

labels = [
    "i","you","we",
    "go","come","eat","drink","like","write","read","play",
    "water","toilet","food","class","teacher","bag"
    ,"please","yes","help","sorry","thank_you"
]

Dataset_path = "dataset"

# Create folders
for label in labels:
    os.makedirs(os.path.join(Dataset_path, label), exist_ok=True)

sequence = []
sequence_length = 30

label_index = 0
current_label = labels[label_index]

sample_count = {label: 0 for label in labels}

recording = False

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    h, w, _ = frame.shape

    if result.hand_landmarks and recording:

        frame_data = []

        for hand_id in range(2):
            if hand_id < len(result.hand_landmarks):
                hand = result.hand_landmarks[hand_id]

                for lm in hand:
                    frame_data.extend([lm.x, lm.y, lm.z])

                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0,255,0), -1)

                for s, e in HAND_CONNECTIONS:
                    x1, y1 = int(hand[s].x * w), int(hand[s].y * h)
                    x2, y2 = int(hand[e].x * w), int(hand[e].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            else:
                frame_data.extend([0] * (21 * 3))

        sequence.append(frame_data)

        if len(sequence) == sequence_length:
            npy_path = os.path.join(
                Dataset_path,
                current_label,
                f"{sample_count[current_label]}.npy"
            )

            np.save(npy_path, np.array(sequence))
            print(f"Saved: {npy_path}")

            sequence = []
            sample_count[current_label] += 1

    #  UI TEXT
    cv2.putText(frame, f"Label: {current_label}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(frame, f"Sample: {sample_count[current_label]}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.putText(frame, "n:next p:prev s:start/stop q:quit",
                (20,h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Show recording status
    status = "RECORDING" if recording else "PAUSED"
    color = (0,255,0) if recording else (0,0,255)
    cv2.putText(frame, status, (w-200,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Dataset Collection - 30 Words", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        label_index = (label_index + 1) % len(labels)
        current_label = labels[label_index]
        sequence = []
        print("Next label:", current_label)

    elif key == ord('p'):
        label_index = (label_index - 1) % len(labels)
        current_label = labels[label_index]
        sequence = []
        print("Previous label:", current_label)

    elif key == ord('s'):
        recording = not recording
        sequence = []
        print("Recording:", recording)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()