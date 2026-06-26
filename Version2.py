import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0) # 0 for default camera, 1 for external USB camera
prev_time = 0

while(True):
    success, img = cap.read()
    print(success)
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            # Draw landmarks
            for idx, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), -1)
                cv2.putText(img, str(idx), (cx, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

            # Draw connections
            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            for conn in connections:
                a = hand[conn.start]
                b = hand[conn.end]
                cv2.line(img,
                         (int(a.x * w), int(a.y * h)),
                         (int(b.x * w), int(b.y * h)),
                         (0, 255, 0), 2)

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()