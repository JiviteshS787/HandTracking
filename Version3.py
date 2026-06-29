import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

MODEL_PATH = "hand_landmarker.task"

# 1. Initialize Window Immediately & Show Loading Screen
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Read one dummy frame to get dimensions for the splash screen
success, loading_img = cap.read()
if success:
    h, w, _ = loading_img.shape
    # Draw a nice loading background
    cv2.rectangle(loading_img, (0, 0), (w, h), (40, 40, 40), -1)
    cv2.putText(loading_img, "Loading AI Models... Please Wait", (int(w/4), int(h/2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Hand Tracking", loading_img)
    cv2.waitKey(1)  # Force OpenCV to update the window layout

# 2. Setup MediaPipe options for Live Stream
latest_results = None

# Using underscores for unused variables prevents Pylance warnings
def render_callback(result, _, __):
    global latest_results
    latest_results = result

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=render_callback,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# This is the heavy line that causes the 2-4 second delay
detector = vision.HandLandmarker.create_from_options(options)

prev_time = time.time()

# 3. Main Loop
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    timestamp_ms = int(time.time() * 1000)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    
    # Send frame asynchronously
    detector.detect_async(mp_image, timestamp_ms)

    # Render results if available
    if latest_results and latest_results.hand_landmarks:
        for hand in latest_results.hand_landmarks:
            # Draw Landmarks
            for idx, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), -1)
                cv2.putText(img, str(idx), (cx, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

            # Draw Connections
            connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
            for conn in connections:
                a = hand[conn.start]
                b = hand[conn.end]
                cv2.line(img, (int(a.x * w), int(a.y * h)), (int(b.x * w), int(b.y * h)), (0, 255, 0), 2)
                
            # Pinch Gesture Logic
            thumb_tip = hand[4]
            index_tip = hand[8]
            distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            
            if distance < 0.05:
                pinch_x = int(((thumb_tip.x + index_tip.x) / 2) * w)
                pinch_y = int(((thumb_tip.y + index_tip.y) / 2) * h) 
                
                cv2.circle(img, (pinch_x, pinch_y), 12, (0, 0, 255), -1)
                cv2.putText(img, "PINCH / MOUSE DOWN", (pinch_x + 15, pinch_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('f') or key == ord('F'):  # F to toggle fullscreen
        is_fullscreen = cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
        if is_fullscreen:
            cv2.