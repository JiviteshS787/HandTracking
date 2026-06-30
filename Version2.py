import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

MODEL_PATH = "hand_landmarker.task"

# Global variable to hold results from the async callback
latest_results = None

def render_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_results
    latest_results = result

# 1. Setup MediaPipe options for Live Stream
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
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# 2. Maximize Field of View (Using standard HD wide aspect ratio)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Warm-up
for _ in range(5):
    cap.read()

# 3. Setup OpenCV window to handle scaling/fullscreen properly
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL) 

prev_time = time.time()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # MediaPipe Live Stream mode requires a monotonically increasing millisecond timestamp
    timestamp_ms = int(time.time() * 1000)
    
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    
    # Async call: sends frame and returns immediately (no blocking!)
    detector.detect_async(mp_image, timestamp_ms)

    render_hand(img, h, w, )

    # Use the most recent async results if they exist
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
                cv2.line(img,
                         (int(a.x * w), int(a.y * h)),
                         (int(b.x * w), int(b.y * h)),
                         (0, 255, 0), 2)
                
            # ========================================================
            # 🤏 PINCH GESTURE LOGIC (For your Panda3D integration)
            # Landmark 4 = Thumb Tip, Landmark 8 = Index Tip
            # ========================================================
            thumb_tip = hand[4]
            index_tip = hand[8]
            
            # Calculate distance between tips (normalized coordinates 0.0 - 1.0)
            distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            
            # Pinch threshold (adjust as needed based on testing)
            if distance < 0.05:
                # Calculate pinch center coordinate
                pinch_x = int(((thumb_tip.x + index_tip.x) / 2) * w)
                pinch_y = int(((thumb_tip.y + index_tip.y) / 2) * h)
                
                cv2.circle(img, (pinch_x, pinch_y), 12, (0, 0, 255), -1)
                cv2.putText(img, "PINCH / MOUSE DOWN", (pinch_x + 15, pinch_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # TODO: send `pinch_x`, `pinch_y` or your calculated dx, dy to Panda3D here!

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
    elif key == ord('f') or key == ord('F'):  # Toggle fullscreen with 'F' key
        # Check if currently fullscreen, if so change to normal, else change to fullscreen
        is_fullscreen = cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
        if is_fullscreen:
            cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap.release()
cv2.destroyAllWindows()
detector.close()