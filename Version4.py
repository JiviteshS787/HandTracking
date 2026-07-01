import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

MODEL_PATH = "hand_landmarker.task"

latest_results = None


# =========================
# CALLBACK
# =========================
def render_callback(result, _, __):
    global latest_results
    latest_results = result


# =========================
# PINCH TRACKER CLASS
# =========================
class PinchTracker:
    def __init__(self, pinch_threshold=0.05):
        self.threshold = pinch_threshold
        self.isPinching = False
        self.startX = None
        self.startY = None
        self.lastX = None
        self.lastY = None

    def reset(self):
        self.isPinching = False
        self.startX = self.startY = None
        self.lastX = self.lastY = None

    def update(self, thumb_tip, index_tip, w, h):
        if thumb_tip is None or index_tip is None:
            self.reset()
            return None

        distance = ((thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2) ** 0.5

        pinch_x = int(((thumb_tip.x + index_tip.x) / 2) * w)
        pinch_y = int(((thumb_tip.y + index_tip.y) / 2) * h)

        if distance < self.threshold:
            if not self.isPinching:
                self.isPinching = True
                self.startX, self.lastX = pinch_x, pinch_x
                self.startY, self.lastY = pinch_y, pinch_y

            dx_frame = pinch_x - self.lastX
            dy_frame = pinch_y - self.lastY
            dx_total = pinch_x - self.startX
            dy_total = pinch_y - self.startY

            self.lastX, self.lastY = pinch_x, pinch_y

            return {
                "active": True,
                "x": pinch_x, "y": pinch_y,
                "dx_frame": dx_frame, "dy_frame": dy_frame,
                "dx_total": dx_total, "dy_total": dy_total,
            }
        else:
            self.reset()
            return None


# =========================
# SETUP FUNCTIONS
# =========================
def setup_detector():
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
    return vision.HandLandmarker.create_from_options(options)


def setup_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(5):
        cap.read()
    return cap


def setup_window():
    cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)


# =========================
# DETECTION
# =========================
def send_frame_to_mediapipe(detector, img):
    timestamp_ms = int(time.time() * 1000)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    detector.detect_async(mp_image, timestamp_ms)


# =========================
# RENDERING + GESTURES
# =========================
def render_hand(img, results, w, h, pinch_tracker):
    if not results or not results.hand_landmarks:
        pinch_tracker.reset()
        return None

    pinch_data = None

    for hand in results.hand_landmarks:
        for idx, lm in enumerate(hand):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), -1)
            cv2.putText(img, str(idx), (cx, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        for conn in connections:
            a = hand[conn.start]
            b = hand[conn.end]
            cv2.line(img,
                     (int(a.x * w), int(a.y * h)),
                     (int(b.x * w), int(b.y * h)),
                     (0, 255, 0), 2)

        thumb_tip = hand[4]
        index_tip = hand[8]
        pinch_data = pinch_tracker.update(thumb_tip, index_tip, w, h)

        if pinch_data:
            cv2.circle(img, (pinch_data["x"], pinch_data["y"]), 12, (0, 0, 255), -1)
            cv2.putText(img, "PINCH", (pinch_data["x"] + 15, pinch_data["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return pinch_data


# =========================
# MAIN LOOP
# =========================
def main():
    detector = setup_detector()
    cap = setup_camera()
    setup_window()
    pinch_tracker = PinchTracker(pinch_threshold=0.05)

    prev_time = time.time()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        send_frame_to_mediapipe(detector, img)
        pinch_data = render_hand(img, latest_results, w, h, pinch_tracker)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking", img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('f'), ord('F')):
            is_fullscreen = cv2.getWindowProperty(
                "Hand Tracking", cv2.WND_PROP_FULLSCREEN
            ) == cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(
                "Hand Tracking",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL if is_fullscreen else cv2.WINDOW_FULLSCREEN
            )

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()