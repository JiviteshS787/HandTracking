import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, Point3, Vec4
import simplepbr

loadPrcFileData("", "notify-level-glgsg debug")

HAND_MODEL_PATH = "hand_landmarker.task"
PIXEL_TO_WORLD = 0.01  # Tune this: larger = model moves faster per pixel

latest_results = None


# =========================
# CALLBACK
# =========================
def render_callback(result, _, __):
    global latest_results
    latest_results = result


# =========================
# PINCH TRACKER
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

        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

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
# CV / MEDIAPIPE HELPERS
# =========================
def setup_detector():
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
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


def send_frame_to_mediapipe(detector, img):
    timestamp_ms = int(time.time() * 1000)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    detector.detect_async(mp_image, timestamp_ms)


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
            cv2.putText(img, "PINCH",
                        (pinch_data["x"] + 15, pinch_data["y"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return pinch_data


# =========================
# PANDA3D APP
# =========================
class App(ShowBase):
    def __init__(self):
        super().__init__()

        # ---------- BACKGROUND + PBR ----------
        self.setBackgroundColor(Vec4(0.00, 0.00, 0.00, 1))
        self.pipeline = simplepbr.init()

        # ---------- LOAD MODEL ----------
        self.model = self.loader.loadModel("models/Hologram3.glb")
        self.model.reparentTo(self.render)
        self.model.setScale(1.0)
        self.model.setPos(0, 15, 0)

        # ---------- CAMERA ----------
        self.disableMouse()

        self.camera_target = Point3(0, 15, 0)
        self.camera_distance = 25.0
        self.camera_heading = 0.0
        self.camera_pitch = -10.0

        self.rotate_sensitivity = 50.0
        self.pan_sensitivity = 10.0
        self.wheel_zoom_sensitivity = 1.5

        self.accept("wheel_up", self.handle_wheel_zoom, [-self.wheel_zoom_sensitivity])
        self.accept("wheel_down", self.handle_wheel_zoom, [self.wheel_zoom_sensitivity])

        self.last_mouse_pos = (0, 0)
        self.update_camera()

        self.taskMgr.add(self.control_camera_task, "control_camera_task")

        # ---------- HAND TRACKING SETUP ----------
        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
        self.cap = setup_camera()
        self.detector = setup_detector()
        self.pinch_tracker = PinchTracker(pinch_threshold=0.05)
        self.prev_time = time.time()

        # Runs every frame alongside the camera task
        self.taskMgr.add(self.hand_tracking_task, "hand_tracking_task")

    # ---------- HAND TRACKING TASK ----------
    def hand_tracking_task(self, task):
        success, img = self.cap.read()
        if not success:
            return task.cont

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        send_frame_to_mediapipe(self.detector, img)
        pinch_data = render_hand(img, latest_results, w, h, self.pinch_tracker)

        if pinch_data:
            # dx_frame: pixels moved left/right since last frame → model X axis
            # dy_frame: pixels moved up/down since last frame → model Z axis (inverted)
            self.model.setX(self.model.getX() + pinch_data["dx_frame"] * PIXEL_TO_WORLD)
            self.model.setZ(self.model.getZ() - pinch_data["dy_frame"] * PIXEL_TO_WORLD)

        # FPS overlay on the CV window
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)  # Non-blocking — just pumps the OpenCV window event queue

        return task.cont

    # ---------- CAMERA METHODS (unchanged) ----------
    def update_camera(self):
        self.camera_pitch = max(-85.0, min(85.0, self.camera_pitch))
        self.camera_distance = max(1.5, self.camera_distance)
        self.camera.setPos(self.camera_target)
        self.camera.setHpr(self.camera_heading, self.camera_pitch, 0)
        self.camera.setPos(self.camera, 0, -self.camera_distance, 0)

    def handle_wheel_zoom(self, amount):
        self.camera_distance += amount
        self.update_camera()

    def control_camera_task(self, task):
        if self.mouseWatcherNode.hasMouse():
            m_pos = self.mouseWatcherNode.getMouse()
            x, y = m_pos.getX(), m_pos.getY()

            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]

            if self.mouseWatcherNode.isButtonDown("mouse1"):
                self.camera_heading -= dx * self.rotate_sensitivity
                self.camera_pitch += dy * self.rotate_sensitivity
                self.update_camera()

            elif self.mouseWatcherNode.isButtonDown("mouse2"):
                self.camera_target.setX(self.camera_target.getX() - (dx * self.pan_sensitivity))
                self.camera_target.setZ(self.camera_target.getZ() - (dy * self.pan_sensitivity))
                self.update_camera()

            self.last_mouse_pos = (x, y)

        return task.cont


# =========================
# ENTRY POINT
# =========================
app = App()
app.run()