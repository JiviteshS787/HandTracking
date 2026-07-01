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
GESTURE_MODEL_PATH = "gesture_recognizer.task"

PIXEL_TO_WORLD = 0.01

latest_results = None
latest_gesture = None


# =========================
# CALLBACKS
# =========================
def render_callback(result, _, __):
    global latest_results
    latest_results = result


def gesture_callback(result, _, __):
    global latest_gesture
    latest_gesture = result


# =========================
# PINCH TRACKER
# =========================
class PinchTracker:
    def __init__(self, pinch_threshold=0.05):
        self.threshold = pinch_threshold
        self.isPinching = False
        self.lastX = None
        self.lastY = None

    def reset(self):
        self.isPinching = False
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
                self.lastX, self.lastY = pinch_x, pinch_y

            dx = pinch_x - self.lastX
            dy = pinch_y - self.lastY

            self.lastX, self.lastY = pinch_x, pinch_y

            return {"dx": dx, "dy": dy}

        self.reset()
        return None


# =========================
# MEDIAPIPE SETUP
# =========================
def setup_hand_detector():
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=render_callback,
        num_hands=1
    )

    return vision.HandLandmarker.create_from_options(options)


def setup_gesture_recognizer():
    base_options = python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH)

    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=gesture_callback,
        num_hands=1
    )

    return vision.GestureRecognizer.create_from_options(options)


# =========================
# HELPERS
# =========================
def send_to_hand(detector, img):
    ts = int(time.time() * 1000)
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    detector.detect_async(mp_img, ts)


def send_to_gesture(detector, img):
    ts = int(time.time() * 1000)
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    detector.recognize_async(mp_img, ts)


def get_gesture():
    if not latest_gesture or not latest_gesture.gestures:
        return None

    g = latest_gesture.gestures[0]
    if len(g) == 0:
        return None

    return g[0].category_name


# =========================
# APP
# =========================
class App(ShowBase):
    def __init__(self):
        super().__init__()

        self.setBackgroundColor(Vec4(0, 0, 0, 1))
        simplepbr.init()

        # Model
        self.model = self.loader.loadModel("models/Hologram3.glb")
        self.model.reparentTo(self.render)
        self.model.setPos(0, 15, 0)

        self.disableMouse()

        # Camera
        self.camera_target = Point3(0, 15, 0)
        self.camera_distance = 25.0
        self.update_camera()

        # Camera input
        self.cap = cv2.VideoCapture(0)

        # Mediapipe
        self.hand_detector = setup_hand_detector()
        self.gesture_detector = setup_gesture_recognizer()
        self.pinch_tracker = PinchTracker()

        self.taskMgr.add(self.update_task, "update_task")

    # =========================
    # MAIN LOOP
    # =========================
    def update_task(self, task):
        success, img = self.cap.read()
        if not success:
            return task.cont

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        send_to_hand(self.hand_detector, img)
        send_to_gesture(self.gesture_detector, img)

        pinch = None

        if latest_results and latest_results.hand_landmarks:
            hand = latest_results.hand_landmarks[0]
            pinch = self.pinch_tracker.update(hand[4], hand[8], w, h)

        gesture = get_gesture()

        # =========================
        # PINCH → ROTATE + TILT
        # =========================
        if pinch:
            dx, dy = pinch["dx"], pinch["dy"]
            self.model.setH(self.model.getH() + dx * 0.5)
            self.model.setP(self.model.getP() - dy * 0.5)

        # =========================
        # OPEN PALM → MOVE MODEL
        # =========================
        
        #elif gesture == "Open_Palm":
        #   hand = latest_results.hand_landmarks[0]

            # use index finger tip as movement driver (more stable than pinch)
        #    index_tip = hand[8]

            # convert normalized coords to screen delta
        #    dx = (index_tip.x - 0.5)
        #    dy = (index_tip.y - 0.5)

        #    self.model.setX(self.model.getX() + dx * 0.5)
        #   self.model.setZ(self.model.getZ() - dy * 0.5)
        
        # =========================
        # CLOSED FIST → ZOOM IN
        # =========================
        elif gesture == "Closed_Fist":
            self.camera_distance -= 0.4
            self.update_camera()

        # =========================
        # THUMBS DOWN → ZOOM OUT
        # =========================
        elif gesture == "Thumb_Down":
            self.camera_distance += 0.4
            self.update_camera()

        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)

        return task.cont

    # =========================
    # CAMERA
    # =========================
    def update_camera(self):
        self.camera_distance = max(2, self.camera_distance)

        self.camera.setPos(self.camera_target)
        self.camera.setPos(self.camera, 0, -self.camera_distance, 0)


# =========================
# RUN
# =========================
app = App()
app.run()