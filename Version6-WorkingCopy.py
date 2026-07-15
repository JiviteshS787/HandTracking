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

PINCH_MOVE_SPEED = 0.5   # Rotation/tilt speed
PALM_MOVE_SPEED = 0.03   # Camera movement speed
ZOOM_SPEED = 0.4  

latest_results = None
latest_gesture = None


# =========================
# CALLBACKS
# =========================

# Required for LIVESTREAM mode
# Stores the landmarkers and such of the current gesture
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
        if index_tip is None or thumb_tip is None:
            self.reset()
            return None

        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5 # Pythagorean Theorem
        
        # Stable midpoint of pinch and map it to on-screen pixels
        pinch_x = int(((thumb_tip.x + index_tip.x) / 2) * w)
        pinch_y = int(((thumb_tip.y + index_tip.y) / 2) * h)

        if distance < self.threshold:
            if not self.isPinching:
                # Set to isPinching and update stored coords.
                self.isPinching = True
                self.lastX, self.lastY = pinch_x, pinch_y #Set previous to current as there is no movement in first detection.

            dx = pinch_x - self.lastX # Change in x = current-previous
            dy = pinch_y - self.lastY # Change in y = current-previous

            self.lastX, self.lastY = pinch_x, pinch_y

            # Return the change in positions.
            return {"dx": dx, "dy": dy}

        self.reset()
        return None


# =========================
# PALM TRACKER
# =========================
class PalmTracker:
    def __init__(self):
        self.isOpenPalm = False
        self.lastX = None
        self.lastY = None

    def reset(self):
        self.isOpenPalm = False
        self.lastX = self.lastY = None

    def update(self, wrist, middle_mcp, w, h):
        if wrist is None or middle_mcp is None:
            self.reset()
            return None
        
        # Stable midpoint of pinch and map it to on-screen pixels
        move_x = int(((wrist.x + middle_mcp.x) / 2) * w)
        move_y = int(((wrist.y + middle_mcp.y) / 2) * h)

        
        if not self.isOpenPalm:
            # Set to isPinching and update stored coords.
            self.isOpenPalm = True
            self.lastX, self.lastY = move_x, move_y #Set previous to current as there is no movement in first detection.
            return None
        
        dx = move_x - self.lastX # Change in x = current-previous
        dy = move_y - self.lastY # Change in y = current-previous
        
        self.lastX, self.lastY = move_x, move_y

        # Return the change in positions.
        return {"dx": dx, "dy": dy}


# =========================
# MEDIAPIPE SETUP
# =========================
def setup_hand_detector():
    # What model to use
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)

    # Set behavious, vision -> API model, HandLandmarker -> AI model, HandLandmarkerOptions -> configurable class of model
    # CONFIGURATION ONLY
    options = vision.HandLandmarkerOptions(
        # Load model
        base_options=base_options,
        # Set detection mode
        running_mode=vision.RunningMode.LIVE_STREAM,
        # What to do once detected/done processing - async. call
        result_callback=render_callback,
        # Number of hands to detect
        num_hands=1
    )

    # Create the detector object using the options just set
    return vision.HandLandmarker.create_from_options(options)

# Same as setup_hand_detector(), only difference is model in use
def setup_gesture_recognizer():
    base_options = python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH)

    # Set behavious, vision -> API model, GestureRecognier -> AI model, GesureRecognizerOptions -> configurable class of model

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
    ts = int(time.time() * 1000) # Current millisecond time
    mp_img = mp.Image( # Convert OpenCV to MediaPipe image, RGB correction.
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    detector.detect_async(mp_img, ts) # process frame, handlandmark detector

def send_to_gesture(detector, img):
    ts = int(time.time() * 1000) # Current millisecond time
    mp_img = mp.Image( # Convert OpenCV to Mediapipe image, RGB correction.
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    detector.recognize_async(mp_img, ts) # process frame, gesture recognizer


def get_gesture():
    if not latest_gesture or not latest_gesture.gestures:
        return None
    #latest_gesture -> MP obj., .gestures -> actual gestures

    g = latest_gesture.gestures[0] # -> gestures[0] -> first hand
    if len(g) == 0: # -> if no gestures detected/predicted
        return None

    # Predicted gestures are ranked in the array based on detection confidence,
    # g[0] always returns highest confidence gesture
    return {"name": g[0].category_name, "confidence": g[0].score} # return name of gesture


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
        # Setup detectors and initiate PinchTracker class and the PalmTracker class
        self.hand_detector = setup_hand_detector()
        self.gesture_detector = setup_gesture_recognizer()
        self.pinch_tracker = PinchTracker()
        self.palm_tracker = PalmTracker()

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

        # Send detcted img/frames to both detectors
        send_to_hand(self.hand_detector, img)
        send_to_gesture(self.gesture_detector, img)

        pinch = None
        palm = None
        hand = None
        
        if latest_results and latest_results.hand_landmarks:
            # .hand_landmarks[0] -> returns landmarks of first hand
            hand = latest_results.hand_landmarks[0]
            # Send thumb_tip, index_tip, w and h of screen to method
            pinch = self.pinch_tracker.update(hand[4], hand[8], w, h)
        
        # Detect gestures
        gesture = get_gesture()
        gesture_name = None
        gesture_confidence = 0

        if gesture:
            gesture_name = gesture["name"]
            gesture_confidence = gesture["confidence"]

        # =========================
        # CLOSED FIST → ZOOM IN
        # =========================
        if gesture_name == "Closed_Fist" and gesture_confidence > 0.8:
            self.camera_distance -= ZOOM_SPEED
            self.update_camera()

        # =========================
        # THUMBS DOWN → ZOOM OUT
        # =========================
        elif gesture_name == "Thumb_Down" and gesture_confidence > 0.9:
            self.camera_distance += ZOOM_SPEED
            self.update_camera()
        
        # =========================
        # PINCH → ROTATE + TILT
        # =========================
        elif pinch:
            dx, dy = pinch["dx"], pinch["dy"] # extract information from return object
            self.model.setH(self.model.getH() + dx * PINCH_MOVE_SPEED) # set heading, *0.5 is a scaling factor. Left, right rotate
            self.model.setP(self.model.getP() + dy * PINCH_MOVE_SPEED) # set pitch, *0.5 is a scaling factor. Up and down tilt

        # =========================
        # OPEN PALM → MOVE MODEL
        # =========================
        elif gesture_name == "Open_Palm" and hand and gesture_confidence >= 0.65:
            palm = self.palm_tracker.update(hand[0], hand[9], w, h)
            if palm:
                dx, dy = palm["dx"], palm["dy"] #extract information from return object
                self.camera_target.setX(
                    self.camera_target.getX() - dx * PALM_MOVE_SPEED
                )

                self.camera_target.setZ(
                    self.camera_target.getZ() + dy * PALM_MOVE_SPEED
                )
                self.update_camera()

        # =========================
        # RESET UNUSED TRACKERS
        # =========================
        if gesture_name != "Open_Palm":
            self.palm_tracker.reset()

        if not pinch:
            self.pinch_tracker.reset()

        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)

        return task.cont

    # =========================
    # CAMERA
    # =========================
    def update_camera(self):
        self.camera_distance = max(2, min(50, self.camera_distance))

        self.camera.setPos(self.camera_target)
        self.camera.setPos(self.camera, 0, -self.camera_distance, 0)


# =========================
# RUN
# =========================
app = App()
app.run()