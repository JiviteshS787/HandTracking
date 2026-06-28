from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, Point3, Vec4
import simplepbr

loadPrcFileData("", "notify-level-glgsg debug")

class App(ShowBase):
    def __init__(self):
        super().__init__()

        # ---------- SET DARK BACKGROUND ----------
        self.setBackgroundColor(Vec4(0.00, 0.00, 0.00, 1))
        # ---------- INITIALIZE PBR PIPELINE ----------
        self.pipeline = simplepbr.init()

        # ---------- LOAD MODEL ----------
        self.model = self.loader.loadModel("models/Hologram3.glb")
        self.model.reparentTo(self.render)
        self.model.setScale(1.0)
        self.model.setPos(0, 15, 0)

        # ---------- CUSTOM ORBITAL CAMERA SETUP ----------
        self.disableMouse()
        
        self.camera_target = Point3(0, 15, 0)
        self.camera_distance = 25.0
        self.camera_heading = 0.0
        self.camera_pitch = -10.0

        # Fine-tune sensitivity settings
        self.rotate_sensitivity = 50.0
        self.pan_sensitivity = 10.0
        self.wheel_zoom_sensitivity = 1.5  # Controls trackpad / scroll wheel step size

        # ---------- BIND SCROLL WHEEL / TRACKPAD EVENTS ----------
        # This catches standard scrolling without needing to click down buttons
        self.accept("wheel_up", self.handle_wheel_zoom, [-self.wheel_zoom_sensitivity])
        self.accept("wheel_down", self.handle_wheel_zoom, [self.wheel_zoom_sensitivity])

        self.last_mouse_pos = (0, 0)
        self.update_camera()

        self.taskMgr.add(self.control_camera_task, "control_camera_task")

    def update_camera(self):
        self.camera_pitch = max(-85.0, min(85.0, self.camera_pitch))
        # Keep distance bounded so you can't zoom infinitely or inside out
        self.camera_distance = max(1.5, self.camera_distance)
        
        self.camera.setPos(self.camera_target)
        self.camera.setHpr(self.camera_heading, self.camera_pitch, 0)
        self.camera.setPos(self.camera, 0, -self.camera_distance, 0)

    def handle_wheel_zoom(self, amount):
        """Executes whenever the mouse wheel or trackpad scroll gesture fires."""
        self.camera_distance += amount
        self.update_camera()

    def control_camera_task(self, task):
        if self.mouseWatcherNode.hasMouse():
            m_pos = self.mouseWatcherNode.getMouse()
            x, y = m_pos.getX(), m_pos.getY()
            
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]

            # 1. LEFT CLICK DRAG: Rotate Camera
            if self.mouseWatcherNode.isButtonDown("mouse1"):
                self.camera_heading -= dx * self.rotate_sensitivity
                self.camera_pitch += dy * self.rotate_sensitivity
                self.update_camera()

            # 2. MIDDLE CLICK DRAG: Pan Target
            elif self.mouseWatcherNode.isButtonDown("mouse2"):
                self.camera_target.setX(self.camera_target.getX() - (dx * self.pan_sensitivity))
                self.camera_target.setZ(self.camera_target.getZ() - (dy * self.pan_sensitivity))
                self.update_camera()

            self.last_mouse_pos = (x, y)
            
        return task.cont

app = App()
app.run()