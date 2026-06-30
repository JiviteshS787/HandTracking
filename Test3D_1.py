from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, Point3, Vec4
import simplepbr

loadPrcFileData("", "notify-level-glgsg debug")

class App(ShowBase):
    def __init__(self):
        super().__init__()

        # ---------- SET DARK BACKGROUND ----------
        # Colors are represented as Vec4(Red, Green, Blue, Alpha) from 0.0 to 1.0.
        # Pure Black: Vec4(0, 0, 0, 1)
        # Deep Charcoal Gray: Vec4(0.1, 0.1, 0.1, 1)
        self.setBackgroundColor(Vec4(0.00, 0.00, 0.00, 1))

        # ---------- INITIALIZE PBR PIPELINE ----------
        self.pipeline = simplepbr.init()

        # ---------- LOAD MODEL ----------
        self.model = self.loader.loadModel("models/Hologram2.glb")
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
        self.zoom_sensitivity = 15.0
        self.pan_sensitivity = 10.0

        self.last_mouse_pos = (0, 0)
        self.update_camera()

        self.taskMgr.add(self.control_camera_task, "control_camera_task")

    def update_camera(self):
        self.camera_pitch = max(-85.0, min(85.0, self.camera_pitch))
        self.camera.setPos(self.camera_target)
        self.camera.setHpr(self.camera_heading, self.camera_pitch, 0)
        self.camera.setPos(self.camera, 0, -self.camera_distance, 0)

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

            elif self.mouseWatcherNode.isButtonDown("mouse3"):
                self.camera_distance += dy * self.zoom_sensitivity
                self.camera_distance = max(2.0, self.camera_distance)
                self.update_camera()

            elif self.mouseWatcherNode.isButtonDown("mouse2"):
                self.camera_target.setX(self.camera_target.getX() - (dx * self.pan_sensitivity))
                self.camera_target.setZ(self.camera_target.getZ() - (dy * self.pan_sensitivity))
                self.update_camera()

            self.last_mouse_pos = (x, y)
            
        return task.cont

app = App()
app.run()