from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, Point3, Vec4, NodePath, CardMaker
import simplepbr

loadPrcFileData("", "notify-level-glgsg debug")

class App(ShowBase):
    def __init__(self):
        super().__init__()

        # ---------- SET DARK BACKGROUND ----------
        self.setBackgroundColor(Vec4(0.00, 0.00, 0.00, 1))
        # ---------- INITIALIZE PBR PIPELINE ----------
        self.pipeline = simplepbr.init()

        # ==========================================
        # ---------- SETUP SPLIT SCREEN ------------
        # ==========================================
        # 1. Resize the main 3D camera's region to only take up the left half
        # Dimensions are normalized: (Left, Right, Bottom, Top) from 0.0 to 1.0
        main_region = self.camNode.getDisplayRegion(0)
        main_region.setDimensions(0.0, 0.5, 0.0, 1.0)

        # 2. Create a new region on the right half of the window
        right_region = self.win.makeDisplayRegion(0.5, 1.0, 0.0, 1.0)
        right_region.setSort(10)  # Make sure it renders properly
        
        # 3. Create a dedicated 2D camera for the right region to display images
        # We use a distinct 2D scene graph root so it doesn't overlap with the 3D scene
        self.right_render2d = NodePath('right_render2d')
        self.right_cam2d = self.makeCamera2d(right_region)
        self.right_cam2d.reparentTo(self.right_render2d)

        # 4. Display an image on the right side using CardMaker
        cm = CardMaker('image_card')
        # Map card coordinates from -1 to 1 to fill the 2D region perfectly
        cm.setFrame(-1, 1, -1, 1) 
        card = self.right_render2d.attachNewNode(cm.generate())
        
        # Load your image texture here (replace with your actual image path)
        # tex = loader.loadTexture("maps/your_image.png")
        # card.setTexture(tex)
        # ==========================================

        # ---------- LOAD MODEL (Stays on Left) ----------
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
        self.wheel_zoom_sensitivity = 1.5  

        # ---------- BIND SCROLL WHEEL / TRACKPAD EVENTS ----------
        self.accept("wheel_up", self.handle_wheel_zoom, [-self.wheel_zoom_sensitivity])
        self.accept("wheel_down", self.handle_wheel_zoom, [self.wheel_zoom_sensitivity])

        self.last_mouse_pos = (0, 0)
        self.update_camera()

        self.taskMgr.add(self.control_camera_task, "control_camera_task")

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