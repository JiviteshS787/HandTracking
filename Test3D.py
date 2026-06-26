from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, Vec4

class App(ShowBase):
    def __init__(self):
        super().__init__()

        # ---------- LOAD MODEL ----------
        self.model = self.loader.loadModel("models/Hologram.glb")
        self.model.reparentTo(self.render)

        self.model.setScale(0.5)
        self.model.setPos(0, 15, 0)

        # IMPORTANT: keep GLB materials working
        self.model.clearColor()
        self.model.setShaderAuto()

        # ---------- LIGHTING ----------
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.6, 0.6, 0.6, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        directional = DirectionalLight("directional")
        directional.setColor(Vec4(0.8, 0.8, 0.8, 1))
        directional_np = self.render.attachNewNode(directional)
        directional_np.setHpr(0, -60, 0)
        self.render.setLight(directional_np)

        # ---------- UPDATE LOOP ----------
        self.taskMgr.add(self.update, "update")

        self.rot_speed = 1.0
        self.tilt_speed = 0.5

    def update(self, task):
        self.model.setH(self.model.getH() + self.rot_speed)
        self.model.setP(self.model.getP() + self.tilt_speed)
        return task.cont


app = App()
app.run()