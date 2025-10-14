# train_demo.py — ring + train with full alignment controls

from math import pi
import time
from pathlib import Path
from dataclasses import dataclass

from spatialmath import SE3
from spatialgeometry import Mesh, Cuboid
from roboticstoolbox.backends import swift

# ----------------- config -----------------
@dataclass
class RingCfg:
    n_segments: int = 9
    radius: float = 1.33
    height_z: float = 0.1
    yaw_tweak_deg: float = 69.6
    center_x: float = 0.0
    center_y: float = 0.0
    global_yaw_deg: float = 0.0
    unit_scale: float = 0.0005
    color: tuple = (0.35, 0.35, 0.38, 1.0)

    
USE_SLIDERS_FOR_RING = False
ADD_TRAIN   = True

# --------------- helpers ------------------
def lay_flat_tangent_pose(i, n, cx, cy, R, Z, yaw_tweak_deg, gyaw_deg):
    dth = 2 * pi / n
    th  = i * dth
    yawfix = yaw_tweak_deg * pi/180
    gyaw   = gyaw_deg      * pi/180
    return (
        SE3(cx, cy, Z)
        * SE3.Rz(gyaw)
        * SE3.Rz(th)
        * SE3.Tx(R)
        * SE3.Rz(pi/2 + yawfix)
        * SE3.Rx(-pi/2)
    )

# --------------- objects ------------------
class TrackRing:
    def __init__(self, env: swift.Swift, segment_path: Path, cfg: RingCfg):
        self.env = env
        self.cfg = cfg
        self.segment_path = segment_path
        s = cfg.unit_scale
        self.scale = [s, s, s]
        self.pieces: list[Mesh] = []

    def build(self):
        for _ in range(self.cfg.n_segments):
            m = Mesh(str(self.segment_path), pose=SE3(), scale=self.scale, color=self.cfg.color)
            self.env.add(m)
            self.pieces.append(m)
        self.update()

    def update(self):
        c = self.cfg
        for i, p in enumerate(self.pieces):
            p.T = lay_flat_tangent_pose(
                i, c.n_segments, c.center_x, c.center_y, c.radius,
                c.height_z, c.yaw_tweak_deg, c.global_yaw_deg
            ).A

class Train:
    """
    Train that follows the ring; has adjustable local offsets and rotations.
    """
    def __init__(self, env: swift.Swift, mesh_path: Path, cfg: RingCfg):
        self.env = env
        self.cfg = cfg
        self.theta = 0.0
        self.speed = 0.25  # rad/s
        s = cfg.unit_scale
        self.mesh = Mesh(str(mesh_path), pose=SE3(), scale=[s, s, s], color=[0.12,0.12,0.12,1])
        env.add(self.mesh)

        # Alignment knobs (these will be bound to sliders)
        self.height_offset = 0.05   # up/down (m). Increase if it sits under the track.
        self.radial_offset = -0.03  # +outwards / -inwards (m)
        self.forward_offset = 0.00  # along tangent +forward (m)
        self.roll_deg  = 0.0        # local Rx (deg)
        self.pitch_deg = 0.0        # local Ry (deg)
        self.yaw_deg   = 0.0        # local Rz (deg)

    def pose_on_ring(self, theta):
        c = self.cfg
        # world pose to a tangent frame on the ring
        T = (SE3(c.center_x, c.center_y, c.height_z) *
             SE3.Rz(c.global_yaw_deg*pi/180) *
             SE3.Rz(theta) * SE3.Tx(c.radius) *
             SE3.Rz(pi/2 + c.yaw_tweak_deg*pi/180) *
             SE3.Rx(-pi/2))

        # local adjustments: first height, then fwd/radial, then fine rotations
        T = (T *
             SE3.Tz(self.height_offset) *
             SE3.Tx(self.forward_offset) *
             SE3.Ty(self.radial_offset) *
             SE3.Rx(self.roll_deg*pi/180) *
             SE3.Ry(self.pitch_deg*pi/180) *
             SE3.Rz(self.yaw_deg*pi/180))
        return T

    def update(self, dt):
        self.theta = (self.theta + self.speed * dt) % (2*pi)
        self.mesh.T = self.pose_on_ring(self.theta).A

# --------------- app shell ----------------
class App:
    def __init__(self, use_sliders_for_ring=False):
        self.env = swift.Swift()
        self.env.launch(realtime=True, port=0)
        self.use_sliders_for_ring = use_sliders_for_ring
        self.dt = 0.05

    def add_floor(self, size=(8,8,0.02), z=-0.01):
        self.env.add(Cuboid(list(size), pose=SE3(0,0,z), color=[0.9,0.9,0.95,1]))

    def add_ring(self, seg_path: Path, cfg: RingCfg):
        self.ring = TrackRing(self.env, seg_path, cfg)
        self.ring.build()

        if self.use_sliders_for_ring:
            self.env.add(swift.Label("— Track Ring Controls —"))
            self.radius_slider = swift.Slider(cb=lambda v: None, min=0.10, max=10.0, step=0.01, value=cfg.radius,      desc="Radius (m)")
            self.height_slider = swift.Slider(cb=lambda v: None, min=0.00, max=0.50, step=0.01, value=cfg.height_z,    desc="Height Z (m)")
            self.yaw_slider    = swift.Slider(cb=lambda v: None, min=-5.0, max=1000.0, step=0.05, value=cfg.yaw_tweak_deg, desc="Per-piece yaw tweak (°)")
            self.cx_slider     = swift.Slider(cb=lambda v: None, min=-3.0, max=3.0, step=0.01,  value=cfg.center_x,    desc="Center X (m)")
            self.cy_slider     = swift.Slider(cb=lambda v: None, min=-3.0, max=3.0, step=0.01,  value=cfg.center_y,    desc="Center Y (m)")
            self.ringyaw_slider= swift.Slider(cb=lambda v: None, min=-180, max=180, step=1.0,   value=cfg.global_yaw_deg, desc="Global yaw (°)")
            for s in (self.radius_slider, self.height_slider, self.yaw_slider, self.cx_slider, self.cy_slider, self.ringyaw_slider):
                self.env.add(s)

    def add_train(self, train_path: Path, cfg: RingCfg):
        self.env.add(swift.Label("— Train —"))
        self.train = Train(self.env, train_path, cfg)

        # motion + alignment sliders
        self.speed_slider   = swift.Slider(cb=lambda v: None, min=0.0,  max=2.0, step=0.01, value=self.train.speed,         desc="Speed (rad/s)")
        self.h_slider       = swift.Slider(cb=lambda v: None, min=-0.2, max=0.5, step=0.005, value=self.train.height_offset, desc="Height (m)")
        self.radial_slider  = swift.Slider(cb=lambda v: None, min=-10, max=0.2, step=0.002, value=self.train.radial_offset, desc="Radial (m)")
        self.forward_slider = swift.Slider(cb=lambda v: None, min=-0.2, max=0.2, step=0.002, value=self.train.forward_offset,desc="Forward (m)")
        self.roll_slider    = swift.Slider(cb=lambda v: None, min=-180, max=180, step=1.0,   value=self.train.roll_deg,      desc="Roll Rx (°)")
        self.pitch_slider   = swift.Slider(cb=lambda v: None, min=-180, max=180, step=1.0,   value=self.train.pitch_deg,     desc="Pitch Ry (°)")
        self.yaw_slider     = swift.Slider(cb=lambda v: None, min=-180, max=180, step=1.0,   value=self.train.yaw_deg,       desc="Yaw Rz (°)")

        for s in (self.speed_slider, self.h_slider, self.radial_slider,
                  self.forward_slider, self.roll_slider, self.pitch_slider, self.yaw_slider):
            self.env.add(s)

    def loop(self):
        try:
            while True:
                if self.use_sliders_for_ring:
                    c = self.ring.cfg
                    c.radius = self.radius_slider.value
                    c.height_z = self.height_slider.value
                    c.yaw_tweak_deg = self.yaw_slider.value
                    c.center_x = self.cx_slider.value
                    c.center_y = self.cy_slider.value
                    c.global_yaw_deg = self.ringyaw_slider.value
                    self.ring.update()

                if hasattr(self, "train"):
                    t = self.train
                    t.speed          = self.speed_slider.value
                    t.height_offset  = self.h_slider.value
                    t.radial_offset  = self.radial_slider.value
                    t.forward_offset = self.forward_slider.value
                    t.roll_deg       = self.roll_slider.value
                    t.pitch_deg      = self.pitch_slider.value
                    t.yaw_deg        = self.yaw_slider.value
                    t.update(self.dt)

                self.env.step()
                time.sleep(self.dt)
        except KeyboardInterrupt:
            print("Exiting.")

# ----------------- main -------------------
if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    seg_path   = here / "track_segment.STL"
    train_path = here / "train.STL"

    print("Segment exists?", seg_path.exists(), seg_path)
    print("Train exists?  ", train_path.exists(), train_path)

    cfg = RingCfg()
    app = App(use_sliders_for_ring=USE_SLIDERS_FOR_RING)
    app.add_floor()
    app.add_ring(seg_path, cfg)

    if ADD_TRAIN and train_path.exists():
        app.add_train(train_path, cfg)
    elif ADD_TRAIN:
        print("[warn] train.STL not found; skipping train.")

    app.loop()
