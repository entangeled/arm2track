# train_demo.py — Ring + Train (aligned)

from math import pi
import time
from pathlib import Path
from dataclasses import dataclass

from spatialmath import SE3
from spatialgeometry import Mesh, Cuboid
from roboticstoolbox.backends import swift

@dataclass
class RingCfg:
    n_segments: int = 9
    radius: float = 2.67
    height_z: float = 0.20
    yaw_tweak_deg: float = 69.65
    center_x: float = 0.0
    center_y: float = 0.0
    global_yaw_deg: float = 0.0
    unit_scale: float = 0.001
    color: tuple = (0.35, 0.35, 0.38, 1.0)

def lay_flat_tangent_pose(i, n, cx, cy, R, Z, yaw_tweak_deg, gyaw_deg):
    dth = 2 * pi / n
    th = i * dth
    yawfix = yaw_tweak_deg * pi / 180
    gyaw = gyaw_deg * pi / 180
    return (
        SE3(cx, cy, Z)
        * SE3.Rz(gyaw)
        * SE3.Rz(th)
        * SE3.Tx(R)
        * SE3.Rz(pi / 2 + yawfix)
        * SE3.Rx(-pi / 2)
    )

class TrackRing:
    def __init__(self, env, segment_path, cfg):
        self.env = env
        self.cfg = cfg
        self.segment_path = segment_path
        s = cfg.unit_scale
        self.scale = [s, s, s]
        self.pieces = []

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
                i, c.n_segments, c.center_x, c.center_y, c.radius, c.height_z, c.yaw_tweak_deg, c.global_yaw_deg
            ).A

class Train:
    """Train perfectly aligned on the track"""
    def __init__(self, env, mesh_path, cfg):
        self.env = env
        self.cfg = cfg
        self.theta = 0.0
        self.speed = 1.0  # rad/s constant speed
        s = cfg.unit_scale

        # load mesh and add to environment
        self.mesh = Mesh(str(mesh_path), pose=SE3(), scale=[s, s, s], color=[0.12, 0.12, 0.12, 1])
        env.add(self.mesh)

        # ✅ Locked-in perfect placement
        self.height_offset = 0.05
        self.radial_offset = 0.35
        self.forward_offset = 0.00
        self.roll_deg = 0.0
        self.pitch_deg = 84.0
        self.yaw_deg = 180.0

    def pose_on_ring(self, theta):
        c = self.cfg
        T = (
            SE3(c.center_x, c.center_y, c.height_z)
            * SE3.Rz(c.global_yaw_deg * pi / 180)
            * SE3.Rz(theta)
            * SE3.Tx(c.radius)
            * SE3.Rz(pi / 2 + c.yaw_tweak_deg * pi / 180)
            * SE3.Rx(-pi / 2)
        )
        T = (
            T
            * SE3.Tz(self.height_offset)
            * SE3.Tx(self.forward_offset)
            * SE3.Ty(self.radial_offset)
            * SE3.Rx(self.roll_deg * pi / 180)
            * SE3.Ry(self.pitch_deg * pi / 180)
            * SE3.Rz(self.yaw_deg * pi / 180)
        )
        return T

    def update(self, dt):
        self.theta = (self.theta + self.speed * dt) % (2 * pi)
        self.mesh.T = self.pose_on_ring(self.theta).A

class App:
    def __init__(self):
        self.env = swift.Swift()
        self.env.launch(realtime=True, port=0)
        self.dt = 0.05

    def add_floor(self, size=(8, 8, 0.02), z=-0.01):
        self.env.add(Cuboid(list(size), pose=SE3(0, 0, z), color=[0.9, 0.9, 0.95, 1]))

    def add_ring(self, seg_path, cfg):
        self.ring = TrackRing(self.env, seg_path, cfg)
        self.ring.build()

    def add_train(self, train_path, cfg):
        self.train = Train(self.env, train_path, cfg)

    def loop(self):
        try:
            while True:
                self.train.update(self.dt)
                self.env.step()
                time.sleep(self.dt)
        except KeyboardInterrupt:
            print("Exiting simulation...")

class Safety:
    """
    Simple safety manager for the train.
    - E-stop
    - Speed limit
    - Radial geofence about the ring center (band)
    - Height window relative to ring plane
    """
    def __init__(self, env, cfg: RingCfg):
        self.cfg = cfg
        self.env = env

        # default limits – tweak to your liking
        self.estop = False
        self.max_speed = 1.25          # rad/s max
        self.band_inner = cfg.radius - 0.08   # m, inner radial bound
        self.band_outer = cfg.radius + 0.08   # m, outer radial bound
        self.z_min = cfg.height_z + 0.02      # m (over ring plane)
        self.z_max = cfg.height_z + 0.12      # m

        # UI
        self.env.add(swift.Label("— Safety —"))
        self.estop_slider = swift.Slider(cb=lambda v: None, min=0, max=1, step=1, value=0,
                                         desc="E-stop (0=run, 1=stop)")
        self.max_speed_slider = swift.Slider(cb=lambda v: None, min=0.1, max=2.0, step=0.01,
                                             value=self.max_speed, desc="Max speed (rad/s)")
        self.band_slider = swift.Slider(cb=lambda v: None, min=0.02, max=0.20, step=0.005,
                                        value=0.08, desc="Rail half-width (m)")
        self.zmin_slider = swift.Slider(cb=lambda v: None, min=0.0, max=0.20, step=0.005,
                                        value=self.z_min - cfg.height_z, desc="Min height over ring (m)")
        self.zmax_slider = swift.Slider(cb=lambda v: None, min=0.05, max=0.30, step=0.005,
                                        value=self.z_max - cfg.height_z, desc="Max height over ring (m)")
        for w in (self.estop_slider, self.max_speed_slider, self.band_slider,
                  self.zmin_slider, self.zmax_slider):
            self.env.add(w)

        self.status = swift.Label("Status: OK")
        self.env.add(self.status)

    def _set_status(self, text, alarm=False):
        self.status.desc = f"Status: {text}"
        # optional: color flip on alarm
        # (Swift doesn't support per-Label color; we’ll tint the train instead.)

    def apply(self, train: "Train"):
        """
        Enforce limits; return allowed_speed and a flag whether we’re safe.
        """
        # sync from UI
        self.estop = (self.estop_slider.value > 0.5)
        self.max_speed = self.max_speed_slider.value
        half_width = self.band_slider.value
        self.band_inner = self.cfg.radius - half_width
        self.band_outer = self.cfg.radius + half_width
        self.z_min = self.cfg.height_z + self.zmin_slider.value
        self.z_max = self.cfg.height_z + self.zmax_slider.value

        # 1) e-stop
        if self.estop:
            self._set_status("ESTOP")
            return 0.0, False

        # grab current world pose
        T = train.mesh.T     # 4x4
        x, y, z = T[0, 3], T[1, 3], T[2, 3]

        # 2) geofence (radial)
        r = (x - self.cfg.center_x)**2 + (y - self.cfg.center_y)**2
        r = r**0.5
        if r < self.band_inner or r > self.band_outer:
            self._set_status("GEOFENCE")
            return 0.0, False

        # 3) height window
        if z < self.z_min or z > self.z_max:
            self._set_status("HEIGHT")
            return 0.0, False

        # 4) speed governor
        allowed = max(-self.max_speed, min(self.max_speed, train.speed))
        self._set_status("OK")
        return allowed, True


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    seg_path = here / "track_segment.STL"
    train_path = here / "train.STL"

    cfg = RingCfg()
    app = App()
    app.add_floor()
    app.add_ring(seg_path, cfg)
    app.add_train(train_path, cfg)
    app.loop()
