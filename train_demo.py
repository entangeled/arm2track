# train_demo.py — Ring + Train + Panda + Linked Track Zones
from math import pi
from pathlib import Path
from dataclasses import dataclass

from spatialmath import SE3
from spatialgeometry import Mesh, Cuboid, Cylinder
from roboticstoolbox.backends import swift
from roboticstoolbox import models  # Panda model


# -------------------------------
# Config
# -------------------------------
@dataclass
class RingCfg:
    n_segments: int = 9
    radius: float = 1.33
    height_z: float = 0.10
    yaw_tweak_deg: float = 69.6
    center_x: float = 0.0
    center_y: float = 0.0
    global_yaw_deg: float = 0.0
    unit_scale: float = 0.0005
    color: tuple = (0.35, 0.35, 0.38, 1.0)


# -------------------------------
# Geometry helpers
# -------------------------------
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


# -------------------------------
# Track ring
# -------------------------------
class TrackRing:
    def __init__(self, env, segment_path, cfg: RingCfg):
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
                i, c.n_segments, c.center_x, c.center_y, c.radius,
                c.height_z, c.yaw_tweak_deg, c.global_yaw_deg
            ).A

    # ---- helpers to place/move any segment at angle theta ----
    def pose_for_theta(self, theta):
        c = self.cfg
        return (
            SE3(c.center_x, c.center_y, c.height_z)
            * SE3.Rz(c.global_yaw_deg * pi / 180)
            * SE3.Rz(theta)
            * SE3.Tx(c.radius)
            * SE3.Rz(pi / 2 + c.yaw_tweak_deg * pi / 180)
            * SE3.Rx(-pi / 2)
        )

    def set_segment_angle(self, idx, theta):
        self.pieces[idx].T = self.pose_for_theta(theta).A


# -------------------------------
# Train cart running on ring
# -------------------------------
class Train:
    """Train perfectly aligned on the track."""
    def __init__(self, env, mesh_path, cfg: RingCfg):
        self.env = env
        self.cfg = cfg
        self.theta = 0.0
        self.speed = 1.0  # rad/s
        s = cfg.unit_scale

        self.mesh = Mesh(str(mesh_path), pose=SE3(), scale=[s, s, s], color=[0.12, 0.12, 0.12, 1])
        env.add(self.mesh)

        # locked-in placement (scaled values)
        self.height_offset = 0.10
        self.radial_offset = 0.18
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
        return (
            T
            * SE3.Tz(self.height_offset)
            * SE3.Tx(self.forward_offset)
            * SE3.Ty(self.radial_offset)
            * SE3.Rx(self.roll_deg * pi / 180)
            * SE3.Ry(self.pitch_deg * pi / 180)
            * SE3.Rz(self.yaw_deg * pi / 180)
        )

    def update(self, dt):
        self.theta = (self.theta + self.speed * dt) % (2 * pi)
        self.mesh.T = self.pose_on_ring(self.theta).A


# -------------------------------
# Linked reach blocks (one per segment)
# -------------------------------
class TrackLinkedZones:
    def __init__(self, env, ring, cfg, lift=0.02, block_offset=0.255):
        self.env = env
        self.cfg = cfg
        self.ring = ring
        self.lift = lift
        self.block_offset = block_offset
        self.n_zones = 6
        self.blocks = []
        self.thetas = [2 * pi / self.n_zones * i for i in range(self.n_zones)]

    def build(self):
        # build the red blocks
        for theta in self.thetas:
            T = self._upright_block_pose(theta)
            cube = Cuboid([0.05, 0.05, 0.05], pose=T, color=[0.9, 0.3, 0.3, 0.7])
            self.env.add(cube)
            self.blocks.append(cube)

        # 🌟 add your sliders here
        self.offset_slider = swift.Slider(
            cb=self.on_offset,
            min=-0.10, max=0.35, step=0.001,
            value=self.block_offset,
            desc="Blocks radial offset (m)"
        )
        self.env.add(self.offset_slider)

        self.height_slider = swift.Slider(
            cb=self.on_height,
            min=-0.05, max=0.05, step=0.001,
            value=self.lift,
            desc="Blocks height over track (m)"
        )
        self.env.add(self.height_slider)

    # ---- add these handlers below build() ----
    def on_theta_offset(self, v):
        self.theta_offset = v
        for i, th in enumerate(self.thetas):
            self.blocks[i].T = self._upright_block_pose(th).A

    def on_offset(self, v):
        """Move blocks radially (closer/farther from ring center)."""
        self.block_offset = v
        for i, th in enumerate(self.thetas):
            self.blocks[i].T = self._upright_block_pose(th).A

    def on_height(self, v):
        """Move blocks up/down relative to track surface."""
        self.lift = v
        for i, th in enumerate(self.thetas):
            self.blocks[i].T = self._upright_block_pose(th).A

    # ---- your existing function ----
    def _upright_block_pose(self, theta):
        c = self.cfg
        base = (
            SE3(c.center_x, c.center_y, 0.0)
            * SE3.Rz(c.global_yaw_deg * pi / 180)
            * SE3.Rz(theta)
            * SE3.Tx(c.radius + self.block_offset)
        )
        return SE3.Tz(c.height_z + self.lift) * base




# -------------------------------
# Panda robot in the middle (tracks the train)
# -------------------------------
class RobotArm:
    """Panda at ring center; downsampled IK + joint interpolation for smoothness."""
    def __init__(self, env, cfg: RingCfg):
        self.env = env
        self.cfg = cfg

        self.robot = models.Panda()
        self.robot.base = SE3(cfg.center_x, cfg.center_y, 0.0)
        env.add(self.robot)

        # pedestal (visual)
        env.add(Cylinder(radius=0.12, length=0.25,
                         pose=SE3(cfg.center_x, cfg.center_y, 0.125),
                         color=[0.2, 0.2, 0.22, 1.0]))

        self.q_home = self.robot.qz
        self.q_target = self.q_home.copy()
        self.alpha = 0.20   # interpolation per frame
        self.ik_hz = 20
        self._accum = 0.0

        self.hover_z = 0.30
        self.lead = 0.10
        self.radial_extra = 0.25
        self._tilt = SE3.RPY([pi * 0.85, 0, 0], order="xyz")
        self._mask = [1, 1, 1, 0.5, 0.5, 0.5]

    def goal_over_train(self, theta, train: Train):
        c = self.cfg
        th = theta + self.lead
        T = (
            SE3(c.center_x, c.center_y, c.height_z)
            * SE3.Rz(c.global_yaw_deg * pi / 180)
            * SE3.Rz(th)
            * SE3.Tx(c.radius)
            * SE3.Rz(pi / 2 + c.yaw_tweak_deg * pi / 180)
            * SE3.Rx(-pi / 2)
        )
        return (
            T
            * SE3.Tz(train.height_offset + self.hover_z)
            * SE3.Tx(train.forward_offset)
            * SE3.Ty(self.radial_extra)
        ) * self._tilt

    def update(self, dt, theta, train):
        self._accum += dt
        ik_period = 1.0 / self.ik_hz
        if self._accum >= ik_period:
            self._accum -= ik_period
            T_goal = self.goal_over_train(theta, train)
            sol = self.robot.ikine_LM(T_goal, q0=self.robot.q, mask=self._mask)
            if sol.success:
                self.q_target = sol.q
            else:
                self.q_target = 0.98 * self.q_target + 0.02 * self.q_home
        self.robot.q = (1 - self.alpha) * self.robot.q + self.alpha * self.q_target


# -------------------------------
# App wrapper
# -------------------------------
class App:
    def __init__(self):
        self.env = swift.Swift()
        self.env.launch(realtime=True, port=0)
        self.dt = 0.05

    def add_floor(self, size=(8, 8, 0.02), z=-0.01):
        self.env.add(Cuboid(scale=list(size), pose=SE3(0, 0, z), color=[0.9, 0.9, 0.95, 1]))

    def add_ring(self, seg_path, cfg: RingCfg):
        self.ring = TrackRing(self.env, seg_path, cfg)
        self.ring.build()

    def add_train(self, train_path, cfg: RingCfg):
        self.train = Train(self.env, train_path, cfg)

    def add_robot(self, cfg: RingCfg):
        self.robot = RobotArm(self.env, cfg)

    def add_linked_zones(self, cfg: RingCfg):
        # block_offset=-0.03 pulls them ~3 cm inward from the ring radius
        self.zones = TrackLinkedZones(self.env, self.ring, cfg, lift=0.00, block_offset= 0.255)
        self.zones.build()


    def loop(self):
        try:
            while True:
                self.train.update(self.dt)
                self.robot.update(self.dt, self.train.theta, self.train)
                self.env.step(self.dt)
        except KeyboardInterrupt:
            print("Exiting simulation...")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    seg_path = here / "track_segment.STL"
    train_path = here / "train.STL"

    cfg = RingCfg()
    app = App()
    app.add_floor()
    app.add_ring(seg_path, cfg)
    app.add_train(train_path, cfg)
    app.add_robot(cfg)
    app.add_linked_zones(cfg)   # blocks on rail centreline, linked to segments
    app.loop()
