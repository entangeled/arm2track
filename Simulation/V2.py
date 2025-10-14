# train_demo.py — Ring + Train (aligned, minimal) + Panda in the middle

from math import pi
import time
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
    height_z: float = 0.1
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

        # locked-in placement (your originals)
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
# Panda robot in the middle (tracks the train)
# -------------------------------
class RobotArm:
    """
    Panda at ring center. Downsamples IK (20 Hz) and interpolates joints each frame.
    """
    def __init__(self, env, cfg: RingCfg):
        self.env = env
        self.cfg = cfg

        self.robot = models.Panda()
        self.robot.base = SE3(cfg.center_x, cfg.center_y, 0.0)
        env.add(self.robot)

        env.add(Cylinder(radius=0.12, length=0.25,
                         pose=SE3(cfg.center_x, cfg.center_y, 0.125),
                         color=[0.2, 0.2, 0.22, 1.0]))

        self.q_home = self.robot.qz
        self.q_target = self.q_home.copy()  # last solved target

        # Smoothing & rates
        self.alpha = 0.20     # joint interpolation gain per frame
        self.ik_hz = 20       # IK solve rate (Hz)
        self._accum = 0.0     # internal accumulator for IK timing

        # tracking offsets
        self.hover_z = 0.30     # a touch lower for smaller ring
        self.lead = 0.10
        self.radial_extra = 0.25

        # slight tilt to avoid singularity
        self._tilt = SE3.RPY([pi * 0.85, 0, 0], order="xyz")

        # IK options: relax orientation a bit for faster convergence
        self._mask = [1, 1, 1, 0.5, 0.5, 0.5]  # [Tx,Ty,Tz,Rx,Ry,Rz] weights

    def goal_over_train(self, theta, train: "Train"):
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
        """Call every frame. Solves IK at ~ik_hz; otherwise interpolates toward last q_target."""
        # Accumulate time and decide whether to (re)solve IK
        self._accum += dt
        ik_period = 1.0 / self.ik_hz
        if self._accum >= ik_period:
            self._accum -= ik_period
            T_goal = self.goal_over_train(theta, train)
            sol = self.robot.ikine_LM(T_goal, q0=self.robot.q, mask=self._mask)
            if sol.success:
                self.q_target = sol.q
            else:
                # drift toward home if IK fails
                self.q_target = 0.98 * self.q_target + 0.02 * self.q_home

        # Interpolate joints toward the latest target (cheap & smooth)
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

    def loop(self):
        try:
            # Use a fixed dt; let Swift render at its own pace
            while True:
                self.train.update(self.dt)
                self.robot.update(self.dt, self.train.theta, self.train)  # new: pass dt, theta, train
                self.env.step(self.dt)  # pass dt explicitly for clarity
                # no time.sleep() here — smoother in most setups
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
    app.add_robot(cfg)   # << add the Panda in the middle
    app.loop()
