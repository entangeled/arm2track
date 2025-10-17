# train_demo.py — Ring + Train + Panda + Linked Track Zones + Segment Gates

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
        self.gapset = set()   # indices of segments currently OUT (missing)

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

    # ---- gating helpers (segment IN/OUT + speed gate) ----
    def theta_to_seg(self, theta: float) -> int:
        n = self.cfg.n_segments
        t = (theta % (2 * pi))
        return int((t / (2 * pi)) * n) % n

    def set_segment_present(self, idx: int, present: bool):
        """
        Move a segment OUT (present=False) by pushing it radially outward;
        move it back IN (present=True) to the exact ring pose.
        """
        out_offset = 0.6  # meters outward for visible gap
        theta = 2 * pi * idx / self.cfg.n_segments
        c = self.cfg

        if present:
            self.pieces[idx].T = self.pose_for_theta(theta).A
            self.gapset.discard(idx)
        else:
            T = (
                SE3(c.center_x, c.center_y, c.height_z)
                * SE3.Rz(c.global_yaw_deg * pi / 180)
                * SE3.Rz(theta)
                * SE3.Tx(c.radius + out_offset)
                * SE3.Rz(pi / 2 + c.yaw_tweak_deg * pi / 180)
                * SE3.Rx(-pi / 2)
            )
            self.pieces[idx].T = T.A
            self.gapset.add(idx)

    def allowed_speed(self, theta_now: float, requested_speed: float) -> float:
        """
        If the current or next segment is OUT, force speed to 0 (stop train).
        Otherwise, allow requested_speed.
        """
        if not self.gapset:
            return requested_speed
        n = self.cfg.n_segments
        i = self.theta_to_seg(theta_now)
        ahead = (i + 1) % n
        if i in self.gapset or ahead in self.gapset:
            return 0.0
        return requested_speed


# -------------------------------
# Train cart running on ring
# -------------------------------
class Train:
    """Train perfectly aligned on the track and driven by angular speed."""
    def __init__(self, env, mesh_path, cfg: RingCfg):
        self.env = env
        self.cfg = cfg
        self.theta = 0.0
        self.speed = 1.0  # rad/s
        s = cfg.unit_scale

        self.mesh = Mesh(str(mesh_path), pose=SE3(), scale=[s, s, s], color=[0.12, 0.12, 0.12, 1])
        env.add(self.mesh)

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
        # legacy updater (uses self.speed)
        self.theta = (self.theta + self.speed * dt) % (2 * pi)
        self.mesh.T = self.pose_on_ring(self.theta).A

    def step(self, dt, speed_override=None):
        v = self.speed if speed_override is None else speed_override
        self.theta = (self.theta + v * dt) % (2 * pi)
        self.mesh.T = self.pose_on_ring(self.theta).A


# -------------------------------
# Linked reach blocks (attached to tracks)
# -------------------------------
class TrackLinkedZones:
    """
    9 red blocks positioned at the midline of each track segment.
    They move with the segments and expose world points you can use later.
    """
    def __init__(self, env, ring: "TrackRing", cfg: "RingCfg"):
        self.env = env
        self.ring = ring
        self.cfg = cfg
        self.blocks = []

        # remembered dimensions (do not change)
        self.block_offset = 0.255                 # outward from ring centreline (m)
        self.lift = -0.005                        # slight drop into the groove (m)
        self.theta_offset = -16.71 * pi / 180.0   # rotation offset (rad)

        self.thetas = []
        self.world_points = []

    def build(self):
        c = self.cfg
        n = c.n_segments
        for i in range(n):
            base_theta = 2 * pi * i / n + self.theta_offset
            self.thetas.append(base_theta)
            T = (
                SE3(c.center_x, c.center_y, 0.0)
                * SE3.Rz(c.global_yaw_deg * pi / 180)
                * SE3.Rz(base_theta)
                * SE3.Tx(c.radius + self.block_offset)
                * SE3.Tz(c.height_z + self.lift)
            )
            cube = Cuboid([0.05, 0.05, 0.02], pose=T, color=[0.95, 0.35, 0.35, 0.9])
            self.env.add(cube)
            self.blocks.append(cube)
            p = T.t
            self.world_points.append((float(p[0]), float(p[1]), float(p[2])))

    def update(self):
        c = self.cfg
        n = c.n_segments
        for i, block in enumerate(self.blocks):
            seg_pose = self.ring.pose_for_theta(2 * pi * i / n + self.theta_offset)
            T = seg_pose * SE3.Tx(self.block_offset) * SE3.Tz(self.lift)
            block.T = T.A
            p = T.t
            self.world_points[i] = (float(p[0]), float(p[1]), float(p[2]))

    def get_points(self):
        return list(self.world_points)


# -------------------------------
# Panda robot in the middle (tracks the train)
# -------------------------------
class RobotArm:
    """Panda robot following the train (lightweight IK)."""
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
        self.q_target = self.q_home.copy()
        self.alpha = 0.20
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
        if self._accum >= 1.0 / self.ik_hz:
            self._accum = 0.0
            T_goal = self.goal_over_train(theta, train)
            sol = self.robot.ikine_LM(T_goal, q0=self.robot.q, mask=self._mask)
            if sol.success:
                self.q_target = sol.q
            else:
                self.q_target = 0.98 * self.q_target + 0.02 * self.q_home
        self.robot.q = (1 - self.alpha) * self.robot.q + self.alpha * self.q_target


# -------------------------------
# Segment gate UI (IN/OUT sliders)
# -------------------------------
class TrackGateUI:
    """
    One slider per segment: 1 = IN (on the ring), 0 = OUT (removed).
    Moving a slider live-updates the track and the ring's gap set.
    """
    def __init__(self, env, ring: "TrackRing"):
        self.env = env
        self.ring = ring
        self.sliders = []

    def build(self):
        n = self.ring.cfg.n_segments
        for i in range(n):
            def _cb(v, i=i):
                present = (v >= 0.5)
                self.ring.set_segment_present(i, present)
            s = swift.Slider(cb=_cb, min=0, max=1, step=1, value=1,
                             desc=f"Seg {i} — 1=IN, 0=OUT")
            self.env.add(s)
            self.sliders.append(s)


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
        self.zones = TrackLinkedZones(self.env, self.ring, cfg)
        self.zones.build()

    def add_track_gates(self):
        self.gates = TrackGateUI(self.env, self.ring)
        self.gates.build()

    def loop(self):
        try:
            while True:
                # gate speed if the current/next segment is OUT
                gated_v = self.ring.allowed_speed(self.train.theta, self.train.speed)
                self.train.step(self.dt, gated_v)

                if hasattr(self, "robot"):
                    self.robot.update(self.dt, self.train.theta, self.train)
                if hasattr(self, "zones"):
                    self.zones.update()

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

    app.add_track_gates()      # sliders: set segments IN/OUT
    app.add_robot(cfg)         # Panda in the middle
    app.add_linked_zones(cfg)  # red reference blocks (move with segments)

    app.loop()
