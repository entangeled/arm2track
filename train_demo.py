# train_demo.py — Ring + Train + Panda follow + Animated Segment Toggles

from math import pi, cos
from pathlib import Path
from dataclasses import dataclass
import numpy as np

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
    radius: float = 0.8
    height_z: float = 0.10
    yaw_tweak_deg: float = 69.65
    center_x: float = 0.0
    center_y: float = 0.0
    global_yaw_deg: float = 0.0
    unit_scale: float = 0.0003
    color: tuple = (0.35, 0.35, 0.38, 1.0)


# -------------------------------
# Track ring (with animation)
# -------------------------------
class TrackRing:
    def __init__(self, env, segment_path, cfg: RingCfg):
        self.env = env
        self.cfg = cfg
        self.segment_path = segment_path
        s = cfg.unit_scale
        self.scale = [s, s, s]
        self.pieces = []

        self.gapset = set()       # indices that are OUT
        self.out_offset = 0.30    # slide distance (meters)
        # idx -> {'t':0.0, 'dur':0.8, 'r0':..., 'r1':...}
        self.anim = {}

    def build(self):
        for _ in range(self.cfg.n_segments):
            m = Mesh(str(self.segment_path), pose=SE3(), scale=self.scale, color=self.cfg.color)
            self.env.add(m)
            self.pieces.append(m)
        self.update()

    def _place_segment(self, i: int, r: float):
        c = self.cfg
        n = c.n_segments
        theta = 2 * pi * i / n
        T = (
            SE3(c.center_x, c.center_y, c.height_z)
            * SE3.Rz(c.global_yaw_deg * pi / 180)
            * SE3.Rz(theta)
            * SE3.Tx(r)
            * SE3.Rz(pi / 2 + c.yaw_tweak_deg * pi / 180)
            * SE3.Rx(-pi / 2)
        )
        self.pieces[i].T = T.A

    def update(self):
        """Per-frame placement and animation."""
        c = self.cfg
        r_in  = c.radius
        r_out = c.radius + self.out_offset

        finished = []
        for i in range(len(self.pieces)):
            if i in self.anim:
                st = self.anim[i]
                st['t'] += 0.05   # matches App.dt
                u = min(1.0, st['t'] / st['dur'])
                s = 0.5 - 0.5 * cos(pi * u)  # cosine ease
                r = st['r0'] + (st['r1'] - st['r0']) * s
                self._place_segment(i, r)
                if u >= 1.0:
                    finished.append(i)
            else:
                r = r_out if (i in self.gapset) else r_in
                self._place_segment(i, r)

        # commit logic when animations finish
        for i in finished:
            final_r = self.anim[i]['r1']
            if abs(final_r - r_out) < 1e-9:
                self.gapset.add(i)      # OUT
            else:
                self.gapset.discard(i)  # IN
            del self.anim[i]

    def start_slide(self, idx: int, present: bool, dur: float = 0.8):
        """present=True -> slide to IN; present=False -> slide to OUT"""
        c = self.cfg
        r_in  = c.radius
        r_out = c.radius + self.out_offset
        r_now = r_out if (idx in self.gapset) else r_in
        r_tar = r_in if present else r_out
        self.anim[idx] = {'t': 0.0, 'dur': max(0.05, dur), 'r0': r_now, 'r1': r_tar}

    def is_animating(self) -> bool:
        return bool(self.anim)

    def allowed_speed(self, theta_now: float, requested_speed: float) -> float:
        """Stop the train if current or next segment is OUT."""
        if not self.gapset:
            return requested_speed
        n = self.cfg.n_segments
        i = int(((theta_now % (2*pi)) / (2*pi)) * n) % n
        ahead = (i + 1) % n
        if i in self.gapset or ahead in self.gapset:
            return 0.0
        return requested_speed

    def segment_pose_with_offset(self, idx: int, theta_offset: float = 0.0):
        """Pose utility used by the red blocks."""
        c = self.cfg
        n = c.n_segments
        theta = 2 * pi * idx / n + theta_offset
        r = (c.radius + self.out_offset) if (idx in self.gapset) else c.radius
        return (
            SE3(c.center_x, c.center_y, c.height_z)
            * SE3.Rz(c.global_yaw_deg * pi / 180)
            * SE3.Rz(theta)
            * SE3.Tx(r)
            * SE3.Rz(pi / 2 + c.yaw_tweak_deg * pi / 180)
            * SE3.Rx(-pi / 2)
        )


# -------------------------------
# Train cart running on ring
# -------------------------------
class Train:
    def __init__(self, env, mesh_path, cfg: RingCfg):
        self.env = env
        self.cfg = cfg
        self.theta = 0.0
        self.speed = 1.0  # rad/s
        s = cfg.unit_scale

        self.mesh = Mesh(str(mesh_path), pose=SE3(), scale=[s, s, s], color=[0.12, 0.12, 0.12, 1])
        env.add(self.mesh)

        self.height_offset = 0.10
        self.radial_offset = 0.11
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

    def step(self, dt, speed_override=None):
        v = self.speed if speed_override is None else speed_override
        self.theta = (self.theta + v * dt) % (2 * pi)
        self.mesh.T = self.pose_on_ring(self.theta).A


# -------------------------------
# Linked reach blocks (attached to tracks)
# -------------------------------
class TrackLinkedZones:
    """Small red blocks that track each segment with a fixed offset."""
    def __init__(self, env, ring: "TrackRing", cfg: "RingCfg"):
        self.env = env
        self.ring = ring
        self.cfg = cfg
        self.blocks = []

        # offsets you’ve been using
        self.block_offset = 0.255
        self.lift = -0.005
        self.theta_offset = -16.71 * pi / 180.0

        self.world_points = []

    def _block_pose_for_index(self, i: int):
        segT = self.ring.segment_pose_with_offset(i, self.theta_offset)
        return segT * SE3.Tx(self.block_offset) * SE3.Tz(self.lift)

    def build(self):
        n = self.cfg.n_segments
        for i in range(n):
            T = self._block_pose_for_index(i)
            cube = Cuboid([0.05, 0.05, 0.02], pose=T, color=[0.95, 0.35, 0.35, 0.9])
            self.env.add(cube)
            self.blocks.append(cube)
            p = T.t
            self.world_points.append((float(p[0]), float(p[1]), float(p[2])))

    def update(self):
        for i, block in enumerate(self.blocks):
            T = self._block_pose_for_index(i)
            block.T = T.A
            p = T.t
            self.world_points[i] = (float(p[0]), float(p[1]), float(p[2]))

    def get_points(self):
        return list(self.world_points)


# -------------------------------
# Panda robot: spin waist to follow the train
# -------------------------------
class RobotArm:
    def __init__(self, env, cfg: RingCfg):
        self.env = env
        self.cfg = cfg

        self.robot = models.Panda()
        self.robot.base = SE3(cfg.center_x, cfg.center_y, 0.0)
        env.add(self.robot)

        # pedestal visual
        env.add(Cylinder(radius=0.12, length=0.25,
                         pose=SE3(cfg.center_x, cfg.center_y, 0.125),
                         color=[0.2, 0.2, 0.22, 1.0]))

        if hasattr(self.robot, "qr") and self.robot.qr is not None:
            q_ready = self.robot.qr.copy()
        else:
            q_ready = self.robot.qz.copy()
        self.q_follow = q_ready


        # joint-0 limits
        self._j0_min = -2.8973
        self._j0_max =  2.8973

    @staticmethod
    def _wrap_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def follow_train_yaw(self, theta, yaw_bias=0.0):
        """Rotate only joint 0 so the arm spins around the pedestal."""
        q = self.q_follow.copy()
        j0 = self._wrap_pi(theta + yaw_bias)
        j0 = float(np.clip(j0, self._j0_min, self._j0_max))
        q[0] = j0
        self.robot.q = q


# -------------------------------
# Segment gate UI (buttons)
# -------------------------------
class TrackGateUI:
    """Buttons toggle segments with animation; labels show IN/OUT."""
    def __init__(self, env, app, ring: "TrackRing"):
        self.env = env
        self.app = app
        self.ring = ring
        self.buttons = []

    def _label_for(self, idx):
        return f"Seg {idx} — {'OFF' if (idx in self.ring.gapset) else 'ON'}"

    def build(self):
        n = self.ring.cfg.n_segments
        for i in range(n):
            def make_cb(idx):
                def _cb(event=None):
                    self.app.toggle_segment(idx)  # simple, no robot
                return _cb
            btn = swift.Button(cb=make_cb(i), desc=self._label_for(i))
            self.env.add(btn)
            self.buttons.append(btn)

    def refresh_labels(self):
        for i, btn in enumerate(self.buttons):
            btn.desc = self._label_for(i)


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
        self.gates = TrackGateUI(self.env, self, self.ring)
        self.gates.build()

    # Called by UI button
    def toggle_segment(self, idx: int):
        want_present = (idx in self.ring.gapset)  # OUT -> IN
        self.ring.start_slide(idx, present=want_present, dur=0.8)

    def loop(self):
        try:
            while True:
                # train motion (with gating)
                gated_v = self.ring.allowed_speed(self.train.theta, self.train.speed)
                self.train.step(self.dt, gated_v)

                # have the robot spin its waist to follow the train
                if hasattr(self, "robot"):
                    self.robot.follow_train_yaw(self.train.theta, yaw_bias=0.0)

                # remember if something was animating
                was_anim = self.ring.is_animating()

                # update ring (animation) and blocks
                self.ring.update()
                self.zones.update()

                # if an animation finished this frame, refresh button labels
                if was_anim and not self.ring.is_animating() and hasattr(self, 'gates'):
                    self.gates.refresh_labels()

                # render
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

    app.add_track_gates()      # side buttons
    app.add_robot(cfg)         # Panda that spins with the train
    app.add_linked_zones(cfg)  # red blocks following segments

    app.loop()
