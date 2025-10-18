# train_demo_clean.py — Ring + Train + Panda follow + Animated Segment Toggles + ONE RTB URx pick/place
from math import pi, cos
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from spatialmath import SE3
from spatialgeometry import Mesh, Cuboid
from roboticstoolbox.backends import swift
from roboticstoolbox import models  # Panda model
import roboticstoolbox as rtb       # RTB robot models (UR3/UR5/etc.)
from pathlib import Path
import sys

# add the EVABOT directory to the Python path so it can be imported
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "Robots" / "EVABOT"))

from three_dof_meshes_with_cyl_viz import ThreeDOFMeshes


# =========================
# User-tweakable settings
# =========================
ROBOT_MODEL = "UR3"        # ← change to "UR5", "UR10", etc. (any rtb.models.<Name>)
BASE_SLOT   = 5            # ← place the robot aligned to this slot (0..n_segments-1)
BASE_MARGIN = 0.10         # ← radial margin outside the ring (meters)
HOVER_LIFT  = 0.20         # ← hover height for pick/place (meters)

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
        self.gapset = set()
        self.out_offset = 0.30
        self.anim = {}
        self.missing_slots = {6, 7}            # <- per your requirements
        self.slot_poses = [None] * self.cfg.n_segments

    def build(self):
        n = self.cfg.n_segments
        self.pieces = [None] * n
        for i in range(n):
            theta = 2 * pi * i / n
            T = (
                SE3(self.cfg.center_x, self.cfg.center_y, self.cfg.height_z)
                * SE3.Rz(self.cfg.global_yaw_deg * pi / 180)
                * SE3.Rz(theta)
                * SE3.Tx(self.cfg.radius)
                * SE3.Rz(pi / 2 + self.cfg.yaw_tweak_deg * pi / 180)
                * SE3.Rx(-pi / 2)
            )
            self.slot_poses[i] = T
            if i in self.missing_slots:
                self.pieces[i] = None
            else:
                m = Mesh(str(self.segment_path), pose=SE3(), scale=self.scale, color=self.cfg.color)
                self.env.add(m)
                self.pieces[i] = m
        self.gapset |= set(self.missing_slots)
        self.update()

    def _place_segment(self, i: int, r: float):
        if self.pieces[i] is None:
            return
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
        c = self.cfg
        r_in, r_out = c.radius, c.radius + self.out_offset
        finished = []
        n = c.n_segments
        for i in range(n):
            if i in self.missing_slots:
                self.gapset.add(i)
                continue
            if i in self.anim:
                st = self.anim[i]
                st['t'] += 0.05
                u = min(1.0, st['t'] / st['dur'])
                s = 0.5 - 0.5 * cos(pi * u)
                r = st['r0'] + (st['r1'] - st['r0']) * s
                self._place_segment(i, r)
                if u >= 1.0:
                    finished.append(i)
            else:
                r = r_out if (i in self.gapset) else r_in
                self._place_segment(i, r)
        for i in finished:
            final_r = self.anim[i]['r1']
            if abs(final_r - (c.radius + self.out_offset)) < 1e-9:
                self.gapset.add(i)
            else:
                if i not in self.missing_slots:
                    self.gapset.discard(i)
            del self.anim[i]

    def start_slide(self, idx: int, present: bool, dur: float = 0.8):
        if idx in self.missing_slots:
            print(f"slot {idx} is missing; toggle ignored")
            return
        c = self.cfg
        r_in, r_out = c.radius, c.radius + self.out_offset
        r_now = r_out if (idx in self.gapset) else r_in
        r_tar = r_in if present else r_out
        self.anim[idx] = {'t': 0.0, 'dur': max(0.05, dur), 'r0': r_now, 'r1': r_tar}

    def is_animating(self) -> bool:
        return bool(self.anim)

    def allowed_speed(self, theta_now: float, requested_speed: float) -> float:
        n = self.cfg.n_segments
        if n <= 0:
            return requested_speed
        gaps = self.gapset | self.missing_slots
        if not gaps:
            return requested_speed
        i = int(((theta_now % (2*pi)) / (2*pi)) * n) % n
        ahead = (i + 1) % n
        if i in gaps or ahead in gaps:
            return 0.0
        return requested_speed

    def segment_pose_with_offset(self, idx: int, theta_offset: float = 0.0):
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

    def theta_of_slot(self, idx: int) -> float:
        n = self.cfg.n_segments
        return 2 * pi * (idx % n) / n

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
# Linked reach blocks (orange)
# -------------------------------
class TrackLinkedZones:
    """Small red/orange blocks fixed to each base slot pose (do not move with IN/OUT)."""
    def __init__(self, env, ring: "TrackRing", cfg: "RingCfg"):
        self.env = env
        self.ring = ring
        self.cfg = cfg
        self.blocks = []
        self.block_offset = 0.255
        self.lift = -0.005
        self.theta_offset = -16.71 * pi / 180.0
        self.world_points = []
        self.overrides = {}  # index -> SE3 (if robot is carrying / repositioned)

    def set_override(self, i: int, T: SE3):
        self.overrides[i] = SE3(T)

    def clear_override(self, i: int):
        self.overrides.pop(i, None)

    def _block_pose_for_index(self, i: int):
        baseT = self.ring.slot_poses[i]
        return baseT * SE3.Rz(self.theta_offset) * SE3.Tx(self.block_offset) * SE3.Tz(self.lift)

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
            if i in self.overrides:
                T = self.overrides[i]
            else:
                T = self._block_pose_for_index(i)
            block.T = T.A
            p = T.t
            self.world_points[i] = (float(p[0]), float(p[1]), float(p[2]))

    def get_points(self):
        return list(self.world_points)

# -------------------------------
# Panda robot in the middle (follow-the-train waist)
# -------------------------------
class RobotArm:
    def __init__(self, env, cfg: RingCfg):
        self.env = env
        self.cfg = cfg
        self.robot = models.Panda()
        self.robot.base = SE3(cfg.center_x, cfg.center_y, 0.0)
        env.add(self.robot)
        self.q_follow = self.robot.qz.copy()
        DOWN_TILT, BACK_REACH, WRIST_PITCH = 0.5, -1.7, 2.1
        self.q_follow[1] = DOWN_TILT
        self.q_follow[2] = 0.40
        self.q_follow[3] = BACK_REACH
        self.q_follow[4] = 0.00
        self.q_follow[5] = WRIST_PITCH
        self.q_follow[6] = 0.70
        self._j0_min, self._j0_max = -2.8973, 2.8973

    @staticmethod
    def _wrap_pi(a): return (a + np.pi) % (2*np.pi) - np.pi

    def follow_train_yaw(self, theta, yaw_bias=0.0):
        q = self.q_follow.copy()
        j0 = self._wrap_pi(theta + yaw_bias)
        j0 = float(np.clip(j0, self._j0_min, self._j0_max))
        q[0] = j0
        self.robot.q = q

# -------------------------------
# Segment gate UI (buttons)
# -------------------------------
class TrackGateUI:
    def __init__(self, env, app, ring: "TrackRing"):
        self.env = env; self.app = app; self.ring = ring
        self.buttons = []
    def _label_for(self, idx): return f"Seg {idx} — {'OFF' if (idx in self.ring.gapset) else 'ON'}"
    def build(self):
        n = self.ring.cfg.n_segments
        for i in range(n):
            def make_cb(idx):
                def _cb(event=None):
                    if idx in self.ring.missing_slots:
                        print(f"slot {idx} is missing; toggle ignored"); return
                    self.app.toggle_segment(idx)
                return _cb
            btn = swift.Button(cb=make_cb(i), desc=self._label_for(i))
            self.env.add(btn); self.buttons.append(btn)
    def refresh_labels(self):
        for i, btn in enumerate(self.buttons): btn.desc = self._label_for(i)

# -------------------------------
# App wrapper
# -------------------------------
class App:
    def __init__(self):
        self.env = swift.Swift()
        self.env.launch(realtime=True, port=0)
        self.dt = 0.05

    # Scene
    def add_floor(self, size=(8, 8, 0.02), z=-0.01):
        self.env.add(Cuboid(scale=list(size), pose=SE3(0, 0, z), color=[0.9, 0.9, 0.95, 1]))
    def add_ring(self, seg_path, cfg: RingCfg):
        self.ring = TrackRing(self.env, seg_path, cfg); self.ring.build()
    def add_train(self, train_path, cfg: RingCfg):
        self.train = Train(self.env, train_path, cfg)
    def add_robot(self, cfg: RingCfg):
        self.robot = RobotArm(self.env, cfg)
    def add_linked_zones(self, cfg: RingCfg):
        self.zones = TrackLinkedZones(self.env, self.ring, cfg); self.zones.build()
    def add_track_gates(self):
        self.gates = TrackGateUI(self.env, self, self.ring); self.gates.build()

    # Ring toggle
    def toggle_segment(self, idx: int):
        want_present = (idx in self.ring.gapset)
        self.ring.start_slide(idx, present=want_present, dur=0.8)

    # ====== ONE outer robot (RTB model switchable by ROBOT_MODEL) ======
    def add_outer_arm(self, cfg: RingCfg, model_name: str = ROBOT_MODEL, slot_index: int = BASE_SLOT, margin: float = BASE_MARGIN):
        # build model by name from rtb.models
        try:
            RobotClass = getattr(rtb.models, model_name)
        except AttributeError:
            raise RuntimeError(f"rtb.models has no '{model_name}'. Try 'UR3', 'UR5', 'UR10', etc.")
        self.arm = RobotClass()

        theta = 2 * np.pi * (slot_index % cfg.n_segments) / cfg.n_segments
        self.arm.base = (
            SE3(cfg.center_x, cfg.center_y, cfg.height_z)
            * SE3.Rz(theta)
            * SE3.Tx(cfg.radius + margin)
            * SE3.Rz(np.pi)   # face inward
        )
        self.env.add(self.arm)
        # neutral-ish pose
        try:
            self.arm.q = [0, -1.57, 1.57, 0, 1.57, 0]
        except Exception:
            pass

    # ===== IK utilities for robot pick/place =====
    def _T_in_base(self, robot, T_world: SE3) -> SE3:  return SE3(robot.base).inv() * T_world
    def _T_out_of_base(self, robot, T_in_base: SE3) -> SE3:  return SE3(robot.base) * T_in_base

    def _ik_to(self, robot, T_goal: SE3, steps=80, label=""):
        sol = robot.ikine_LM(self._T_in_base(robot, T_goal))
        if not sol.success:
            print(f"[{ROBOT_MODEL} IK] failed at {label or 'target'}"); return None
        q_goal = sol.q
        traj = rtb.jtraj(robot.q, q_goal, steps)
        for q in traj.q:
            robot.q = q
            if getattr(self, "_carry", None):
                Ttool = robot.fkine(robot.q)
                world_Ttool = self._T_out_of_base(robot, Ttool)
                block, T_tool_to_block, idx = self._carry
                T_now = world_Ttool * T_tool_to_block
                block.T = T_now.A
                self.zones.set_override(idx, T_now)
            self.env.step(self.dt)
        return q_goal

    # ===== public: keep name as requested =====
    def ur3_pick_and_place_orange(self, src=5, dst=7, lift=HOVER_LIFT):
        """
        Visually pick orange[src] with the selected outer robot (ROBOT_MODEL) and place at orange[dst].
        Uses a tool-down orientation for robust IK.
        """
        robot = self.arm
        block = self.zones.blocks[src]

        # world poses of source/target orange frames
        T_src_ref = self.zones._block_pose_for_index(src)
        T_dst_ref = self.zones._block_pose_for_index(dst)
        p_src, p_dst = T_src_ref.t, T_dst_ref.t

        TOOL_DOWN = SE3.RPY([np.pi/2, 0, -np.pi], order="xyz")
        T_src       = SE3(p_src) * TOOL_DOWN
        T_dst       = SE3(p_dst) * TOOL_DOWN
        T_src_hover = SE3([p_src[0], p_src[1], p_src[2] + lift]) * TOOL_DOWN
        T_dst_hover = SE3([p_dst[0], p_dst[1], p_dst[2] + lift]) * TOOL_DOWN
        T_via       = SE3(((T_src_hover.t + T_dst_hover.t) * 0.5)) * TOOL_DOWN

        # approach/grasp
        if self._ik_to(robot, T_src_hover, steps=70, label="src_hover") is None: return
        if self._ik_to(robot, T_src,       steps=60, label="src_down")  is None: return

        # measure tool→block offset once at grasp
        Ttool_world = self._T_out_of_base(robot, robot.fkine(robot.q))
        T_tool_to_block = Ttool_world.inv() * SE3(block.T)
        self._carry = (block, T_tool_to_block, src)  # start carrying
        self.zones.set_override(src, SE3(block.T))   # prevent snap-back

        # transit
        if self._ik_to(robot, T_src_hover, steps=50, label="lift")      is None: return
        if self._ik_to(robot, T_via,       steps=70, label="via")       is None: return
        if self._ik_to(robot, T_dst_hover, steps=70, label="dst_hover") is None: return
        if self._ik_to(robot, T_dst,       steps=60, label="dst_down")  is None: return

        # release at destination
        self._carry = None
        block.T = T_dst.A
        self.zones.set_override(src, T_dst)          # leave it there (even if overlaps)

        # clear up
        self._ik_to(robot, T_dst_hover, steps=50, label="clear")
        print(f"{ROBOT_MODEL} moved orange block {src} → {dst}")

    # Main loop
    def loop(self):
        try:
            while True:
                gated_v = self.ring.allowed_speed(self.train.theta, self.train.speed)
                self.train.step(self.dt, gated_v)
                if hasattr(self, "robot"):
                    self.robot.follow_train_yaw(self.train.theta, yaw_bias=-pi/3.5)
                was_anim = self.ring.is_animating()
                self.ring.update()
                self.zones.update()
                if was_anim and not self.ring.is_animating() and hasattr(self, 'gates'):
                    self.gates.refresh_labels()
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

    # one outer arm — model & placement are configurable at the top
    app.add_outer_arm(cfg, model_name=ROBOT_MODEL, slot_index=BASE_SLOT, margin=BASE_MARGIN)

    app.add_track_gates()
    app.add_robot(cfg)          # Panda (center) with follow posture
    app.add_linked_zones(cfg)   # orange blocks

    # Button to run the move (kept the same name/signature)
    btn_ur3move = swift.Button(
        cb=lambda e=None: app.ur3_pick_and_place_orange(5, 7),
        desc=f"{ROBOT_MODEL}: orange 5 → 7"
    )
    app.env.add(btn_ur3move)

    app.loop()
