# tracks_pick_place.py
# Panda + Swift — pick a track at slot A, carry with EE, place at slot B (brick-style logic)

from math import pi
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from spatialmath import SE3
from spatialgeometry import Mesh, Cuboid, Cylinder
from roboticstoolbox.backends import swift
from roboticstoolbox import models


# --------- small safety shims (like your brick helpers) ----------
def _as_pose_matrix(T) -> np.ndarray:
    if hasattr(T, "A"):
        A = np.asarray(T.A, dtype=float)
    else:
        A = np.asarray(T, dtype=float)
    if A.shape == (3, 4):
        A = np.vstack([A, [0, 0, 0, 1]])
    if A.shape != (4, 4):
        A = np.eye(4, dtype=float)
    return A

def _as_SE3(T) -> SE3:
    return SE3(_as_pose_matrix(T))

def _set_T(node, T):
    node.T = _as_pose_matrix(T)


# ---------------- config ----------------
@dataclass
class RingCfg:
    n_segments: int = 9
    radius: float = 0.80
    height_z: float = 0.10
    yaw_tweak_deg: float = 69.65
    center_x: float = 0.0
    center_y: float = 0.0
    global_yaw_deg: float = 0.0
    unit_scale: float = 0.0003
    seg_color: tuple = (0.35, 0.35, 0.38, 1.0)

def slot_pose(i: int, c: RingCfg) -> SE3:
    th = 2 * pi * i / c.n_segments
    return (SE3(c.center_x, c.center_y, c.height_z)
            * SE3.Rz(c.global_yaw_deg * pi/180)
            * SE3.Rz(th)
            * SE3.Tx(c.radius)
            * SE3.Rz(pi/2 + c.yaw_tweak_deg * pi/180)
            * SE3.Rx(-pi/2))


# ------------- ring with occupancy -------------
class TrackRing:
    def __init__(self, env, seg_path: Path, cfg: RingCfg, missing_slots=(2, 6)):
        self.env = env
        self.cfg = cfg
        self.slots = [slot_pose(i, cfg) for i in range(cfg.n_segments)]
        self.pieces = []                         # list[Mesh] piece id -> node
        self.slot_to_pid = [None]*cfg.n_segments # slot idx -> piece id or None
        self.pid_to_slot = {}                    # reverse map

        s = cfg.unit_scale
        present = [i for i in range(cfg.n_segments) if i not in missing_slots]
        for pid, si in enumerate(present):
            m = Mesh(str(seg_path), pose=self.slots[si],
                     scale=[s, s, s], color=cfg.seg_color)
            env.add(m)
            self.pieces.append(m)
            self.slot_to_pid[si] = pid
            self.pid_to_slot[pid] = si

    def snap_pid_to_slot(self, pid: int, slot_idx: int):
        _set_T(self.pieces[pid], self.slots[slot_idx])
        # fix maps
        old = self.pid_to_slot.get(pid, None)
        if old is not None and self.slot_to_pid[old] == pid:
            self.slot_to_pid[old] = None
        self.slot_to_pid[slot_idx] = pid
        self.pid_to_slot[pid] = slot_idx

    def pid_at_slot(self, slot_idx: int):
        return self.slot_to_pid[slot_idx]

    def slot_SE3(self, slot_idx: int) -> SE3:
        return self.slots[slot_idx]


# ------------- robot (brick-style motion) -------------
class Robot:
    def __init__(self, env, cfg: RingCfg):
        self.robot = models.Panda()
        self.robot.base = SE3(cfg.center_x, cfg.center_y, 0.0)
        env.add(self.robot)

        env.add(Cylinder(radius=0.12, length=0.25,
                         pose=SE3(cfg.center_x, cfg.center_y, 0.125),
                         color=[0.2,0.2,0.22,1]))

        self.alpha = 0.35
        self.ik_hz = 20
        self._accum = 0.0
        self._mask = [1,1,1,0.5,0.5,0.5]
        self.q_home = self.robot.qz
        self.q_tgt  = self.q_home.copy()
        # tool down, jaws face outward (like before)
        self._tilt  = SE3.RPY([-pi/2, 0, pi], order="xyz")

    def ee(self) -> SE3:
        return SE3(self.robot.fkine(self.robot.q).A)

    def set_goal(self, T: SE3):
        self.goal = _as_SE3(T)

    def at_goal(self, pos_tol=0.01, rot_tol=7.0):
        Acur  = _as_pose_matrix(self.ee())
        Agoal = _as_pose_matrix(self.goal)
        p_err = float(np.linalg.norm(Acur[0:3,3] - Agoal[0:3,3]))
        Rerr  = Acur[0:3,0:3].T @ Agoal[0:3,0:3]
        tr    = float(np.trace(Rerr)); tr = max(-1.0, min(3.0, tr))
        ang   = float(np.degrees(np.arccos(max(-1.0, min(1.0, (tr-1)/2)))))
        return (p_err <= pos_tol) and (ang <= rot_tol)

    def update(self, dt):
        self._accum += dt
        if self._accum >= 1.0/self.ik_hz:
            self._accum = 0.0
            sol = self.robot.ikine_LM(self.goal, q0=self.robot.q, mask=self._mask)
            if sol.success:
                self.q_tgt = sol.q
            else:
                self.q_tgt = 0.98*self.q_tgt + 0.02*self.q_home
        self.robot.q = (1-self.alpha)*self.robot.q + self.alpha*self.q_tgt

    # pose helpers
    def hover_over(self, Tslot: SE3, hover=0.22) -> SE3:
        px,py,pz = map(float, Tslot.t[:3])
        return SE3(px,py,pz+hover) * self._tilt

    def down_at(self, Tslot: SE3, dz=0.06) -> SE3:
        px,py,pz = map(float, Tslot.t[:3])
        return SE3(px,py,pz+dz) * self._tilt


# ------------- app orchestrator -------------
class App:
    def __init__(self, missing=(2,6), dt=0.05):
        self.env = swift.Swift()
        self.env.launch(realtime=True, port=0)
        self.dt = dt
        self.cfg = RingCfg()
        self.ring = None
        self.robot = None
        self.holding = None   # (pid, local_offset SE3)

    def add_floor(self):
        self.env.add(Cuboid([8,8,0.02], pose=SE3(0,0,-0.01), color=[0.9,0.9,0.95,1]))

    def build(self, missing=(2,6)):
        here = Path(__file__).resolve().parent
        seg_path = here / "track_segment.STL"
        train_path = here / "train.STL"

        self.add_floor()
        self.ring  = TrackRing(self.env, seg_path, self.cfg, missing_slots=missing)
        self.robot = Robot(self.env, self.cfg)

        # little train just for vibe
        s = self.cfg.unit_scale
        train = Mesh(str(train_path), pose=SE3(self.cfg.center_x+0.1, self.cfg.center_y+0.6, self.cfg.height_z+0.11),
                     scale=[s,s,s], color=[0.12,0.12,0.12,1])
        self.env.add(train)

    # ---- inner stepping ----
    def _tick(self, follow=False):
        self.robot.update(self.dt)
        if follow and self.holding is not None:
            pid, L = self.holding
            _set_T(self.ring.pieces[pid], self.robot.ee() * L)
        self.env.step(self.dt)

    def _move_to(self, Tgoal: SE3, follow=False, timeout=6.0):
        self.robot.set_goal(Tgoal)
        t = 0.0
        while not self.robot.at_goal(0.02) and t < timeout:
            self._tick(follow=follow)
            t += self.dt

    # ---- PICK (like bricks: measure offset & attach) ----
    def pick_from_slot(self, slot_idx: int):
        pid = self.ring.pid_at_slot(slot_idx)
        if pid is None:
            raise RuntimeError(f"No piece at slot {slot_idx} to pick")

        Tslot = self.ring.slot_SE3(slot_idx)
        self._move_to(self.robot.hover_over(Tslot), follow=False)
        self._move_to(self.robot.down_at(Tslot),    follow=False)

        # latch (piece follows EE with constant local transform)
        Tee  = self.robot.ee()
        Pcur = SE3(self.ring.pieces[pid].T)
        self.holding = (pid, Tee.inv() * Pcur)

        # lift
        self._move_to(self.robot.hover_over(Tslot), follow=True)
        # free the source slot in the map
        self.ring.slot_to_pid[slot_idx] = None
        self.ring.pid_to_slot[pid] = None
        return pid

    # ---- PLACE (carry following EE, then snap & release) ----
    def place_into_slot(self, slot_idx: int):
        if self.holding is None:
            raise RuntimeError("Not holding anything to place")
        pid, L = self.holding
        Tdst = self.ring.slot_SE3(slot_idx)

        self._move_to(self.robot.hover_over(Tdst), follow=True)
        self._move_to(self.robot.down_at(Tdst),    follow=True)

        # release: snap to canonical slot, clear hold
        self.ring.snap_pid_to_slot(pid, slot_idx)
        self.holding = None

        # small lift ready for next move
        self._move_to(self.robot.hover_over(Tdst), follow=False)

    # ---- plan & execute (no home between moves) ----
    def run_pairs(self, pairs):
        # small settle first
        for _ in range(10):
            self._tick()

        for src, dst in pairs:
            print(f"[move] {src} -> {dst}")
            pid = self.pick_from_slot(src)
            self.place_into_slot(dst)

        # idle end
        for _ in range(20):
            self._tick()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app = App(missing=(2,6))
    app.build(missing=(2,6))

    # Your sequence: grab from 6→8, 5→7, 4→6, 3→5, 2→4
    pairs = [(6, 8), (5, 7), (4, 6), (3, 5), (2, 4)]
    app.run_pairs(pairs)
