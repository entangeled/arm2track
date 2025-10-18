# train_demo.py — Ring + Train + Panda follow + Animated Segment Toggles

from math import pi, cos
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from spatialmath import SE3
from spatialgeometry import Mesh, Cuboid, Cylinder
from roboticstoolbox.backends import swift
from roboticstoolbox import models  # Panda model
import roboticstoolbox as rtb  # for UR3 models
UR3 = rtb.models.UR3




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

        # --- NEW: two slots physically missing (adjust indices if you like)
        self.missing_slots = {6, 7}

        # --- NEW: per-slot world SE3 at base radius (no OUT offset)
        self.slot_poses = [None] * self.cfg.n_segments


    def build(self):
        n = self.cfg.n_segments

        # re-init pieces so indices align 1:1 with slots
        self.pieces = [None] * n

        # base-radius SE3 per slot (no OUT offset), also spawn meshes if not missing
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
                # no mesh spawned for missing slots
                self.pieces[i] = None
            else:
                m = Mesh(str(self.segment_path), pose=SE3(), scale=self.scale, color=self.cfg.color)
                self.env.add(m)
                self.pieces[i] = m

        # Treat missing slots as permanent gaps for gating
        self.gapset |= set(self.missing_slots)

        self.update()

    def _place_segment(self, i: int, r: float):
        if self.pieces[i] is None:
            return  # missing slot: nothing to place
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
        n = c.n_segments
        for i in range(n):
            if i in self.missing_slots:
                # keep them flagged as gaps; nothing to move
                self.gapset.add(i)
                continue

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
                # never clear gaps for missing slots
                if i not in self.missing_slots:
                    self.gapset.discard(i)  # IN
            del self.anim[i]


    def start_slide(self, idx: int, present: bool, dur: float = 0.8):
        """present=True -> slide to IN; present=False -> slide to OUT"""
        if idx in self.missing_slots:
            print(f"slot {idx} is missing; toggle ignored")
            return

        c = self.cfg
        r_in  = c.radius
        r_out = c.radius + self.out_offset
        r_now = r_out if (idx in self.gapset) else r_in
        r_tar = r_in if present else r_out
        self.anim[idx] = {'t': 0.0, 'dur': max(0.05, dur), 'r0': r_now, 'r1': r_tar}


    def is_animating(self) -> bool:
        return bool(self.anim)

    def allowed_speed(self, theta_now: float, requested_speed: float) -> float:
        """Stop the train if current or next segment is OUT or missing."""
        n = self.cfg.n_segments
        if n <= 0:
            return requested_speed

        gaps = self.gapset | self.missing_slots  # <- include missing explicitly
        if not gaps:
            return requested_speed

        i = int(((theta_now % (2*pi)) / (2*pi)) * n) % n
        ahead = (i + 1) % n
        if i in gaps or ahead in gaps:
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
    
    def theta_of_slot(self, idx: int) -> float:
        n = self.cfg.n_segments
        return 2 * pi * (idx % n) / n

    def detach_segment(self, idx: int):
        """Remove mesh handle from slot (keeps gap). Returns Mesh or None."""
        if idx in self.missing_slots:
            return None
        m = self.pieces[idx]
        if m is None:
            return None
        self.pieces[idx] = None
        # ensure it's considered OUT
        self.gapset.add(idx)
        return m

    def attach_segment(self, idx: int, mesh):
        """Place mesh handle into slot, as IN (r_in)."""
        if idx in self.missing_slots:
            # shouldn't happen: target must not be permanently missing
            return
        self.pieces[idx] = mesh
        # mark as IN (will slide to r_in in place_slide_in state)
        self.gapset.discard(idx)
        # snap mesh to base radius now (visuals settle via slide_in)
        self._place_segment(idx, self.cfg.radius)

# --- NEW: orchestrator that creates a 2-slot move to fill a gap ---
class GapOrchestrator:
    """
    Watches the ring; if there is a gap, ask the nearest arm's mover to
    move a donor piece from (gap-2) -> (gap) along the outside arc.
    """
    def __init__(self, app: "App"):
        self.app = app
        self.cooldown = 0.0

    def _theta_of_slot(self, idx: int) -> float:
        n = self.app.ring.cfg.n_segments
        return 2 * pi * (idx % n) / n

    def step(self, dt: float):
        # simple cooldown so we don't spam jobs
        self.cooldown = max(0.0, self.cooldown - dt)
        if self.cooldown > 0.0:
            return

        ring = self.app.ring
        n = ring.cfg.n_segments

        # find the first real gap: missing or OUT or piece==None
        gaps = []
        for i in range(n):
            if i in ring.missing_slots or i in ring.gapset or ring.pieces[i] is None:
                gaps.append(i)
        if not gaps:
            return

        # choose the gap that blocks the train first (current or ahead)
        theta = self.app.train.theta
        i_now = int(((theta % (2*pi)) / (2*pi)) * n) % n
        ahead = (i_now + 1) % n
        target = ahead if ahead in gaps else (gaps[0])

        # donor is 2 slots behind target (wrap)
        donor = (target - 2) % n

        # donor must be present and not missing
        if donor in ring.missing_slots or donor in ring.gapset or ring.pieces[donor] is None:
            # fallback: find any present donor nearest to target
            cand = []
            for j in range(n):
                if j in ring.missing_slots: continue
                if j in ring.gapset: continue
                if ring.pieces[j] is None: continue
                d = min((target - j) % n, (j - target) % n)
                cand.append((d, j))
            if not cand:
                return
            cand.sort()
            donor = cand[0][1]

        # pick the nearest arm to the donor angle
        th_d = self._theta_of_slot(donor)
        arm = self.app.arms.nearest_by_theta(th_d)

        # pick a free mover (parked at arm.home_theta)
        mover = None
        for m in self.app.movers:
            if not m.busy() and abs(((m.home_theta - arm.angle()) + 2*pi) % (2*pi)) < 1e-6:
                mover = m; break
        if mover is None:
            return  # that arm is already busy

        # Allow filling "missing" targets
        if target in ring.missing_slots:
            ring.missing_slots.discard(target)
            ring.gapset.add(target)

        ok = mover.request_move(target_idx=target, donor_idx=donor)
        if ok:
            self.cooldown = 0.3  # small guard






class TrackMover:
    """
    Logical mover used later for pick/place orchestration.
    No rotating carrier cylinders — all visuals removed.
    """
    def __init__(self, env, ring: TrackRing, cfg: RingCfg, *,
                 start_theta: float = 0.0, name: str = "M", show_carrier=False):
        self.env = env
        self.ring = ring
        self.cfg = cfg
        self.name = name
        self.home_theta = float(start_theta)

        # basic lift/carry radii for planning math only
        self.z_lift = cfg.height_z + 0.12
        self.r_carry = cfg.radius + self.ring.out_offset + 0.10

        # remove rotating cylinder visuals entirely
        self.show_carrier = show_carrier
        if show_carrier:
            T_start = (SE3(cfg.center_x, cfg.center_y, self.z_lift)
                       * SE3.Rz(self.home_theta)
                       * SE3.Tx(self.r_carry))
            self.carrier = Cylinder(0.03, 0.06, pose=T_start, color=[0.3, 0.7, 0.9, 0.9])
            env.add(self.carrier)
        else:
            self.carrier = None

        # state
        self.job = None
        self.speed_arc = 0.9
        self.carry_theta = self.home_theta
        self.idle_speed = 0.0  # stays parked by default

    def busy(self) -> bool:
        return self.job is not None

    def request_move(self, target_idx: int, donor_idx: int = None):
        """
        Simple stub — logic kept for later pick/place.
        Does not animate anything visually.
        """
        if target_idx in self.ring.missing_slots:
            print(f"target {target_idx} is permanently missing; remove from missing_slots to fill")
            return False
        if self.busy():
            print("mover busy; queueing not implemented (one job at a time)")
            return False

        n = self.cfg.n_segments
        # pick donor if not specified
        if donor_idx is None:
            candidates = []
            for i in range(n):
                if i in self.ring.missing_slots: continue
                if i in self.ring.gapset: continue
                if self.ring.pieces[i] is None: continue
                candidates.append((min((target_idx - i) % n, (i - target_idx) % n), i))
            if not candidates:
                print("no available donor piece to move")
                return False
            candidates.sort()
            donor_idx = candidates[0][1]

        # just flag a placeholder job
        self.job = {"donor": donor_idx, "target": target_idx}
        print(f"[{self.name}] planning move {donor_idx} → {target_idx}")
        return True

    def update(self, dt: float):
        # nothing moves visually
        if self.job:
            print(f"[{self.name}] completed move {self.job['donor']} → {self.job['target']}")
            self.job = None



    def add_outer_arms(self, cfg: RingCfg):
        """Spawn ONE UR3 just outside the ring, lined up with slot 5, facing inward."""
        import roboticstoolbox as rtb
        self.arm = rtb.models.UR3()

        # place just outside the ring (small margin so blocks are reachable)
        margin = 0.10  # was out_offset+0.10 (=0.40) → too far; 0.10 works
        theta5 = 2 * np.pi * 5 / cfg.n_segments  # align base with slot 5

        self.arm.base = (
            SE3(cfg.center_x, cfg.center_y, cfg.height_z)   # same height as ring
            * SE3.Rz(theta5)                                # line up with slot 5
            * SE3.Tx(cfg.radius + margin)                   # just outside ring
            * SE3.Rz(np.pi)                                 # face inward
        )
        self.env.add(self.arm)

        # neutral ready pose
        try:
            self.arm.q = [0, -1.57, 1.57, 0, 1.57, 0]
        except Exception:
            pass


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
        self.overrides = {}              # ← NEW: i -> SE3 forced pose



    # ← NEW helpers
    def set_override(self, i: int, T: SE3):
        self.overrides[i] = SE3(T)

    def clear_override(self, i: int):
        self.overrides.pop(i, None)




    def _block_pose_for_index(self, i: int):
        # Use the base-radius pose that never changes with IN/OUT/missing
        baseT = self.ring.slot_poses[i]          # fixed world SE3 at base radius
        return baseT * SE3.Rz(self.theta_offset) \
                    * SE3.Tx(self.block_offset)  \
                    * SE3.Tz(self.lift)



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
            if i in self.overrides:      # ← NEW
                T = self.overrides[i]
            else:
                T = self._block_pose_for_index(i)
            block.T = T.A
            p = T.t
            self.world_points[i] = (float(p[0]), float(p[1]), float(p[2]))

    def get_points(self):
        return list(self.world_points)


# -------------------------------
# Panda robot in the middle (goal-seeking IK with smoothing + reach clamp)
# -------------------------------
class RobotArm:
    def __init__(self, env, cfg: RingCfg):
        self.env = env
        self.cfg = cfg

        # 1) Create the robot FIRST
        self.robot = models.Panda()
        self.robot.base = SE3(cfg.center_x, cfg.center_y, 0.0)
        env.add(self.robot)

        # --- FOLLOW POSTURE (EDITED SECTION) ---
        # Start from a neutral pose and bend down/back toward the train
        self.q_follow = self.robot.qz.copy()

        # ↓↓↓ TUNE THESE IF NEEDED ↓↓↓
        DOWN_TILT   = 0.5  # more negative = lean lower toward the track
        BACK_REACH  = -1.7  # more negative = elbow bends back (brings tool backward)
        WRIST_PITCH =  2.1  # higher = tilt wrist down toward the train

        # Panda joint order: [j0, j1, j2, j3, j4, j5, j6]
        self.q_follow[1] = DOWN_TILT        # shoulder pitch (down)
        self.q_follow[2] =  0.40            # shoulder lift
        self.q_follow[3] = BACK_REACH       # elbow back
        self.q_follow[4] =  0.00            # forearm roll
        self.q_follow[5] = WRIST_PITCH      # wrist pitch
        self.q_follow[6] =  0.70            # wrist roll (cosmetic)
        # --- END EDITED SECTION ---

        # Waist limits (joint 0)
        self._j0_min = -2.8973
        self._j0_max =  2.8973

        # Optional: define a home pose slightly above center
        self.q_home   = self.robot.qz
        self.q_target = self.q_home.copy()
        self.alpha    = 0.35
        self.ik_hz    = 20
        self._accum   = 0.0
        self._mask    = [1, 1, 1, 0.5, 0.5, 0.5]

        # Tool orientation (pointing downward)
        self._tilt = SE3.RPY([-pi/2, 0, pi], order="xyz")

        self.reach_limit  = 0.80
        self.reach_margin = 0.10
        self.goal_T = self.home_pose()

    def home_pose(self):
        # Hover above center with tilt applied
        return SE3(self.cfg.center_x, self.cfg.center_y, self.cfg.height_z + 0.45) * self._tilt

    @staticmethod
    def _wrap_pi(a):
        """wrap angle to (-pi, pi]"""
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
                    # NEW: guard for missing slots
                    if idx in self.ring.missing_slots:
                        print(f"slot {idx} is missing; toggle ignored")
                        return
                    self.app.toggle_segment(idx)  # simple, no robot
                return _cb
            btn = swift.Button(cb=make_cb(i), desc=self._label_for(i))
            self.env.add(btn)
            self.buttons.append(btn)

    def refresh_labels(self):
        for i, btn in enumerate(self.buttons):
            btn.desc = self._label_for(i)

from math import pi

def test_move_one(self, donor_idx: int, target_idx: int):
    """
    One-shot move: carry the segment at donor_idx to target_idx.
    No other moves will run unless you call this again.
    """
    # lazy-create a tiny mover fleet parked at 0°, 120°, 240° (invisible carriers)
    if not hasattr(self, "movers"):
        self.movers = []
        for j in range(3):
            start_theta = 2 * pi * j / 3
            self.movers.append(
                TrackMover(self.env, self.ring, self.ring.cfg,
                           start_theta=start_theta, name=f"M{j+1}", show_carrier=False)
            )

    # ensure target is a gap we’re allowed to fill
    self.ring.missing_slots.discard(target_idx)
    self.ring.gapset.add(target_idx)

    # pick the mover whose home angle is closest to donor
    th_d = self.ring.theta_of_slot(donor_idx)
    def angdist(a,b):
        d = (a-b) % (2*pi)
        return min(d, 2*pi-d)
    mover = min(self.movers, key=lambda m: angdist(m.home_theta, th_d))

    # fire the one job
    ok = mover.request_move(target_idx=target_idx, donor_idx=donor_idx)
    if not ok:
        print("test_move_one: request_move declined")

    # store for loop updates
    self._single_move_active = True


    btn_test = swift.Button(
    cb=lambda e=None: self.app.test_move_one(donor_idx=0, target_idx=(0+2) % self.ring.cfg.n_segments),
    desc="Test move: 0 → 2"
    )
    self.env.add(btn_test)




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

    def add_mover(self, cfg: RingCfg):
        self.mover = TrackMover(self.env, self.ring, cfg)


    def add_track_gates(self):
        self.gates = TrackGateUI(self.env, self, self.ring)
        self.gates.build()

    # Called by UI button
    def toggle_segment(self, idx: int):
        want_present = (idx in self.ring.gapset)  # OUT -> IN
        self.ring.start_slide(idx, present=want_present, dur=0.8)

    def add_movers(self, cfg: RingCfg, k: int = 3):
        self.movers = []
        for j in range(k):
            start_theta = 2 * pi * j / k
            self.movers.append(TrackMover(self.env, self.ring, cfg, start_theta=start_theta, name=f"M{j+1}"))

    def get_pick_pose(self, i: int):
        return self._block_pose_for_index(i)
    
    def add_outer_arms(self, cfg: RingCfg):
        """Spawn ONE UR3 just outside the ring, lined up with slot 5, facing inward."""
        import roboticstoolbox as rtb
        self.arm = rtb.models.UR3()

        # place just outside the ring (small margin so blocks are reachable)
        margin = 0.10  # was out_offset+0.10 (=0.40) → too far; 0.10 works
        theta5 = 2 * np.pi * 5 / cfg.n_segments  # align base with slot 5

        self.arm.base = (
            SE3(cfg.center_x, cfg.center_y, cfg.height_z)   # same height as ring
            * SE3.Rz(theta5)                                # line up with slot 5
            * SE3.Tx(cfg.radius + margin)                   # just outside ring
            * SE3.Rz(np.pi)                                 # face inward
        )
        self.env.add(self.arm)

        # neutral ready pose
        try:
            self.arm.q = [0, -1.57, 1.57, 0, 1.57, 0]
        except Exception:
            pass

    
    # ---------------- UR3 visual pick→place of ONE orange block ----------------
    def _ik_to(self, robot, T_goal: SE3, steps=80, label=""):
        """Solve IK to T_goal and follow a joint trajectory; return final q or None."""
        import roboticstoolbox as rtb
        sol = robot.ikine_LM(self._T_in_base(robot, T_goal))
        if not sol.success:
            print(f"[UR3 IK] failed at {label or 'target'}")
            return None
        q_goal = sol.q
        traj = rtb.jtraj(robot.q, q_goal, steps)
        for q in traj.q:
            robot.q = q
            # if we're carrying a block, keep updating it here
            if getattr(self, "_carry", None):
                Ttool = robot.fkine(robot.q)
                world_Ttool = self._T_out_of_base(robot, Ttool)
                block, T_tool_to_block, idx = self._carry
                T_now = world_Ttool * T_tool_to_block
                block.T = T_now.A
                # tell zones not to snap it back
                self.zones.set_override(idx, T_now)
            self.env.step(self.dt)
        return q_goal

    def _T_in_base(self, robot, T_world: SE3) -> SE3:
        """Convert world pose to robot-base frame."""
        # robot.base is world_T_base
        return SE3(robot.base).inv() * T_world

    def _T_out_of_base(self, robot, T_in_base: SE3) -> SE3:
        """Convert base-frame pose to world pose."""
        return SE3(robot.base) * T_in_base
    
    def ur3_pick_and_place_orange(self, src=5, dst=7, lift=0.20):
        """
        Single UR3 visually picks orange[src] and places at orange[dst].
        Uses tool-down orientation to make IK easy.
        """
        import roboticstoolbox as rtb

        robot = self.arm
        block = self.zones.blocks[src]

        # world positions of the orange frames (keep your frame logic)
        T_src_ref = self.zones._block_pose_for_index(src)
        T_dst_ref = self.zones._block_pose_for_index(dst)
        p_src = T_src_ref.t
        p_dst = T_dst_ref.t

        # Tool-down orientation (z-axis down). Same everywhere for simplicity.
        TOOL_DOWN = SE3.RPY([np.pi/2, 0, -np.pi], order="xyz")

        # Build target poses: tool-down at those XYZs
        T_src       = SE3(p_src) * TOOL_DOWN
        T_dst       = SE3(p_dst) * TOOL_DOWN
        T_src_hover = SE3([p_src[0], p_src[1], p_src[2] + lift]) * TOOL_DOWN
        T_dst_hover = SE3([p_dst[0], p_dst[1], p_dst[2] + lift]) * TOOL_DOWN
        T_via       = SE3(((T_src_hover.t + T_dst_hover.t) * 0.5)) * TOOL_DOWN

        # IK helper
        def _move_to(T_goal, steps=70, label=""):
            # Convert world → base frame
            T_goal_in_base = SE3(robot.base).inv() * T_goal
            sol = robot.ikine_LM(T_goal_in_base, q0=np.asarray(robot.q, dtype=float))
            if not sol.success:
                print(f"[UR3 IK] failed at {label or 'target'}")
                return False
            traj = rtb.jtraj(robot.q, sol.q, steps)
            for q in traj.q:
                robot.q = q
                if getattr(self, "_carry", False):
                    T_tool_world = SE3(robot.base) * robot.fkine(q)
                    # attach block directly to the TCP, no offset error
                    block.T = (T_tool_world * SE3.Rx(np.pi) * SE3.Tz(0.05)).A
                    # keep zones from snapping it back
                    self.zones.set_override(src, SE3(block.T))

                self.env.step(self.dt)
            return True


        # Sequence: hover→down→attach→lift→via→hover(dst)→down→release→lift
        if not _move_to(T_src_hover, label="src_hover"): return
        if not _move_to(T_src,       label="src_down"):  return
        self._carry = True
        if not _move_to(T_src_hover, label="lift"):      return
        if not _move_to(T_via,       label="via"):       return
        if not _move_to(T_dst_hover, label="dst_hover"): return
        if not _move_to(T_dst,       label="dst_down"):  return
        self._carry = False
        block.T = T_dst.A
        self.zones.set_override(src, T_dst)   # leave block visibly at destination
        _move_to(T_dst_hover, label="clear")
        print(f"UR3 moved orange block {src} → {dst}")



    def ur3_move_orange_5_to_7(self, arm_index=0, src=5, dst=7, lift=0.12):
        """
        VISUAL pick&place with a UR3:
        - hover above orange[src] -> grasp -> lift
        - mid-air via -> hover above orange[dst] -> place -> lift away
        Uses your TrackLinkedZones block poses; leaves block overridden at dst.
        """
        # pick one of your outer UR3s
        robot = self.arms.arms[arm_index].robot

        # source/destination frames (world)
        T_src = self.zones._block_pose_for_index(src)
        T_dst = self.zones._block_pose_for_index(dst)
        T_src_hover = T_src * SE3.Tz(lift)
        T_dst_hover = T_dst * SE3.Tz(lift)
        T_mid = SE3(((T_src_hover.t + T_dst_hover.t) * 0.5))  # keep orientation simple

        # block we will carry
        block = self.zones.blocks[src]

        # 0) make sure zones won't fight us during the motion
        self.zones.set_override(src, T_src)

        # 1) move tool: hover over src → descend to grasp
        if self._ik_to(robot, T_src_hover, steps=80, label="src_hover") is None: return
        if self._ik_to(robot, T_src,       steps=60, label="src_grasp") is None: return

        # measure tool→block offset ONCE at grasp so it won't teleport
        Ttool_world = self._T_out_of_base(robot, robot.fkine(robot.q))
        T_tool_to_block = Ttool_world.inv() * SE3(block.T)

        # start carrying (tuple so _ik_to can update visuals each step)
        self._carry = (block, T_tool_to_block, src)

        # 2) lift back to hover, go via mid, go above dst, then down to place
        if self._ik_to(robot, T_src_hover, steps=50, label="lift") is None: return
        if self._ik_to(robot, T_mid,       steps=70, label="via")  is None: return
        if self._ik_to(robot, T_dst_hover, steps=70, label="dst_hover") is None: return
        if self._ik_to(robot, T_dst,       steps=60, label="place") is None: return

        # 3) stop carrying; leave block at dst (override so it stays there)
        Ttool_world = self._T_out_of_base(robot, robot.fkine(robot.q))
        T_now = Ttool_world * T_tool_to_block
        block.T = T_now.A
        self.zones.set_override(src, T_now)   # keep at new spot even if it overlaps
        self._carry = None

        # 4) lift away a bit (optional)
        self._ik_to(robot, T_dst_hover, steps=50, label="clear")
        print("UR3 visual move: orange 5 → 7 done.")
    # ---------------------------------------------------------------------------




    def loop(self):
        try:
            while True:
                # train motion (with gating)
                gated_v = self.ring.allowed_speed(self.train.theta, self.train.speed)
                self.train.step(self.dt, gated_v)

                # have the robot spin its waist to follow the train
                if hasattr(self, "robot"):
                    self.robot.follow_train_yaw(self.train.theta, yaw_bias= -pi/3.5)

                # remember if something was animating
                was_anim = self.ring.is_animating()

                # update ring (animation) and blocks
                self.ring.update()
                self.zones.update()

                # --- movers block (multi + orchestrator) ---
                if hasattr(self, "movers"):
                    # orchestrator decides when to create a two-slot move
                    if hasattr(self, "orchestrator"):
                        self.orchestrator.step(self.dt)

                    # update all movers (will run jobs or stay parked)
                    for m in self.movers:
                        m.update(self.dt)
                # --- end movers block ---

                # after self.ring.update(); self.zones.update()
                if getattr(self, "_single_move_active", False) and hasattr(self, "movers"):
                    all_idle = True
                    for m in self.movers:
                        m.update(self.dt)
                        all_idle = all_idle and (not m.busy())
                    if all_idle:
                        self._single_move_active = False



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


    app.add_outer_arms(cfg)    # <- fixed UR-style arms outside
    # app.add_movers(cfg, k=3)   # <- one mover per arm angle
    # app.orchestrator = GapOrchestrator(app)

    app.add_track_gates()      # side buttons
    app.add_robot(cfg)         # Panda that spins with the train
    app.add_linked_zones(cfg)  # red blocks following segments

    btn_ur3move = swift.Button(
        cb=lambda e=None: app.ur3_pick_and_place_orange(5, 7),
        desc="UR3: orange 5 → 7"
    )
    app.env.add(btn_ur3move)



    app.loop()
