# train_demo.py — Ring + Train + Panda + Linked Track Zones + Segment Buttons

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
    radius: float = 0.8
    height_z: float = 0.10
    yaw_tweak_deg: float = 69.65
    center_x: float = 0.0
    center_y: float = 0.0
    global_yaw_deg: float = 0.0
    unit_scale: float = 0.0003
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
        self.gapset = set()        # indices of segments currently OUT (missing)
        self.out_offset = 0.3      # one place to control how far OUT pieces slide

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

    # ---- NEW: current segment pose that respects IN/OUT and allows a phase offset
    def segment_pose_with_offset(self, idx: int, theta_offset: float = 0.0):
        c = self.cfg
        n = c.n_segments
        theta = 2 * pi * idx / n + theta_offset
        r = c.radius + (self.out_offset if idx in self.gapset else 0.0)
        return (
            SE3(c.center_x, c.center_y, c.height_z)
            * SE3.Rz(c.global_yaw_deg * pi / 180)
            * SE3.Rz(theta)
            * SE3.Tx(r)
            * SE3.Rz(pi / 2 + c.yaw_tweak_deg * pi / 180)
            * SE3.Rx(-pi / 2)
        )

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
        c = self.cfg
        theta = 2 * pi * idx / self.cfg.n_segments

        if present:
            self.pieces[idx].T = self.pose_for_theta(theta).A
            self.gapset.discard(idx)
        else:
            T = (
                SE3(c.center_x, c.center_y, c.height_z)
                * SE3.Rz(c.global_yaw_deg * pi / 180)
                * SE3.Rz(theta)
                * SE3.Tx(c.radius + self.out_offset)
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

    # ---- NEW (for later): make a segment follow a given block pose
    #     Keep this here; we won't call it yet until you want "block drives track".
    def align_segment_to_block(self, idx: int, block_T: SE3, theta_offset: float = 0.0):
        """
        Infer angle & radius from a block, decide IN/OUT, then place the segment
        tangent to the ring at that angle, respecting our yaw_tweak and base height.
        """
        import math
        c = self.cfg
        px, py, pz = float(block_T.t[0]), float(block_T.t[1]), float(block_T.t[2])

        dx = px - c.center_x
        dy = py - c.center_y
        r = (dx*dx + dy*dy) ** 0.5
        theta = (math.atan2(dy, dx) - (c.global_yaw_deg * pi / 180))  # world -> ring frame

        # Decide IN/OUT based on radius (threshold halfway to out_offset)
        is_out = r > (c.radius + 0.5 * self.out_offset)
        self.set_segment_present(idx, present=not is_out)

        # Place the segment with our ring tangent frame at this angle (optionally phase-offset)
        theta += theta_offset
        base_r = c.radius + (self.out_offset if is_out else 0.0)
        Tseg = (
            SE3(c.center_x, c.center_y, c.height_z)
            * SE3.Rz(c.global_yaw_deg * pi / 180)
            * SE3.Rz(theta)
            * SE3.Tx(base_r)
            * SE3.Rz(pi / 2 + c.yaw_tweak_deg * pi / 180)
            * SE3.Rx(-pi / 2)
        )
        self.pieces[idx].T = Tseg.A


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
# Linked reach blocks (attached to tracks) — FIXED PLACEMENT
# -------------------------------
class TrackLinkedZones:
    """
    9 red blocks positioned at the midline of each track segment.
    They follow the segment state (IN/OUT) AND keep your original phase offset.
    Exposes a stable 9-point array you’ll give to the outer robots later.
    """
    def __init__(self, env, ring: "TrackRing", cfg: "RingCfg"):
        self.env = env
        self.ring = ring
        self.cfg = cfg
        self.blocks = []

        # Your remembered dimensions (unchanged)
        self.block_offset = 0.255                 # radial tweak (m)
        self.lift = -0.005                        # vertical tweak (m)
        self.theta_offset = -16.71 * pi / 180.0   # ring phase offset (rad)

        self.world_points = []

    def _block_pose_for_index(self, i: int):
        """
        Start from the segment’s *current* pose (IN/OUT radius)
        with your theta_offset applied, then add your radial+vertical tweaks.
        """
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
        n = self.cfg.n_segments
        for i, block in enumerate(self.blocks):
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
    """
    Lightweight IK: set a goal pose; each update() step nudges joints toward it.
    Provides helpers to compute a pose above an orange block and 'home' at center.
    Uses a reach clamp so goals stay solvable when the ring radius > Panda reach.
    """
    def __init__(self, env, cfg: RingCfg):
        self.env = env
        self.cfg = cfg

        self.robot = models.Panda()
        self.robot.base = SE3(cfg.center_x, cfg.center_y, 0.0)
        env.add(self.robot)

        # pedestal (visual only)
        env.add(Cylinder(radius=0.12, length=0.25,
                         pose=SE3(cfg.center_x, cfg.center_y, 0.125),
                         color=[0.2, 0.2, 0.22, 1.0]))

        # IK state
        self.q_home = self.robot.qz
        self.q_target = self.q_home.copy()
        self.alpha = 0.35   # responsive; adjust 0.25–0.45 to taste
        self.ik_hz = 20     # enough at dt=0.05; use 30 if you later speed up the loop
        self._accum = 0.0
        self._mask = [1, 1, 1, 0.5, 0.5, 0.5]   # relax orientation slightly


        # tool tilt: gripper facing DOWN toward the track (flipped 180° about Z)
        self._tilt = SE3.RPY([-pi/2, 0, pi], order="xyz")



        # ---- NEW: soft reach limit (meters) so goals are solvable
        # Panda effective comfortable reach ~0.85 m; keep a little margin
        self.reach_limit = 0.80
        self.reach_margin = 0.10

        # current goal pose (SE3)
        self.goal_T = self.home_pose()

    # --- goal helpers ---
    def home_pose(self):
        # hover above center
        return SE3(self.cfg.center_x, self.cfg.center_y, self.cfg.height_z + 0.45) * self._tilt

    def _clamp_xy_to_reach(self, px: float, py: float):
        """
        Keep goal within a disk of radius (reach_limit - margin) around the base.
        Returns (x, y) possibly contracted along the ray from center to (px, py).
        """
        import math
        cx, cy = self.cfg.center_x, self.cfg.center_y
        dx, dy = (px - cx), (py - cy)
        r = math.hypot(dx, dy)
        rmax = max(0.05, self.reach_limit - self.reach_margin)
        if r <= rmax:
            return px, py
        ux, uy = dx / r, dy / r
        return (cx + ux * rmax, cy + uy * rmax)

    def pose_above_block(self, block_T: SE3, hover: float = 0.25, down: float = 0.05, mode: str = "hover"):
        """
        Compose a goal from a block's transform.
        mode="hover": stand above the block by `hover` in world Z
        mode="down":  stand close to the block by `down` in world Z
        NOTE: XY is clamped to reach; Z is clamped to stay above the table.
        """
        dz = hover if mode == "hover" else down
        bx, by, bz = float(block_T.t[0]), float(block_T.t[1]), float(block_T.t[2])
        gx, gy = self._clamp_xy_to_reach(bx, by)
        gz = self._safe_z(bz + dz, pad=0.03)  # never dip below ring plane
        return SE3(gx, gy, gz) * self._tilt


    # --- control API ---
    def set_goal(self, T_goal: SE3):
        self.goal_T = T_goal

    def at_goal(self, pos_tol=0.01, rot_tol_deg=7.0):
        """Rough proximity check: ~1 cm and ~7°."""
        import numpy as np, math
        Tcur = SE3(self.robot.fkine(self.robot.q).A)
        p_err = np.linalg.norm(Tcur.t - self.goal_T.t)
        Rerr = (Tcur.R.T @ self.goal_T.R)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rerr) - 1) / 2))))
        return (p_err <= pos_tol) and (ang <= rot_tol_deg)

    def update(self, dt):
        """Solve IK toward goal at ~ik_hz, then smooth joints."""
        self._accum += dt
        if self._accum >= 1.0 / self.ik_hz:
            self._accum = 0.0
            sol = self.robot.ikine_LM(self.goal_T, q0=self.robot.q, mask=self._mask)
            if sol.success:
                self.q_target = sol.q
            else:
                # gentle drift to home if solver fails
                self.q_target = 0.98 * self.q_target + 0.02 * self.q_home
        # smooth toward target
        self.robot.q = (1 - self.alpha) * self.robot.q + self.alpha * self.q_target

    def _safe_z(self, z_world: float, pad: float = 0.03) -> float:
        """
        Prevent goals from dipping below the tabletop/track plane.
        We keep at least `pad` above ring height_z.
        """
        z_min = self.cfg.height_z + pad
        return max(z_world, z_min)


# -------------------------------
# Segment gate UI (ON/OFF buttons, enqueue robot action)
# -------------------------------
class TrackGateUI:
    """
    One button per segment: enqueues an action for the app to:
      hover -> down -> toggle the segment -> back to home.
    Labels show 'ON' when the segment is IN, 'OFF' when OUT.
    """
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
                def _cb(event=None):   # Swift passes one arg
                    # enqueue: robot visit + toggle
                    self.app.enqueue_toggle(idx)
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
        self.actions = []          # queued jobs like ('toggle', idx)
        self.task = None           # current task state machine dict, or None


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
        self.gates = TrackGateUI(self.env, self, self.ring)  # pass app=self
        self.gates.build()


    def loop(self):
        try:
            while True:
                # stop train if current/next segment is OUT
                gated_v = self.ring.allowed_speed(self.train.theta, self.train.speed)
                self.train.step(self.dt, gated_v)

                # NEW 1) Robot IK toward its current goal
                if hasattr(self, "robot"):
                    self.robot.update(self.dt)

                # keep orange blocks riding with their OWN segment
                self.zones.update()

                # NEW 2) Start next queued action if idle                
                if (self.task is None) and self.actions:
                    kind, idx, want_present = self.actions.pop(0)
                    if kind == 'toggle':
                        self._start_toggle_task(idx, want_present)


                # NEW 3) Advance task phases (hover -> down -> toggle -> up -> home)
                self._advance_task()

                # Render
                self.env.step(self.dt)
        except KeyboardInterrupt:
            print("Exiting simulation...")


    def enqueue_toggle(self, idx: int, preempt=True):
        """
        Queue 'visit block & set segment to desired state'.
        We compute the desired state now (so it always toggles),
        and optionally preempt the current task so the new click
        starts right away.
        """
        want_present = (idx in self.ring.gapset)  # if currently OUT, set present=True (IN)
        job = ('toggle', idx, want_present)

        if preempt and self.task is not None:
            print(f"[preempt] stopping current task for seg {idx}")
            self.task = None
            self.actions.clear()

        # start immediately if idle, else queue
        if self.task is None:
            self._start_toggle_task(idx, want_present)
        else:
            self.actions.append(job)


    def _start_toggle_task(self, idx: int, want_present: bool):
        """Begin the 4-phase routine for this segment (hover -> down -> set -> up -> home)."""
        block_T = SE3(self.zones.blocks[idx].T)   # live frame of the orange block
        hover_T = self.robot.pose_above_block(block_T, hover=0.25, mode="hover")
        self.robot.set_goal(hover_T)
        self.task = {
            'type': 'toggle',
            'idx': idx,
            'want_present': want_present,
            'phase': 'go_hover'
        }
        print(f"[start] seg {idx} -> {'IN' if want_present else 'OUT'}")


    def _advance_task(self):
        """Advance the state machine when the robot reaches sub-goals."""
        if self.task is None:
            return
        t = self.task
        idx = t['idx']
        block_T = SE3(self.zones.blocks[idx].T)

        if t['phase'] == 'go_hover' and self.robot.at_goal():
            self.robot.set_goal(self.robot.pose_above_block(block_T, down=0.05, mode="down"))
            t['phase'] = 'go_down'

        elif t['phase'] == 'go_down' and self.robot.at_goal():
            # apply the desired state NOW (explicit, no ambiguity)
            self.ring.set_segment_present(idx, present=t['want_present'])
            print(f"[toggle] seg {idx} -> {'IN' if t['want_present'] else 'OUT'}; gapset={sorted(self.ring.gapset)}")
            if hasattr(self, 'gates'):
                self.gates.refresh_labels()
            # go back up
            self.robot.set_goal(self.robot.pose_above_block(block_T, hover=0.25, mode="hover"))
            t['phase'] = 'go_up'

        elif t['phase'] == 'go_up' and self.robot.at_goal():
            self.robot.set_goal(self.robot.home_pose())
            t['phase'] = 'go_home'

        elif t['phase'] == 'go_home' and self.robot.at_goal():
            # finished; immediately start next queued action if any
            self.task = None
            if self.actions:
                kind, idx2, want_present2 = self.actions.pop(0)
                if kind == 'toggle':
                    self._start_toggle_task(idx2, want_present2)


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

    app.add_track_gates()      # ON/OFF buttons
    app.add_robot(cfg)         # Panda at center (static for now)
    app.add_linked_zones(cfg)  # red reference blocks (follow segment state)

    app.loop()
