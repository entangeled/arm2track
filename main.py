# main.py 
# 
# Contains:
#   - Segment and TrackRing class to create the tracks 
#   - Train class that moves along track until
#   - MiddleRobot class that will follow/push the Train
#   - Simulation class that runs everything together in one Swift
# 
from math import pi
from pathlib import Path
from dataclasses import dataclass

from spatialmath import SE3
from spatialgeometry import Mesh, Cuboid, Cylinder
from roboticstoolbox.backends import swift
from roboticstoolbox import models, DHRobot  # Panda model
from KUKAiiwa import KUKAiiwa
import numpy as np

from enum import Enum


# -------------------------------
# Config
# -------------------------------
'''''
Enum used to force a variable to define which robot's turn it is to adhere to the train
Uses: - When it is a certain robots turn, the robot will (from his 3 track pieces) 
        be ready to move them to support the train
      - When robot is not in use, he will place his tracks in a certain position ready for the arrival of train
      - When robot is not in use (not their turn), they will return to home static position
      - When its NO ROBOTS TURN, all robots will be standing by and DO NOTHING (E-STOP INTEGRATION)
'''''
class Turn(Enum):
    ROBOT1 = 1
    ROBOT2 = 2
    ROBOT3 = 3
    NONE = 0

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

# ============================================================================
#  SEGMENT--> Class that creates segments at a pose, and assigns them a robot
# ============================================================================
class Segment:
    def __init__(self, segment_path, position: SE3, origin, robot: DHRobot, cfg: RingCfg):
        self.cfg = cfg
        s = self.cfg.unit_scale
        self.scale = [s, s, s]
        self.mesh = Mesh(str(segment_path), pose=SE3(), scale=self.scale, color=self.cfg.color)

        self.position = position
        self.origin = origin
        self.grabbed = False
        self.robot = robot

    def update_pose(self, robot: DHRobot):
        self.position = robot.get_ee_pose()

    def update_assigned_robot(self, robot: DHRobot):
        self.robot = robot
    
    def update_mesh(self, pose):
        self.mesh.T = pose


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
        self.out_offset = 0.6      # one place to control how far OUT pieces slide

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

    def build(self):
        for _ in range(self.cfg.n_segments):
            m = Mesh(str(self.segment_path), pose=SE3(), scale=self.scale, color=self.cfg.color)
            self.env.add(m)
            self.pieces.append(m)
        self.update()

    def update(self):
        c = self.cfg
        for i, p in enumerate(self.pieces):
            p.T = self.lay_flat_tangent_pose(
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

# ============================================================================
#  TRAIN --> Class creates a moving train on the track
# ============================================================================
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


# ============================================================================
#  SIMULATION --> Class that creates, runs and loops movements
# ============================================================================
class Simulation:

    # Create swift and begin simulation
    def __init__(self):
        self.env = swift.Swift()
        self.env.launch(realtime=True, port=0)
        self.dt = 0.05
        self.cfg = RingCfg()

        segment_direction = Path(__file__).resolve().parent
        seg_path = segment_direction / "track_segment.STL"
        train_path = segment_direction / "train.STL"

        self._add_floor()
        self._place_robots()
        self._add_ring(seg_path)
        self._add_train(train_path)
        print("--> Ctrl + C to exit simulation")


    def _place_robots(self):
        self.iiwa1 = KUKAiiwa()
        self.iiwa2 = KUKAiiwa()
        self.iiwa3 = KUKAiiwa()

        # Positions are exaclty at 120 degrees from each other at 1m away from origin (each have 1/3 sector)
        position1 = SE3(0, -1, 0) 
        position2 = SE3(np.sqrt(3)/2, 1/2, 0)
        position3 = SE3(-np.sqrt(3)/2, 1/2, 0)
        self.robot_positions = [position1, position2, position3]
        self.robots = [self.iiwa1, self.iiwa2, self.iiwa3]

        for i, robot in enumerate(self.robots):
            robot.base = self.robot_positions[i]
            robot.add_to_env(self.env)

        self.env.step(self.dt)
        

    def _add_floor(self, size=(8, 8, 0.1), z=-0.01):
        self.env.add(Cuboid(scale=list(size), pose=SE3(0, 0, z), color=[0.9, 0.9, 0.95, 1]))

    def _add_ring(self, seg_path):
        self.ring = TrackRing(self.env, seg_path, self.cfg)
        self.ring.build()
        self.env.step(self.dt)

    def _add_train(self, train_path):
        self.train = Train(self.env, train_path, self.cfg)


    def loop(self):
        try:
            while True:
                # stop train if current/next segment is OUT
                gated_v = self.ring.allowed_speed(self.train.theta, self.train.speed)
                self.train.step(self.dt, gated_v)

                # keep orange blocks riding with their OWN segment

                # NEW 2) Start next queued action if idle                

                # NEW 3) Advance task phases (hover -> down -> toggle -> up -> home)

                self.env.step(self.dt)
        except KeyboardInterrupt:
            print("Exiting simulation...")



# ============================================================================
#  MAIN --> Starts the simulation
# ============================================================================
if __name__ == "__main__":

    sim = Simulation()
    sim.loop()
