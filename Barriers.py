#safety barrier control system
#code for 3 forms of safety barriers


from moveit_commander import RobotCommander, MoveGroupCommander
import math 
import time
import numpy as np
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from sensor_msgs.msg import JointStat

# safety.py
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional

# ---------------------------
# Core Safety Manager
# ---------------------------

@dataclass
class SafetyManager:
    """
    Coordinates safety states across multiple barriers.
    Callbacks are supplied by your motion/IO layer.
    """
    on_stop_immediate: Callable[[], None]           # must halt robot/tracks NOW (non-blocking if possible)
    on_motion_pause: Optional[Callable[[], None]]   # softer pause hook (optional)
    on_ready_to_resume: Callable[[], None]          # arm the system (enables motion AFTER operator confirms)
    on_resume_motion: Callable[[], None]            # actually resume motion after operator confirmation

    # Internal state flags
    _estop_latched: bool = field(default=False, init=False)
    _resume_armed: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def trip_estop(self, reason: str = "E-STOP"):
        """Hard stop and latch the system; requires 2 actions to run again."""
        with self._lock:
            if not self._estop_latched:
                print(f"[SAFETY] TRIP: {reason}")
                self._estop_latched = True
                self._resume_armed = False
                try:
                    self.on_stop_immediate()
                except Exception as e:
                    print(f"[SAFETY] on_stop_immediate error: {e}")

    def reset_estop(self):
        """
        Operator action #1: reset the estop latch, but DO NOT move yet.
        System becomes 'armed to resume' only.
        """
        with self._lock:
            if self._estop_latched:
                print("[SAFETY] RESET: Latch cleared. System requires explicit resume to move.")
                self._estop_latched = False
                self._resume_armed = True
                try:
                    self.on_ready_to_resume()
                except Exception as e:
                    print(f"[SAFETY] on_ready_to_resume error: {e}")

    def resume_if_safe(self):
        """
        Operator action #2: only allowed if latched is clear AND armed.
        """
        with self._lock:
            if not self._estop_latched and self._resume_armed:
                print("[SAFETY] RESUME: Motion re-enabled.")
                self._resume_armed = False
                try:
                    self.on_resume_motion()
                except Exception as e:
                    print(f"[SAFETY] on_resume_motion error: {e}")
            else:
                print("[SAFETY] RESUME blocked: still latched or not armed.")

    def is_stopped(self) -> bool:
        with self._lock:
            return self._estop_latched

# ---------------------------
# Barrier Watcher Base
# ---------------------------

class BarrierWatcher:
    def __init__(self, name: str, manager: SafetyManager, poll_hz: float = 50.0):
        self.name = name
        self.manager = manager
        self.poll_dt = 1.0 / max(1e-6, poll_hz)
        self._running = False
        self._th = None

    def start(self):
        if self._running: 
            return
        self._running = True
        self._th = threading.Thread(target=self._loop, name=f"{self.name}-watcher", daemon=True)
        self._th.start()
        print(f"[SAFETY] {self.name}: watcher started.")

    def stop(self):
        self._running = False
        if self._th:
            self._th.join(timeout=1.0)
            self._th = None
        print(f"[SAFETY] {self.name}: watcher stopped.")

    def _loop(self):
        try:
            while self._running:
                if self.check_violation():
                    self.manager.trip_estop(reason=self.name)
                time.sleep(self.poll_dt)
        except Exception as e:
            print(f"[SAFETY] {self.name}: watcher error: {e}")

    def check_violation(self) -> bool:
        raise NotImplementedError

# ---------------------------
# 1) Emergency Stop Button
# ---------------------------

class EmergencyStopButton(BarrierWatcher):
    """
    Hook this to your physical button (Arduino/ESP) or GUI flag.
    polling_fn() must return True when the button is PRESSED.
    """
    def __init__(self, manager: SafetyManager, polling_fn: Callable[[], bool], poll_hz: float = 100.0):
        super().__init__("Emergency Stop Button", manager, poll_hz=poll_hz)
        self.polling_fn = polling_fn

    def check_violation(self) -> bool:
        try:
            return bool(self.polling_fn())
        except Exception as e:
            print(f"[SAFETY] E-STOP polling error: {e}")
            return False

# ---------------------------
# 2) Workspace Sensing (Unsafe Zones)
# ---------------------------

# Axis-aligned bounding box: ((minx,miny,minz), (maxx,maxy,maxz))
AABB = Tuple[Tuple[float, float, float], Tuple[float, float, float]]

class WorkspaceSensing(BarrierWatcher):
    """
    Light curtain / unsafe zone.
    get_tcp() returns current tool pose (x,y,z) in same frame as boxes.
    """
    def __init__(self, manager: SafetyManager, get_tcp: Callable[[], Tuple[float,float,float]],
                 unsafe_boxes: List[AABB], margin: float = 0.0, poll_hz: float = 60.0):
        super().__init__("Workspace Sensing", manager, poll_hz=poll_hz)
        self.get_tcp = get_tcp
        self.unsafe_boxes = unsafe_boxes
        self.margin = margin

    def _in_box(self, p, box: AABB) -> bool:
        (mnx,mny,mnz), (mxx,mxy,mxz) = box
        return (mnx - self.margin) <= p[0] <= (mxx + self.margin) and \
               (mny - self.margin) <= p[1] <= (mxy + self.margin) and \
               (mnz - self.margin) <= p[2] <= (mxz + self.margin)

    def check_violation(self) -> bool:
        try:
            p = self.get_tcp()
            for box in self.unsafe_boxes:
                if self._in_box(p, box):
                    return True
            return False
        except Exception as e:
            print(f"[SAFETY] Workspace polling error: {e}")
            return False

# ---------------------------
# 3) Collision Detection (distance-to-obstacle)
# ---------------------------

class CollisionDetection(BarrierWatcher):
    """
    Provide min_distance() -> float (meters) to closest obstacle (or predicted min).
    Triggers when distance < threshold.
    """
    def __init__(self, manager: SafetyManager, min_distance: Callable[[], float],
                 threshold_m: float = 0.05, poll_hz: float = 100.0):
        super().__init__("Collision Detection", manager, poll_hz=poll_hz)
        self.min_distance = min_distance
        self.threshold_m = threshold_m

    def check_violation(self) -> bool:
        try:
            d = float(self.min_distance())
            return d < self.threshold_m
        except Exception as e:
            print(f"[SAFETY] Collision polling error: {e}")
            return False

# ---------------------------
# 4) End-Effector (force/torque/pressure limit)
# ---------------------------

class EndEffectorGuard(BarrierWatcher):
    """
    Provide read_force() -> float (N) OR read_torque() -> float (Nm).
    Triggers when reading > limit.
    """
    def __init__(self, manager: SafetyManager, reader_fn: Callable[[], float],
                 limit: float, label: str = "force (N)", poll_hz: float = 250.0):
        super().__init__("End-Effector", manager, poll_hz=poll_hz)
        self.reader_fn = reader_fn
        self.limit = float(limit)
        self.label = label

    def check_violation(self) -> bool:
        try:
            v = float(self.reader_fn())
            if v > self.limit:
                print(f"[SAFETY] End-Effector limit exceeded: {v:.2f} > {self.limit:.2f} ({self.label})")
                return True
            return False
        except Exception as e:
            print(f"[SAFETY] End-Effector polling error: {e}")
            return False

# ---------------------------
# Factory function (matches your stub)
# ---------------------------

def safety_barrier_control_system(barrier_type: str, **kwargs) -> BarrierWatcher:
    """
    Create and return a configured BarrierWatcher.
    Required kwargs depend on barrier_type (see below).
    """
    manager: SafetyManager = kwargs.get("manager")
    if manager is None:
        raise ValueError("manager=SafetyManager(...) is required")

    if barrier_type == "Emergency Stop Button":
        # kwargs: polling_fn
        polling_fn = kwargs.get("polling_fn")
        if not callable(polling_fn):
            raise ValueError("polling_fn callable is required for Emergency Stop Button")
        return EmergencyStopButton(manager, polling_fn)

    elif barrier_type == "Workspace Sensing":
        # kwargs: get_tcp, unsafe_boxes[, margin, poll_hz]
        get_tcp = kwargs.get("get_tcp")
        unsafe_boxes = kwargs.get("unsafe_boxes", [])
        margin = kwargs.get("margin", 0.0)
        poll_hz = kwargs.get("poll_hz", 60.0)
        if not callable(get_tcp) or not unsafe_boxes:
            raise ValueError("get_tcp callable and non-empty unsafe_boxes are required for Workspace Sensing")
        return WorkspaceSensing(manager, get_tcp, unsafe_boxes, margin=margin, poll_hz=poll_hz)

    elif barrier_type == "Collision Detection":
        # kwargs: min_distance[, threshold_m, poll_hz]
        min_distance = kwargs.get("min_distance")
        threshold_m = kwargs.get("threshold_m", 0.05)
        poll_hz = kwargs.get("poll_hz", 100.0)
        if not callable(min_distance):
            raise ValueError("min_distance callable is required for Collision Detection")
        return CollisionDetection(manager, min_distance, threshold_m=threshold_m, poll_hz=poll_hz)

    elif barrier_type == "end-effector":
        # kwargs: reader_fn, limit[, label, poll_hz]
        reader_fn = kwargs.get("reader_fn")
        limit = kwargs.get("limit")
        label = kwargs.get("label", "force (N)")
        poll_hz = kwargs.get("poll_hz", 250.0)
        if not callable(reader_fn) or limit is None:
            raise ValueError("reader_fn callable and limit are required for end-effector")
        return EndEffectorGuard(manager, reader_fn, limit=limit, label=label, poll_hz=poll_hz)

    else:
        raise ValueError("Invalid barrier type")
