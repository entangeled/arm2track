# ring_train_demo.py
import numpy as np
from collections import deque
from spatialmath import SE3
import roboticstoolbox as rtb
import swift                 # ✅ use the separate swift module
from math import pi

# ------------------
# Params
# ------------------
R = 1.20                 # ring radius (m)
SEG_COUNT = 9
FILLED = 7               # 2 gaps
SEG_ARC = 2 * np.pi / SEG_COUNT
SEG_LEN = 2 * R * np.sin(SEG_ARC / 2)   # chord length approximation
SEG_W   = 0.10           # segment width (m)
SEG_H   = 0.05           # height (m)
TRAIN_LEN = 0.30

omega = 0.30             # train angular speed (rad/s)
dt = 0.05                # sim step (s)

# ------------------
# Swift environment
# ------------------
env = swift.Swift()
env.launch(realtime=True)




# ------------------
# Simple geometry helpers (swap to STL later)
# ------------------
def make_track_geom():
    # start with a thin box to represent a track segment
    return rtb.tools._common.SwiftShape.cuboid([SEG_LEN, SEG_W, SEG_H], color=[0.6,0.6,0.6,1])

def make_train_geom():
    return rtb.tools._common.SwiftShape.cuboid([TRAIN_LEN, 0.12, 0.12], color=[0.9,0.2,0.2,1])

# ------------------
# Poses on ring
# ------------------
def slot_pose(theta):
    # place chord tangent to circle at angle theta (center at chord midpoint)
    # chord midpoint location:
    x = R*np.cos(theta)
    y = R*np.sin(theta)
    # the chord’s long axis is tangent; rotate by theta + 90deg
    return SE3(x, y, SEG_H/2) * SE3.Rz(theta + np.pi/2)

def on_ring(theta, radius):
    return SE3(radius*np.cos(theta), radius*np.sin(theta), SEG_H/2)

# ------------------
# Robots (UR5s) and backend
# ------------------
def make_ur5_at(angle, r_robot=R+0.5):
    # place base at a circle outside the ring
    base = SE3(r_robot*np.cos(angle), r_robot*np.sin(angle), 0) * SE3.Rz(angle + np.pi)
    ur = rtb.models.UR5()      # uses DH model, good enough for demo
    ur.base = base.A
    return ur

# ------------------
# Scene setup
# ------------------
env = Swift()
env.launch(realtime=True)

# Centre pusher robot
ur_center = rtb.models.UR5()
ur_center.base = SE3(0,0,0).A
env.add(ur_center)

# Three outer robots
outer_angles = [0, 2*np.pi/3, 4*np.pi/3]
outer = [make_ur5_at(a) for a in outer_angles]
for r in outer:
    env.add(r)

# Train
train = make_train_geom()
env.add(train)

# Track segments in ring
slot_thetas = [(2*np.pi*i/SEG_COUNT) for i in range(SEG_COUNT)]
# choose 7 filled starting from i=0; the two gaps lead the train
filled_idx = deque(list(range(FILLED)))    # indices into slot_thetas that are filled (cyclic queue)
gaps = set(range(SEG_COUNT)) - set(filled_idx)

segments = {}
for i in filled_idx:
    geom = make_track_geom()
    env.add(geom)
    geom.T = slot_pose(slot_thetas[i]).A
    segments[i] = geom

# Train initial state: just behind the first gap
theta_train = slot_thetas[max(filled_idx)] - SEG_ARC/2
train.T = (on_ring(theta_train, R - 0.05) * SE3.Rz(theta_train + np.pi/2)).A

# ------------------
# Utility: nearest outer robot to a slot index
# ------------------
def nearest_robot(slot_idx):
    slot_theta = slot_thetas[slot_idx]
    diffs = [abs(np.arctan2(np.sin(a-slot_theta), np.cos(a-slot_theta))) for a in outer_angles]
    return int(np.argmin(diffs))

# ------------------
# Outer robot pick-place (very simplified: teleport segment waypoints + IK)
# ------------------
def pick_place(robot, seg_idx, dest_idx):
    seg = segments[seg_idx]
    # pregrasp above current segment
    T_pick = slot_pose(slot_thetas[seg_idx]) * SE3.Trans(0, 0, 0.15)
    T_lift = slot_pose(slot_thetas[seg_idx]) * SE3.Trans(0, 0, 0.30)
    T_drop = slot_pose(slot_thetas[dest_idx]) * SE3.Trans(0, 0, 0.15)
    T_place= slot_pose(slot_thetas[dest_idx])

    def solve_and_move(T):
        sol = rtb.tools.ik.IK_NR(robot, T.A, q0=robot.q)
        if sol.success:
            traj = rtb.jtraj(robot.q, sol.q, 25)
            for q in traj.q:
                robot.q = q
                env.step(dt/5)
        else:
            # if IK fails, just skip (for demo robustness)
            pass

    for T in [T_pick, T_lift, T_drop, T_place]:
        solve_and_move(T)

    # move the geometry instantly (attach/detach not modeled here)
    seg.T = slot_pose(slot_thetas[dest_idx]).A
    # bookkeeping
    del segments[seg_idx]
    segments[dest_idx] = seg

# ------------------
# Main loop
# ------------------
t = 0.0
last_passed = -1
while env.isopen():
    t += dt
    # advance train
    theta_train = (theta_train + omega*dt) % (2*np.pi)
    train.T = (on_ring(theta_train, R - 0.05) * SE3.Rz(theta_train + np.pi/2)).A

    # centre robot "push" TCP follows just inside ring
    T_push = on_ring(theta_train, R - 0.10) * SE3.Rz(theta_train + np.pi/2)
    # small IK step to follow
    solc = rtb.tools.ik.IK_NR(ur_center, T_push.A, q0=ur_center.q)
    if solc.success:
        ur_center.q = solc.q

    # detect when train exits a segment: pick trailing slot index
    # find slot whose center is just behind train angle
    idx = int(np.floor(((theta_train + SEG_ARC/2) % (2*np.pi)) / SEG_ARC)) % SEG_COUNT
    if idx != last_passed:
        last_passed = idx
        # if this slot is filled, mark it "spent" and schedule move to the leading gap
        if idx in segments:
            # find the gap ahead of the train (furthest along train direction)
            if len(gaps) == 0:
                # should never happen (we maintain 2 gaps)
                pass
            else:
                # choose the gap with smallest positive angular distance from train head
                def ahead_dist(gi):
                    d = (slot_thetas[gi] - (theta_train + SEG_ARC)) % (2*np.pi)
                    return d
                dest = min(gaps, key=ahead_dist)

                # assign nearest robot
                r_id = nearest_robot(idx)
                pick_place(outer[r_id], idx, dest)

                # update filled/gap sets
                gaps.remove(dest)
                gaps.add(idx)

    env.step(dt)
