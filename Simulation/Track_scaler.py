# train_demo.py — 9-piece circular track with sliders:
# radius, height, per-piece yaw tweak, center X/Y, global yaw

from math import pi
import time
from pathlib import Path

from spatialmath import SE3
from spatialgeometry import Mesh, Cuboid
from roboticstoolbox.backends import swift


# ---------------------------- helpers ----------------------------

def launch_swift(port: int = 0):
    env = swift.Swift()
    env.launch(realtime=True, port=port)
    return env

def add_floor(env, size=(4.0, 4.0, 0.02), z=-0.01, color=(0.9, 0.9, 0.95, 1)):
    floor = Cuboid(list(size), pose=SE3(0, 0, z), color=list(color))
    env.add(floor)
    return floor

def ring_piece_pose(i, n_segments, center_xy, radius, z, yaw_tweak_deg, ring_yaw_deg):
    """
    Pose for segment i on a ring.
    Order: center → global yaw → Rz(theta) → Tx(radius) → yaw tangent → lay flat
    """
    cx, cy = center_xy
    dtheta = 2 * pi / n_segments
    theta  = i * dtheta
    yawfix = yaw_tweak_deg * pi / 180.0
    gyaw   = ring_yaw_deg  * pi / 180.0

    return (
        SE3(cx, cy, z)      # move ring center in world
        * SE3.Rz(gyaw)      # global yaw of the whole ring
        * SE3.Rz(theta)     # segment index around ring
        * SE3.Tx(radius)    # out to radius
        * SE3.Rz(pi/2 + yawfix)  # face tangent (+ tweak)
        * SE3.Rx(-pi/2)     # lay STL flat (Z-up)
    )


# ------------------------------ main -----------------------------

def main():
    here = Path(__file__).resolve().parent
    seg_path = here / "track_segment.STL"

    print("Script dir:", here)
    print("Segment exists?", seg_path.exists(), seg_path)

    env = launch_swift(port=0)
    add_floor(env)

    # constants
    N_SEGMENTS = 9
    UNIT_SCALE = 0.0005
    SCALE_VEC  = [UNIT_SCALE, UNIT_SCALE, UNIT_SCALE]
    COLOR      = [0.35, 0.35, 0.38, 1]

    # sliders
    env.add(swift.Label("— Track Ring Controls —"))
    radius_slider = swift.Slider(cb=lambda v: None, min=0.10, max=10.50, step=0.01, value=0.60, desc="Radius (m)")
    height_slider = swift.Slider(cb=lambda v: None, min=0.00, max=0.50, step=0.01, value=0.20, desc="Height Z (m)")
    yaw_slider    = swift.Slider(cb=lambda v: None, min=-5.0, max=1000.0, step=0.05, value=0.00, desc="Per-piece yaw tweak (°)")
    cx_slider     = swift.Slider(cb=lambda v: None, min=-1.50, max=1.50, step=0.01, value=0.00, desc="Center X (m)")
    cy_slider     = swift.Slider(cb=lambda v: None, min=-1.50, max=1.50, step=0.01, value=0.00, desc="Center Y (m)")
    ringyaw_slider= swift.Slider(cb=lambda v: None, min=-180, max=180, step=1.0, value=0.0, desc="Global yaw (°)")
    for s in (radius_slider, height_slider, yaw_slider, cx_slider, cy_slider, ringyaw_slider):
        env.add(s)

    # create the 9 pieces once
    pieces = []
    for _ in range(N_SEGMENTS):
        m = Mesh(str(seg_path), pose=SE3(), scale=SCALE_VEC, color=COLOR)
        env.add(m)
        pieces.append(m)

    # live update
    try:
        while True:
            R  = radius_slider.value
            Z  = height_slider.value
            YT = yaw_slider.value
            CX = cx_slider.value
            CY = cy_slider.value
            GY = ringyaw_slider.value

            for i, p in enumerate(pieces):
                pose = ring_piece_pose(i, N_SEGMENTS, (CX, CY), R, Z, YT, GY)
                p.T = pose.A

            env.step()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Exiting.")


# --------------------------- entrypoint --------------------------

if __name__ == "__main__":
    main()
