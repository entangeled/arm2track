import time
from math import pi
import numpy as np
import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support import CylindricalDHRobotPlot

# import the barbie robot from the other file
from barbie import barbie

CYL_RADIUS = 0.01
CYL_COLOR  = "#3478f6"

def run_with_sliders(bot: barbie):
    """
    Run a Swift simulation with sliders to control the robot's joints.
    """
    env = swift.Swift()
    env.launch(realtime=True)

    # Set the robot's base pose
    bot.base = SE3(0, 0, 0)

    # Add the robot to the environment
    bot.add_to_env(env)

    # Create a DHRobot for the cylindrical plot overlay
    robot_for_cyl = rtb.DHRobot(bot.links, name="BarbieCyl")
    
    # ---- cylinder overlay (draws DH geometry) ----
    cyl = CylindricalDHRobotPlot(robot_for_cyl, cylinder_radius=CYL_RADIUS, color=CYL_COLOR)
    robot_cyl = cyl.create_cylinders()
    robot_cyl.q = bot.q[:]
    env.add(robot_cyl)

    # ---- EE readout ----
    def ee_xyz_rpy():
        T = bot.fkine(bot.q).A
        xyz = np.round(T[:3, 3], 3)
        rpy = np.round(spb.tr2rpy(T, unit="deg"), 2)
        return xyz, rpy

    xyz, rpy = ee_xyz_rpy()
    labels = [swift.Label(f"X: {xyz[0]}"), swift.Label(f"Y: {xyz[1]}"), swift.Label(f"Z: {xyz[2]}"),
              swift.Label(f"Roll (φ): {rpy[0]}"), swift.Label(f"Pitch (θ): {rpy[1]}"), swift.Label(f"Yaw (ψ): {rpy[2]}")]
    for w in labels: env.add(w)

    # ---- Slider callback ----
    def on_slider(val_deg, j):
        bot.q[j] = val_deg * pi / 180.0

        # By re-assigning bot.q, we trigger the property setter, which updates the visualization
        bot.q = bot.q
        robot_cyl.q = bot.q[:]  # cylinders follow DH
        
        # Update EE readout
        xyz2, rpy2 = ee_xyz_rpy()
        labels[0].desc, labels[1].desc, labels[2].desc = f"X: {xyz2[0]}", f"Y: {xyz2[1]}", f"Z: {xyz2[2]}"
        labels[3].desc, labels[4].desc, labels[5].desc = f"Roll (φ): {rpy2[0]}", f"Pitch (θ): {rpy2[1]}", f"Yaw (ψ): {rpy2[2]}"

    # Create sliders for each joint
    sliders = []
    for i in range(bot.n):
        min_val, max_val = bot.qlim[:, i]
        s = swift.Slider(
            cb=lambda v, j=i: on_slider(v, j),
            min=np.rad2deg(min_val),
            max=np.rad2deg(max_val),
            step=1,
            value=np.rad2deg(bot.q[i]),
            desc=f"Joint {i+1}",
            unit="°"
        )
        env.add(s)
        sliders.append(s)

    # Main simulation loop
    try:
        while True:
            env.step()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Simulation stopped.")

if __name__ == "__main__":
    # Create an instance of the barbie robot
    robot = barbie()
    # Run the simulation with sliders
    run_with_sliders(robot)