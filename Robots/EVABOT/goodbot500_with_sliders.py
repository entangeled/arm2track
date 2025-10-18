# three_dof_meshes_with_cyl_viz.py
import os, time
from math import pi
import numpy as np
import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
from ir_support import CylindricalDHRobotPlot
from spatialgeometry import Mesh as SGMesh  # <-- dynamic mesh nodes

CYL_RADIUS = 0.005
CYL_COLOR  = "#3478f6"

class ThreeDOFMeshes(DHRobot3D):
    def __init__(self):
        # ---- STANDARD DH ----
        l0 = rtb.RevoluteDH(d=0.29, a=0.0, alpha=pi/2, qlim=[-2*pi, 2*pi])  # base yaw (z-rotation)
        l1 = rtb.RevoluteDH(d=0, a=0.7, alpha=0, qlim=[0, pi])
        l2 = rtb.RevoluteDH(d=0.0, a=0.6, alpha=0.0, qlim=[-2.5, 2.5])
        l3 = rtb.RevoluteDH(d=0, a=0.0, alpha=pi/2)
        l4 = rtb.RevoluteDH(d=0.28, a=0, alpha=-pi/2)
        l5 = rtb.RevoluteDH(d=0.08, a=0.0, alpha=0) 
        links = [l0, l1, l2, l3, l4, l5]

        # ---- STL meshes (files placed next to this script) ----
        # link0..link6 names (you provided these)
        self.link3D_names = dict(
            link0="stationary_base",    # base
            link1="rotation_base",      # link 1
            link2="link1",              # link 2
            link3="link2",              # link 3
            link4="link3",              # link 4
            link5="link4",              # link 5
            link6="end_effector_cap"    # tool cap
        )

        # reference joint pose + world placements of each STL at that pose
        self.qtest = [0, 0, 0, 0, 0, 0]
        self.qtest_transforms = [
            spb.transl(0,0,0)       @ spb.rpy2tr(0,0,0),          # base mesh (world)
            spb.transl(0,0,0.2)     @ spb.rpy2tr(0,0,0),          # link1 mesh (world)
            spb.transl(0,0,0.29)    @ spb.rpy2tr(0,pi/2,0),       # link2 mesh (world)
            spb.transl(0.7,0,0.29)  @ spb.rpy2tr(0,0,0),          # link3 mesh (world)
            spb.transl(1.3,0,0.29)  @ spb.rpy2tr(0,pi/2,0),       # link4 mesh (world)
            spb.transl(1.3,0,0.09)  @ spb.rpy2tr(0,pi/2,pi),      # link5 mesh (world)
            spb.transl(1.3,-0.08,0.01) @ spb.rpy2tr(0,0,pi),      # tool cap (world, rides last link)
        ]

        self.mesh_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'stl')

        # keep DH kinematics & name (but we won't let DHRobot3D draw its meshes)
        super().__init__(
            links,
            self.link3D_names,
            name="ThreeDOFMeshes",
            link3d_dir=self.mesh_dir,
            qtest=self.qtest,
            qtest_transforms=self.qtest_transforms,
        )

        self.q = self.qtest
        self.robot_for_cyl = rtb.DHRobot(self.links, name="ThreeDOFCyl")

        # ---- dynamic-mesh attachment setup (tiny + explicit) ----
        # Map each STL name -> which DH link frame it should ride (0-based link index).
        # Use -1 for the stationary base mesh.
        self.mesh_to_link = {
            "stationary_base": -1,   # base (not a DH link)
            "rotation_base":    0,   # rides DH link 0
            "link1":            1,   # rides DH link 1
            "link2":            2,
            "link3":            3,
            "link4":            4,
            "end_effector_cap": 5,   # rides last DH link
        }

        # Precompute per-mesh constant offsets:  offset = A_i(qtest)^-1 * mesh_world(qtest)
        # A_i list (world -> DH link i) at qtest
        self._A_qtest = [self.A(i, self.qtest) for i in range(len(self.links))]  # SE3 list, len = 6
        # Build offsets (SE3) for every mesh name
        self._offsets = {}
        for logical_key, stlname in zip(self.link3D_names.keys(), self.link3D_names.values()):
            idx = int(logical_key.replace("link", ""))  # 0..6 indexes your qtest_transforms
            mesh_world = SE3(self.qtest_transforms[idx])
            link_i = self.mesh_to_link.get(stlname, -1)
            if link_i == -1:
                # base mesh offset relative to world base = identity
                self._offsets[stlname] = SE3()  # just use mesh_world directly at runtime
            else:
                self._offsets[stlname] = self._A_qtest[link_i].inv() * mesh_world

        # Create Swift Mesh nodes we control (one per STL)
        self._sg_nodes = {}  # name -> SGMesh node
        for stlname, link_i in self.mesh_to_link.items():
            pth = os.path.join(self.mesh_dir, f"{stlname}.stl")
            if link_i == -1:
                # base mesh: place at base *its world at qtest* initially
                node = SGMesh(pth, pose=SE3(self.qtest_transforms[0]))
            else:
                node = SGMesh(pth, pose=(self._A_qtest[link_i] * self._offsets[stlname]))
            self._sg_nodes[stlname] = node

    # tiny helper to update mesh nodes given current q
    def sync_meshes(self):
        # Update each STL pose: base or A_i(q) @ offset
        # Base mesh (stationary)
        base_node = self._sg_nodes.get("stationary_base", None)
        if base_node is not None:
            base_node.T = SE3(self.qtest_transforms[0]).A  # stays where you positioned it
        # Link-attached meshes
        for stlname, link_i in self.mesh_to_link.items():
            if link_i == -1:
                continue
            A_i = self.A(link_i, self.q)                 # SE3 base->link_i at current q
            node = self._sg_nodes[stlname]
            node.T = (A_i * self._offsets[stlname]).A

def run_with_sliders(bot: ThreeDOFMeshes):
    env = swift.Swift()
    env.launch(realtime=True)

    bot.base = SE3(0, 0, 0)

    # ---- we do NOT call bot.add_to_env(env) to avoid duplicate static meshes ----
    # Add our dynamic STL nodes instead:
    for node in bot._sg_nodes.values():
        env.add(node)

    # ---- cylinder overlay (draws DH geometry) ----
    cyl = CylindricalDHRobotPlot(bot.robot_for_cyl, cylinder_radius=CYL_RADIUS, color=CYL_COLOR)
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

    # initial sync
    bot.sync_meshes()

    def on_slider(val_deg, j):
        bot.q[j] = val_deg * pi / 180.0
        robot_cyl.q = bot.q[:]      # cylinders follow DH
        bot.sync_meshes()           # meshes follow DH (this is the lock)
        xyz2, rpy2 = ee_xyz_rpy()
        labels[0].desc, labels[1].desc, labels[2].desc = f"X: {xyz2[0]}", f"Y: {xyz2[1]}", f"Z: {xyz2[2]}"
        labels[3].desc, labels[4].desc, labels[5].desc = f"Roll (φ): {rpy2[0]}", f"Pitch (θ): {rpy2[1]}", f"Yaw (ψ): {rpy2[2]}"

    # sliders
    s0 = swift.Slider(cb=lambda v: on_slider(v, 0), min=-180, max=180, step=1, value=0,  desc="Joint 1 (base yaw)", unit="°")
    s1 = swift.Slider(cb=lambda v: on_slider(v, 1), min=0, max=180, step=1, value=0,  desc="Joint 2", unit="°")
    s2 = swift.Slider(cb=lambda v: on_slider(v, 2), min=-145, max=145, step=1, value=0,  desc="Joint 3", unit="°")
    s3 = swift.Slider(cb=lambda v: on_slider(v, 3), min=-360, max=360, step=1, value=0,  desc="Joint 4", unit="°")
    s4 = swift.Slider(cb=lambda v: on_slider(v, 4), min=-360, max=360, step=1, value=0,  desc="Joint 5", unit="°")
    s5 = swift.Slider(cb=lambda v: on_slider(v, 5), min=-360, max=360, step=1, value=0,  desc="Joint 6", unit="°")
    env.add(s0); env.add(s1); env.add(s2); env.add(s3); env.add(s4); env.add(s5)

    try:
        while True:
            env.step()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run_with_sliders(ThreeDOFMeshes())
