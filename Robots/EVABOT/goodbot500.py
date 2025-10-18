##  @file
#   @brief UR3 Robot defined by standard DH parameters with 3D model
#   @author Ho Minh Quang Ngo
#   @date Jul 20, 2023

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
from ir_support import CylindricalDHRobotPlot
import time
import os

# Useful variables
from math import pi

# -----------------------------------------------------------------------------------#
class goodbot500(DHRobot3D):
    def __init__(self):
        links = self._create_DH()

        # Names of the robot link files in the directory
        link3D_names = dict(
            link0="stationary_base",    # base
            link1="rotation_base",      # link 1
            link2="link1",              # link 2
            link3="link2",              # link 3
            link4="link3",              # link 4
            link5="link4",              # link 5
            link6="end_effector_cap"    # tool cap
        )


        # A joint config and the 3D object transforms to match that config
        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [
            spb.transl(0,0,0)       @ spb.rpy2tr(0,0,0),          # base mesh (world)
            spb.transl(0,0,0.2)     @ spb.rpy2tr(0,0,0),          # link1 mesh (world)
            spb.transl(0,0,0.29)    @ spb.rpy2tr(0,pi/2,0),       # link2 mesh (world)
            spb.transl(0.7,0,0.29)  @ spb.rpy2tr(0,0,0),          # link3 mesh (world)
            spb.transl(1.3,0,0.29)  @ spb.rpy2tr(0,pi/2,0),       # link4 mesh (world)
            spb.transl(1.3,0,0.09)  @ spb.rpy2tr(0,pi/2,pi),      # link5 mesh (world)
            spb.transl(1.3,-0.08,0.01) @ spb.rpy2tr(0,0,pi),      # tool cap (world, rides last link)
        ]

        current_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'stl')
        super().__init__(links, link3D_names, name = 'goodbot500', link3d_dir = current_path, qtest = qtest, qtest_transforms = qtest_transforms)
        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create robot's standard DH model
        """
        l0 = rtb.RevoluteDH(d=0.29, a=0.0, alpha=pi/2, qlim=[-pi, pi])  # base yaw (z-rotation)
        l1 = rtb.RevoluteDH(d=0, a=0.7, alpha=0, qlim=[-pi, pi])
        l2 = rtb.RevoluteDH(d=0.0, a=0.6, alpha=0.0, qlim=[-pi, pi])
        l3 = rtb.RevoluteDH(d=0, a=0.0, alpha=pi/2)
        l4 = rtb.RevoluteDH(d=0.28, a=0, alpha=-pi/2)
        l5 = rtb.RevoluteDH(d=0.08, a=0.0, alpha=0)
        links = [l0, l1, l2, l3, l4, l5]

        return links

    # -----------------------------------------------------------------------------------#
    def test(self):
        """
        Test the class by adding 3d objects into a new Swift window and do a simple movement
        """
        env = swift.Swift()
        env.launch(realtime= True)
        self.q = self._qtest
        self.base = SE3(0,0,0)
        self.add_to_env(env)

        q_goal = [pi/2,pi/2,0,pi/2,pi/2,pi]
        qtraj = rtb.jtraj(self.q, q_goal, 150).q
        fig = self.plot(self.q)
        for q in qtraj:
            self.q = q
            env.step(0.02)
            # fig.step(0.01)
        time.sleep(3)
        env.hold()

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = goodbot500()
    r.test()

