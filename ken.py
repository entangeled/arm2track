##  @file
#   @brief UR3 Robot defined by standard DH parameters with 3D model
#   @author Ho Minh Quang Ngo
#   @date Jul 20, 2023

# @file
#   @brief UR3-like arm defined by standard DH parameters with 3D model
#   @author Ho Minh Quang Ngo
#   @date Jul 20, 2023

import time
import os

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D

# Useful variables
from math import pi, sqrt, atan

# -----------------------------------------------------------------------------------#
class ken(DHRobot3D):
    def __init__(self):
        """
        UR3 Robot by DHRobot3D class

        Example:
            >>> r = ken()
            >>> env = swift.Swift(); env.launch(realtime=True)
            >>> r.add_to_env(env)
            >>> q_goal = [r.q[i] - pi/4 for i in range(r.n)]
            >>> qtraj = rtb.jtraj(r.q, q_goal, 50).q
            >>> for q in qtraj:
            ...     r.q = q
            ...     env.step(0.02)
        """
        # DH links
        links = self._create_DH()

        # Names of the robot link files in the directory
        link3D_names = dict(
            link0="base",
            link1="joint2",
            link2="joint3",
            link3="joint4",
            link4="joint5",
            link5="joint6",  # fixed typo
            link6="cap",
        )

        # A joint config and the 3D object transforms to match that config (preview only)
        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [
            spb.transl(0, 0, 0.000),
            spb.transl(0, 0, 0.265),
            spb.transl(0, 0, 0.355),
            spb.transl(0.400, 0, 0.355),
            spb.transl(0.700, 0, 0.355),
            spb.transl(0.700, 0, 0.225),
            spb.transl(0.700, -0.090, 0.225),
        ]

        current_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "stl")
        super().__init__(
            links,
            link3D_names,
            name="barbie",
            link3d_dir=current_path,
            qtest=qtest,
            qtest_transforms=qtest_transforms,
        )
        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create robot's standard DH model (from provided figure).

        Standard DH parameters (meters, radians):
          Joint   theta        d        alpha      a         offset
          J1      θ1         0.267     -π/2       0          0
          J2      θ2         0         0          a2         T2_offset
          J3      θ3         0         -π/2       0.0775     T3_offset
          J4      θ4         0.3425    +π/2       0          0
          J5      θ5         0         -π/2       0.076      0
          J6      θ6         0.097     0          0          0
        where:
          a2 = sqrt(284.5^2 + 53.5^2) mm = 0.28948866 m
          T2_offset = -atan(284.5 / 53.5) = -1.3849179 rad
          T3_offset = +1.3849179 rad
        """
        a2 = sqrt(284.5**2 + 53.5**2) / 1000.0
        T2_offset = -atan(284.5 / 53.5)
        T3_offset = -T2_offset

        l0 = rtb.RevoluteDH(d=0.267,  a=0.0,    alpha=-pi/2, offset=0.0,       qlim=[-pi, pi])  # J1
        l1 = rtb.RevoluteDH(d=0.0,    a=a2,     alpha=0.0,   offset=T2_offset, qlim=[-pi, pi])  # J2
        l2 = rtb.RevoluteDH(d=0.0,    a=0.0775, alpha=-pi/2, offset=T3_offset, qlim=[-pi, pi])  # J3
        l3 = rtb.RevoluteDH(d=0.3425, a=0.0,    alpha=pi/2,  offset=0.0,       qlim=[-pi, pi])  # J4
        l4 = rtb.RevoluteDH(d=0.0,    a=0.076,  alpha=-pi/2, offset=0.0,       qlim=[-pi, pi])  # J5
        l5 = rtb.RevoluteDH(d=0.097,  a=0.0,    alpha=0.0,   offset=0.0,       qlim=[-pi, pi])  # J6

        return [l0, l1, l2, l3, l4, l5]

    # -----------------------------------------------------------------------------------#
    def test(self):
        """
        Test the class by adding 3D objects into a Swift window and do a simple movement
        """
        env = swift.Swift()
        env.launch(realtime=True)
        self.q = self._qtest
        self.base = SE3(0.5, 0, 0.5)
        self.add_to_env(env)

        q_goal = [self.q[i] + pi / 3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 100).q
        for q in qtraj:
            self.q = q
            env.step(0.05)
        time.sleep(3)
        env.hold()

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = ken()
    r.test()

