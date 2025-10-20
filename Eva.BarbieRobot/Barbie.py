##  @file
#   @brief UR3 Robot defined by standard DH parameters with 3D model
#   @author Ho Minh Quang Ngo
#   @date Jul 20, 2023

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import time
import os

# Useful variables
from math import pi

# -----------------------------------------------------------------------------------#
class barbie(DHRobot3D):
    def __init__(self):
        """
            UR3 Robot by DHRobot3D class

            Example usage:
            >>> from ir-support import UR3
            >>> import swift

            >>> r = UR3()
            >>> q = [0,-pi/2,pi/4,0,0,0]r
            >>> r.q = q
            >>> q_goal = [r.q[i]-pi/4 for i in range(r.n)]
            >>> env = swift.Swift()
            >>> env.launch(realtime= True)
            >>> r.add_to_env(env)
            >>> qtraj = rtb.jtraj(r.q, q_goal, 50).q
            >>> for q in qtraj:r
            >>>    r.q = q
            >>>    env.step(0.02)
        """
        # DH links
        links = self._create_DH()

        # Names of the robot link files in the directory
        link3D_names = dict(link0 = 'base',
                            link1 = 'joint1',
                            link2 = 'joint2',
                            link3 = 'joint3',
                            link4 = 'joint4',
                            link5 = 'joint5',
                            link6 = 'cap')

        # A joint config and the 3D object transforms to match that config
        qtest = [0,0,0,0,0,0]
        qtest_transforms = [spb.transl(0,0,0),
                            spb.transl(0,0,0.265),
                            spb.transl(0,0,0.355),
                            spb.transl(0.400,0,0.355),
                            spb.transl(0.7,0,0.355),
                            spb.transl(0.7,0,0.225),
                            spb.transl(0.7,-0.09,0.225)]

        current_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'stl')
        super().__init__(links, link3D_names, name = 'barbie', link3d_dir = current_path, qtest = qtest, qtest_transforms = qtest_transforms)
        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create robot's standard DH model
        """
        l0 = rtb.RevoluteDH(d=0.355, a=0.0, alpha=pi/2, qlim=[-pi, pi])  # base yaw (z-rotation)
        l1 = rtb.RevoluteDH(d=0, a=0.4, alpha=0, qlim=[-pi, pi])
        l2 = rtb.RevoluteDH(d=0.0, a=0.3, alpha=0.0, qlim=[-pi, pi])
        l3 = rtb.RevoluteDH(d=0, a=0.0, alpha=pi/2, qlim=[-pi, pi])
        l4 = rtb.RevoluteDH(d=0.13, a=0, alpha=-pi/2, qlim=[-pi, pi])
        l5 = rtb.RevoluteDH(d=0.105, a=0.0, alpha=0, qlim=[-pi, pi])
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
        self.base = SE3(0.5,0,0.5)
        self.add_to_env(env)
        

        q_goal = [self.q[i]+pi/3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 100).q
        # fig = self.plot(self.q)
        for q in qtraj:
            self.q = q
            env.step(0.05)
            # fig.step(0.01)
        time.sleep(3)
        env.hold()

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = barbie()
    r.test()

