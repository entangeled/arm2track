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
            link0="stationary_base", color0 = "pink",   # base
            link1="rotation_base", color1 = "white",    # link 1
            link2="link1", color2 = "pink",              # link 2
            link3="link2", color3 = "white",              # link 3
            link4="link3", color4 = "pink",            # link 4
            link5="link4", color5 = "white",             # link 5
            link6="end_effector_cap", color6 = "pink"   # tool cap
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

# ---------- Swift helper ----------
    def add_to_env(self, env):
        """Add all meshes/cylinders to Swift (same as your Eva_demo)."""
        super().add_to_env(env)

    # ---------- Frame converters ----------
    def _T_in_base(self, T_world: SE3) -> SE3:
        """World -> robot base frame."""
        return SE3(self.base).inv() * T_world

    def _T_out_of_base(self, T_in_base: SE3) -> SE3:
        """Robot base frame -> world."""
        return SE3(self.base) * T_in_base

    # ---------- IK + follow ----------
    def move_tool_to(self, env, T_world: SE3, steps: int = 80, label: str = "") -> bool:
        """
        Solve IK (Levenberg–Marquardt) to a world pose and follow a joint trajectory.
        Returns True on success, False otherwise.
        """
        T_goal_base = self._T_in_base(T_world)
        sol = self.ikine_LM(T_goal_base, q0=np.asarray(self.q, dtype=float))
        if not sol.success:
            print(f"[goodbot500 IK] failed at {label or 'target'}")
            return False

        traj = rtb.jtraj(self.q, sol.q, steps)
        for q in traj.q:
            self.q = q
            # If carrying, update the carried object
            if getattr(self, "_carry", None):
                Ttool_b = self.fkine(self.q)                      # base->tcp
                Ttool_w = self._T_out_of_base(Ttool_b)            # world->tcp
                block, T_tool_to_block, idx, zones = self._carry   # unpack
                T_now = Ttool_w * T_tool_to_block
                block.T = T_now.A
                zones.set_override(idx, T_now)
            env.step(0.02)
        return True

    # ---------- Attach/Detach (visual) ----------
    def _attach_now(self, zones, idx: int):
        """
        Measure the tool->object SE3 so the object doesn't teleport while carried.
        Assumes the tool is already at grasp height over zones.blocks[idx].
        """
        block = zones.blocks[idx]
        Ttool_b = self.fkine(self.q)
        Ttool_w = self._T_out_of_base(Ttool_b)
        T_block_w = SE3(block.T)
        T_tool_to_block = Ttool_w.inv() * T_block_w
        self._carry = (block, T_tool_to_block, idx, zones)

    def _detach(self):
        self._carry = None

    # ---------- Public: pick & place a red block by index ----------
    def pick_and_place_block(self, env, zones, src: int, dst: int, lift: float = 0.18):
        """
        VISUAL pick & place between TrackLinkedZones indices (e.g., 5 -> 7).
        This uses tool-down orientation and leaves the block at dst using a zones override.
        """
        # tool-down orientation, same everywhere for simple IK
        TOOL_DOWN = SE3.RPY([np.pi/2, 0, -np.pi], order="xyz")

        # source/destination nominal frames (from your zones)
        T_src_ref = zones._block_pose_for_index(src)
        T_dst_ref = zones._block_pose_for_index(dst)

        p_src, p_dst = T_src_ref.t, T_dst_ref.t

        T_src       = SE3(p_src) * TOOL_DOWN
        T_dst       = SE3(p_dst) * TOOL_DOWN
        T_src_hover = SE3([p_src[0], p_src[1], p_src[2] + lift]) * TOOL_DOWN
        T_dst_hover = SE3([p_dst[0], p_dst[1], p_dst[2] + lift]) * TOOL_DOWN
        T_mid       = SE3(((T_src_hover.t + T_dst_hover.t) * 0.5)) * TOOL_DOWN

        # keep zones from snapping while we move it
        zones.set_override(src, T_src)

        # hover → down → attach
        if not self.move_tool_to(env, T_src_hover, label="src_hover"): return False
        if not self.move_tool_to(env, T_src,       label="src_down"):  return False
        self._attach_now(zones, src)

        # lift → via → dst_hover → down
        if not self.move_tool_to(env, T_src_hover, label="lift"):       return False
        if not self.move_tool_to(env, T_mid,       label="via"):        return False
        if not self.move_tool_to(env, T_dst_hover, label="dst_hover"):  return False
        if not self.move_tool_to(env, T_dst,       label="dst_down"):   return False

        # detach + leave the block at dst (override so it stays)
        block, T_tool_to_block, _, _ = self._carry
        Ttool_b = self.fkine(self.q)
        Ttool_w = self._T_out_of_base(Ttool_b)
        T_now = Ttool_w * T_tool_to_block
        block.T = T_now.A
        zones.set_override(src, T_now)
        self._detach()

        # clear away
        self.move_tool_to(env, T_dst_hover, label="clear")
        print(f"[goodbot500] moved orange block {src} → {dst}")
        return True

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = goodbot500()
    r.test()

