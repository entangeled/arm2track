"""
Author: Danil Vasiliev (25299862)
KUKA LBR iiwa 7 R800 Robot Model 
Custom DH Parameters with Official KUKA Meshes
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
from math import pi
from roboticstoolbox import DHRobot, RevoluteDH, Robot, jtraj
from spatialmath import SE3
import swift
from ir_support import CylindricalDHRobotPlot, DHRobot3D
from spatialgeometry import Mesh
import spatialmath.base as spb
from pathlib import Path
import os
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# ============================================================================
# ROBOT CLASS
# ============================================================================
class KUKAiiwa(DHRobot3D):
    """
    KUKA LBR iiwa 7 R800 - 7-DOF Collaborative Robot
    
    DH Parameters based on KUKA specifications:
    - Reach: 800mm
    - Payload: 7kg
    - 7 Revolute Joints
    Visual Meshes: Official KUKA URDF (for visualization only)

    Reference: https://xpert.kuka.com/service-express/portal/project1_p/document/kuka-project1_p-basic_AR1089_en?context=%7B%22filter%22%3A%7B%22ProductCategory%22%3A%5B%22PC531%22%5D%7D,%22text%22%3A%22LBR%20iiwa%20R800%22,%22useExpertQuery%22%3A0%7D
    
    """
    
    def __init__(self):
        
        #  ============ NAMES & QTEST TRANSFORMS ============

        links = self._create_DH()

        link3D_names = dict(
            link0='link_0', color0= [1.0, 0.4, 0.0, 1.0],
            link1='link_1', color1= 'white',
            link2='link_2', color2= [1.0, 0.4, 0.0, 1.0],
            link3='link_3', color3= 'white',
            link4='link_4', color4= [1.0, 0.4, 0.0, 1.0],
            link5='link_5', color5= 'white',
            link6='link_6', color6= [1.0, 0.4, 0.0, 1.0],
            link7='link_7', color7= 'white'
        )
        
        qtest = [0, 0, 0, 0, 0, 0, 0]
        qtest_transforms = [
            spb.transl(0, 0, 0) @ spb.trotz(pi),                                                # Base / link_0
            spb.transl(0, 0, 0.158),                                                            # link_1
            spb.transl(0, 0, 0.34) @ spb.trotz(pi/2) @ spb.troty(pi/2) @ spb.trotz(pi/2),       # link_2
            spb.transl(0, 0, 0.525),                                                            # link_3
            spb.transl(0, 0, 0.74) @ spb.trotz(3*pi/2) @ spb.troty(pi/2) @ spb.trotz(pi/2),     # link_4
            spb.transl(0, 0, 0.925),                                                            # link_5
            spb.transl(0, 0.06, 1.14) @ spb.trotx(pi/2),                                        # link_6
            spb.transl(0, 0, 1.22)                                                              # link_7
        ]
        """
        qtest_SE3_transforms = [
            (SE3(0, 0, 0) @ SE3.Rz(pi)),                                        # Base - no offset needed
            (SE3(0, 0, 0.158)),                                                 # Link 1
            (SE3(0, 0, 0.34) @ SE3.Rz(pi/2) @ SE3.Ry(pi/2) @ SE3.Rz(pi/2)),     # Link 2  
            (SE3(0, 0, 0.525)),                                                 # Link 3
            (SE3(0, 0, 0.74) @ SE3.Rz(3*pi/2) @ SE3.Ry(pi/2) @ SE3.Rz(pi/2)),   # Link 4
            (SE3(0, 0, 0.925)),                                                 # Link 5
            (SE3(0, 0.06, 1.14) @ SE3.Rx(pi/2)),                                # Link 6
            (SE3(0, 0, 1.22)),                                                  # Link 7
        ]
        """
        
        # Define home position (all joints at 0)
        self.q_home = np.array([0, 90, 0, 90, 90, 90, 0]) * pi/180
        
        # Define a ready position (comfortable starting pose)
        self.q_ready = np.array([0, 0, 0, 0, 0, 0, 0]) * pi/180
       
        # Get stl files path
        mesh_dir = Path(__file__).parent
        mesh_path = mesh_dir / "iiwa7_mesh_files" 

        super().__init__(links, link3D_names, name = "KUKA_iiwa_7_R800", link3d_dir = mesh_path, qtest = qtest, qtest_transforms = qtest_transforms)
    
    def _create_DH(self):
        """
        Create robot's standard DH model
        """
        #  ============ LINK INFO ============

        # Link 1 (0.340m) : Base to Joint 1
        # Link 2 (0.400m) : Joint 1 to joint 3 (combined l2+l3)
        # Link 3 (0.400m) : Joint 3 to joint 5 (combined l4+l5)
        # Link 4 (0.126m) : Joint 5 to flange
    
        #  ============ DH PARAMETERS ============

        d = [0.340, 0, 0.400, 0, 0.400, 0, 0.126]
        a = [0, 0, 0, 0, 0, 0, 0]
        alpha = [-pi/2, pi/2, pi/2, -pi/2, -pi/2, pi/2, 0]
        qlim = [
                (-170 *pi/180, 170 *pi/180), 
                (-120 *pi/180, 120 *pi/180), 
                (-170 *pi/180, 170 *pi/180),
                (-120 *pi/180, 120 *pi/180), 
                (-170 *pi/180, 170 *pi/180), 
                (-120 *pi/180, 120 *pi/180), 
                (-175 *pi/180, 175 *pi/180)
                ]
        links = []
        for i in range(7):
            link = RevoluteDH(d= d[i], a= a[i], alpha= alpha[i], qlim= qlim[i])
            links.append(link)
        return links

    def test_kinematics(self):
        """Test forward kinematics at home position"""
        
        print("\n" + "="*50)
        print("TESTING KINEMATICS")
        print("="*50)
        
        # Calculate forward kinematics at home position
        T = self.fkine(self.q_home)
        T_rounded = np.round(T.t, 4)
        print(f"  Position (x, y, z): {T_rounded}")
        # print(f"  Total reach: {np.linalg.norm(T.t):.3f}m")
        
        T_ready = self.fkine(self.q_ready)
        print("\n" + "="*50)

    def test_in_swift(self):
        env = swift.Swift()
        env.launch(realtime=True)
        self.add_to_env(env)
        
        # Move through a few poses quickly
        poses = [
            [0, 0, 0, 0, 0, 0, 0],           # Straight up
            [pi/2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, pi/2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, pi/2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, pi/2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, pi/2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, pi/2, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, pi/2]
        ]
        input("Press enter to start...")
        for q in poses:
            trajectory = jtraj(iiwa.q, q, 50).q
            for i in trajectory:
                iiwa.q = i
                env.step(0.03)
            print(f"    Pose: {q}")
            time.sleep(1)
        
        env.close()

    def test_dh_vs_visual_in_swift(self):
        """
        Visualize DH end-effector frames vs visual mesh positions in Swift.
        Places colored spheres 0.3m away in Y direction to show where DH frames are.
        Robot visual updates simultaneously for side-by-side comparison.
        
        Sphere Colors:
        - Black: Base (frame 0)
        - Red: Joint 1 (frame 1)
        - Green: Joint 2 (frame 2)
        - Blue: Joint 3 (frame 3)
        - Yellow: Joint 4 (frame 4)
        - Magenta: Joint 5 (frame 5)
        - Cyan: Joint 6 (frame 6)
        - Orange: Joint 7/End-effector (frame 7)
        """
        env = swift.Swift()
        env.launch(realtime=True)
        self.add_to_env(env)
        env.set_camera_pose([3, 3, 2], [0, 0, 0.6])
        
        # Test poses
        poses = [
            [0, 0, 0, 0, 0, 0, 0],           # Straight up
            [pi/2, 0, 0, 0, 0, 0, 0],        # Joint 1
            [0, 0, 0, 0, 0, 0, 0],
            [0, pi/2, 0, 0, 0, 0, 0],        # Joint 2
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, pi/2, 0, 0, 0, 0],        # Joint 3
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, pi/2, 0, 0, 0],        # Joint 4
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, pi/2, 0, 0],        # Joint 5
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, pi/2, 0],        # Joint 6
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, pi/2]         # Joint 7
        ]
        
        # Define colors for each frame
        frame_colors = [
            [0.0, 0.0, 0.0, 1.0],      # 0: Black - Base
            [1.0, 0.0, 0.0, 1.0],      # 1: Red - Joint 1
            [0.0, 1.0, 0.0, 1.0],      # 2: Green - Joint 2
            [0.0, 0.0, 1.0, 1.0],      # 3: Blue - Joint 3
            [1.0, 1.0, 0.0, 1.0],      # 4: Yellow - Joint 4
            [1.0, 0.0, 1.0, 1.0],      # 5: Magenta - Joint 5
            [0.0, 1.0, 1.0, 1.0],      # 6: Cyan - Joint 6
            [1.0, 0.5, 0.0, 1.0],      # 7: Orange - Joint 7/End-effector
        ]
        
        frame_names = [
            "Base (frame 0)",
            "Joint 1 (frame 1)",
            "Joint 2 (frame 2)",
            "Joint 3 (frame 3)",
            "Joint 4 (frame 4)",
            "Joint 5 (frame 5)",
            "Joint 6 (frame 6)",
            "Joint 7/EE (frame 7)"
        ]
        
        # Create spheres to mark DH frame positions (offset in Y by 0.3m)
        from spatialgeometry import Sphere
        dh_frame_markers = []
        y_offset = 0.3  # Offset spheres by 0.3m in Y direction
        
        for i in range(8):  # 8 frames (base + 7 joints)
            marker = Sphere(radius=0.03, color=frame_colors[i])
            env.add(marker)
            dh_frame_markers.append(marker)
        
        # Print color legend
        print("\n" + "="*60)
        print("SPHERE COLOR LEGEND:")
        print("="*60)
        for i, name in enumerate(frame_names):
            color_name = ["Black", "Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange"][i]
            print(f"  {color_name:8s} - {name}")
        print("="*60)
        
        input("\nPress enter to start visualization...\n")
        iiwa.q = poses[0]
        for pose_idx, q in enumerate(poses):
            # Calculate DH frame positions FIRST (before updating visual)
            link_transforms = self._get_transforms(q)
            
            # Update robot visual to match the pose
            if pose_idx < 13:
                iiwa.q = poses[pose_idx +1]
            else:
                iiwa.q = q
            
            
            # Update DH frame marker positions with Y offset
            for i, T in enumerate(link_transforms):
                # Apply Y offset to the DH frame position
                T_offset = T.copy()
                T_offset[1, 3] += y_offset  # Add 0.3m in Y direction
                dh_frame_markers[i].T = T_offset
            
            # Update Swift environment (both robot and spheres move together)
            env.step(0.05)
            
            # Calculate end-effector position from DH
            T_ee = self.fkine(q)
            ee_pos = T_ee.t
            
            # Print DH information
            print("\n" + "="*60)
            print(f"Pose {pose_idx + 1}: q = {np.round(np.array(q) * 180/pi, 1)} deg")
            print("="*60)
            print(f"End-effector DH position: {np.round(ee_pos, 4)}")
            
            # Show all DH frame positions (actual positions, not offset)
            print("\nAll DH frame positions (X, Y, Z):")
            for i, T in enumerate(link_transforms):
                pos = T[:3, 3]
                color_name = ["Black", "Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange"][i]
                print(f"  {color_name:8s} sphere (Frame {i}): {np.round(pos, 4)}")
            
            time.sleep(0.5)
            
            # Highlight active joint
            active_joint = None
            for i, angle in enumerate(q):
                if abs(angle) > 0.01:  # Non-zero joint
                    active_joint = i
                    break
            
            if active_joint is not None:
                color_name = ["Black", "Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange"][active_joint + 1]
                print(f"\n>>> Active Joint: {active_joint + 1}")
                print(f">>> Look for the {color_name.upper()} SPHERE")
                print(f">>> Does the visual mesh joint align with the {color_name.lower()} sphere?")
                print(f">>> The mesh should rotate around the same point as this sphere!")
            
            input("Press enter for next pose...\n")
        
        env.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*50)
    print("KUKA LBR iiwa 7 R800 Robot TESTING")
    print("="*50 + "\n")
    
    iiwa = KUKAiiwa()
    # iiwa.test_kinematics()
    iiwa.test_in_swift()
    # iiwa.test_dh_vs_visual_in_swift()
