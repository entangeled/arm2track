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
from roboticstoolbox import DHRobot, RevoluteDH, Robot
from spatialmath import SE3
import swift
from ir_support import CylindricalDHRobotPlot, DHRobot3D
from spatialgeometry import Mesh
import spatialmath.base as spb
from pathlib import Path
import os
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
        
        # DH Parameters for KUKA LBR iiwa 7 R800
        # Format: RevoluteDH(d, a, alpha, offset)
        # All lengths in meters, angles in radians
        
        # Link 1 (0.340m) : Base to Joint 1
        # Link 2 (0.400m) : Joint 1 to joint 3 (combined l2+l3)
        # Link 3 (0.400m) : Joint 3 to joint 5 (combined l4+l5)
        # Link 4 (0.126m) : Joint 5 to flange
    
        #  ============ DH PARAMETERS ============
        d = [0.340, 0, 0.400, 0, 0.400, 0, 0.126]
        a = [0, 0, 0, 0, 0, 0, 0]
        alpha = [pi/2, pi/2, pi/2, pi/2, pi/2, pi/2, pi/2]
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

        #  ============ NAMES & QTEST TRANSFORMS ============

        link3D_names = dict(
            link0='link_0',
            link1='link_1',
            link2='link_2',
            link3='link_3',
            link4='link_4',
            link5='link_5',
            link6='link_6',
            link7='link_7'
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
       
        mesh_dir = Path(__file__).parent
        mesh_path = mesh_dir / "iiwa7_mesh_files" 
        print(f"Current Path: \n    {mesh_path}")
        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(links, link3D_names, name = "KUKA_iiwa_7_R800", link3d_dir = mesh_path, qtest = qtest, qtest_transforms = qtest_transforms)
    

    def show_in_swift(self):
        """Launch Swift simulator and display the robot"""
        
        # Create Swift environment
        env = swift.Swift()
        env.launch(realtime=True)
        # Add robot to environment
        self.add_to_env(env)
        self.q = self.q_home

        #  Cylinder visualisation using a model
        # cyl_viz = CylindricalDHRobotPlot(self.robot, cylinder_radius=0.05, color="#3478f6")
        # self.cylinders = cyl_viz.create_cylinders()
        # self.cylinders.q = self.q_ready
        # env.add(self.cylinders)
    
        print("\n> Swift environment launched!")
        print("  - Press Ctrl + C to exit")
        
        # Keep environment running
        try:
            while True:
                env.step(0.05)  # Update at 20Hz
        except KeyboardInterrupt:
            print("\n> Shutting down Swift...")
            env.close()
    
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
            [0, 0, pi/2, 0, 0, 0, 0],        
            [pi/2, pi/2, 0, pi/2, 0, 0, 0],     
            [pi/2, pi/2, pi/2, pi/2, pi/2, pi/2, pi/2] # More bends
        ]
        
        for q in poses:
            iiwa.q = q
            env.step()
            input("Press Enter for next pose...")
        
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

    # print("\nLaunching Swift simulator...")
    # iiwa.show_in_swift()
