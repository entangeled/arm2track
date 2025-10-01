"""
KUKA LBR iiwa 7 R800 Robot Model 
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
from math import pi
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import swift
from ir_support import CylindricalDHRobotPlot

# ============================================================================
# ROBOT CLASS
# ============================================================================
class KUKAiiwa:
    """
    KUKA LBR iiwa 7 R800 - 7-DOF Collaborative Robot
    
    DH Parameters based on KUKA specifications:
    - Reach: 800mm
    - Payload: 7kg
    - 7 Revolute Joints
    
    Reference: https://www.robot-forum.com/robotforum/thread/15102-how-to-obtain-dh-parameter-from-kuka-iiwa-7-r800/
    """
    
    def __init__(self):
        
        # DH Parameters for KUKA LBR iiwa 7 R800
        # Using Modified DH Convention (Proximal)
        # Format: RevoluteDH(d, a, alpha, offset)
        # All lengths in meters, angles in radians
        
        # Link 1: Base to Joint 1

        d = [0.340, 0, 0.400, 0, 0.400, 0, 0.126]
        a = [0, 0, 0, 0, 0, 0, 0]
        alpha = [pi/2, pi/2, pi/2, pi/2, pi/2, pi/2, pi/2]
        qlim = [
                (-170*pi/180, 170*pi/180), 
                (-120*pi/180, 120*pi/180), 
                (-170*pi/180, 170*pi/180),
                (-120*pi/180, 120*pi/180), 
                (-170*pi/180, 170*pi/180), 
                (-120*pi/180, 120*pi/180), 
                (-175*pi/180, 175*pi/180)
                ]
        L = []
        for i in range(7):
            link = RevoluteDH(d= d[i], a= a[i], alpha= alpha[i], qlim= qlim[i])
            L.append(link)
        self.robot = DHRobot(L, name="KUKA_iiwa_7_R800")
    
        
        # Set base transform if needed (robot sits on ground)
        self.robot.base = SE3(0, 0, 0)
        
        # Define home position (all joints at 0)
        self.q_home = np.array([0, 0, 0, 0, 0, 0, 0])
        
        # Define a ready position (comfortable starting pose)
        self.q_ready = np.array([0, -30, 0, 60, 0, -90, 0]) * pi/180
        
        print(f"> {self.robot.name} initialized successfully!")
        print(f"  - DOF: {self.robot.n}")
        print(f"  - Reach: ~800mm")
        print(f"  - Payload: 7kg")

        # Create the cylinders for the robot model
        cyl_viz = CylindricalDHRobotPlot(self.robot, cylinder_radius=0.05, color="#3478f6")
        self.model = cyl_viz.create_cylinders()
    
    def show_in_swift(self):
        """Launch Swift simulator and display the robot"""
        
        # Create Swift environment
        env = swift.Swift()
        env.launch(realtime=True)
        
        # Add robot to environment
        env.add(self.robot)
        
        # Set initial pose to ready position
        self.robot.q = self.q_ready
        
        print("\n> Swift environment launched!")
        print("  - Robot displayed in ready position")
        print("  - Close the Swift window when done")
        
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
        T = self.robot.fkine(self.q_home)
        
        print(f"  Position (x, y, z): {T.t}")
        # print(f"  Total reach: {np.linalg.norm(T.t):.3f}m")
        
        T_ready = self.robot.fkine(self.q_ready)
    
        
        print("\n" + "="*50)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("KUKA LBR iiwa 7 R800 Robot Initialization")
    print("="*50 + "\n")
    
    # Create robot instance
    robot = KUKAiiwa()
    
    # Test kinematics
    robot.test_kinematics()
    
    # Launch Swift simulator
    print("\nLaunching Swift simulator...")
    print("(Press Ctrl+C to exit)")
    robot.show_in_swift()