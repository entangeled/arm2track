# DVRobot --> imported from "Yet to decide"
# Basic skeleton layout for class and functions

import numpy as np
from roboticstoolbox import jtraj
from spatialmath import SE3
from ir_support import UR3
from spatialmath.base import transl
from math import pi
import time

class MiddleRobot:
    def __init__(self):
    
        # Use UR3 as dummy robot for now
        self.robot = UR3()
        
        # Default configurations
        self.home_pose = np.zeros(self.robot.n)  # All joints at 0
        self.current_pose = self.home_pose.copy()
        
        # Motion parameters
        self.default_steps = 50  # Default trajectory steps
        self.approach_height = 0.1  # Height above target for approach (meters)
        
        print(f"DVRobot created with {self.robot.name}")
        print(f"Joint limits: \n{self.robot.qlim}")

    def pick_up_object(self, target_position: np.array) -> bool:
        """
        Pick up object from specified pose
        target_position: [x, y, z] coordinates for pickup location 
        """
        print(f"Starting pickup at position: {target_position}")
        
        try:
            # Step 1: Create target transformation matrix
            
            T_target = SE3.Trans(target_position) * SE3.Rx(pi)  # Rotate gripper to face it down
            
            # Step 2: Create approach pose (above target)
            T_approach = T_target * SE3.Trans(0, 0, self.approach_height)
            
            # Step 3: Solve inverse kinematics for approach pose
            q_approach = self.robot.ikine_LM(T_approach, q0=self.current_pose).q
            if not self._check_joint_limits(q_approach):
                print("ERROR: Approach pose violates joint limits")
                return False
            
            # Step 4: Solve inverse kinematics for pickup pose
            q_pickup = self.robot.ikine_LM(T_target, q0=q_approach).q
            if not self._check_joint_limits(q_pickup):
                print("ERROR: Pickup pose violates joint limits")
                return False
            
            # Step 5: Execute trajectory to approach pose
            print("Moving to approach position...")
            traj_approach = jtraj(self.current_pose, q_approach, self.default_steps)
            self._execute_trajectory(traj_approach.q)
            
            # Step 6: Execute trajectory to pickup pose
            print("Descending to pickup position...")
            traj_pickup = jtraj(q_approach, q_pickup, self.default_steps)
            self._execute_trajectory(traj_pickup.q)
            
            # Step 7: Needs to be done- Activate gripper/end-effector
            print("Gripper activated - object picked up")
            
            # Step 8: Return to approach pose
            print("Lifting object...")
            traj_lift = jtraj(q_pickup, q_approach, self.default_steps)
            self._execute_trajectory(traj_lift.q)
            
            self.current_pose = q_approach
            print("Pickup completed successfully")
            return True
            
        except Exception as e:
            print(f"Pickup failed: {e}")
            return False

    def drop_off(self, target_position: np.array) -> bool:
        """
        Drop off object at specified pose
        target_position: [x, y, z] coordinates for drop-off location 
        """
        print(f"Starting drop-off at position: {target_position}")
        
        try:
            # Step 1: Create target transformation matrix
            T_target = SE3.Trans(target_position) * SE3.Rx(pi)  # Rotate gripper to face it down
                
            # Step 2: Create approach pose (above target)
            T_approach = T_target * SE3.Trans(0, 0, self.approach_height)
            
            # Step 3: Solve inverse kinematics for approach pose
            q_approach = self.robot.ikine_LM(T_approach, q0=self.current_pose).q
            if not self._check_joint_limits(q_approach):
                print("ERROR: Approach pose violates joint limits")
                return False
            
            # Step 4: Solve inverse kinematics for drop-off pose
            q_dropoff = self.robot.ikine_LM(T_target, q0=q_approach).q
            if not self._check_joint_limits(q_dropoff):
                print("ERROR: Drop-off pose violates joint limits")
                return False
            
            # Step 5: Execute trajectory to approach pose
            print("Moving to approach position...")
            traj_approach = jtraj(self.current_pose, q_approach, self.default_steps)
            self._execute_trajectory(traj_approach.q)
            
            # Step 6: Execute trajectory to drop-off pose
            print("Descending to drop-off position...")
            traj_dropoff = jtraj(q_approach, q_dropoff, self.default_steps)
            self._execute_trajectory(traj_dropoff.q)
            
            # Step 7: TODO - Deactivate gripper/end-effector
            print("Gripper deactivated - object dropped off")
            
            # Step 8: Return to approach pose
            print("Retracting from drop-off...")
            traj_retract = jtraj(q_dropoff, q_approach, self.default_steps)
            self._execute_trajectory(traj_retract.q)
            
            self.current_pose = q_approach
            print("Drop-off completed successfully")
            return True
            
        except Exception as e:
            print(f"Drop-off failed: {e}")
            return False

    def go_home(self):
        """
        Return robot to home position
        """
        print("Returning to home position...")
        try:
            traj_home = jtraj(self.current_pose, self.home_pose, self.default_steps)
            self._execute_trajectory(traj_home.q)
            self.current_pose = self.home_pose
            print("Home position reached")
            return True
        except Exception as e:
            print(f"Failed to return home: {e}")
            return False

    def _execute_trajectory(self, trajectory):
        """
        Execute joint trajectory (placeholder for actual robot control)
        trajectory: Array of joint configurations (steps x joints)
        """
        # TODO: Replace with actual robot control commands
        # For now, just update final position
        if len(trajectory) > 0:
            self.current_pose = trajectory[-1]
            # Simulate execution time
            time.sleep(0.1)

    def _check_joint_limits(self, q) -> bool:
        """
        Check if joint configuration is within limits
        q: Joint configuration array
        """
        q = np.array(q)
        # CORRECT - Returns True only when ALL joints are within limits
        within_lower = np.all(q >= self.robot.qlim[0])
        within_upper = np.all(q <= self.robot.qlim[1])
        return within_lower and within_upper

    def get_current_position(self):
        """
        Get current end-effector Cartesian position
        tuple: (position, orientation) of end-effector
        """
        T = self.robot.fkine(self.current_pose)
        return T.t, T.R

    def plot_robot(self):
        """
        Plot current robot configuration
        """
        self.robot.plot(self.current_pose)

# Example usage
if __name__ == "__main__":
    
    robot = MiddleRobot()
    
    # Pretend train track locations
    pickup_location = [0.5, 0.2, 0.1]    # x, y, z coordinates
    dropoff_location = [0.5, -0.2, 0.1]  # x, y, z coordinates
    
    print("--- Trial pick-up put-down example ---")
    
    # Pick up object
    success = robot.pick_up_object(pickup_location)
    if success:
        print("<YES> Pickup successful")
        
        # Drop off object  
        success = robot.drop_off(dropoff_location)
        if success:
            print("<YES> Drop-off successful --- YAY SUCCESS! ---")
        else:
            print("<NO> Drop-off failed")
    else:
        print("<NO> Pickup failed")
    
    # Return home
    robot.go_home()
    
    # Get current position
    position, rotation = robot.get_current_position()
    print(f"Final position: {position}")
    
    print("\n--- DVRobot Operations Complete ---\n")
