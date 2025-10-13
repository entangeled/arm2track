#skeleton code for track 
#code made by: Eva R. Gaarder (25089246) 



#a multi arm track system 

import rclpy 
from rclpy.node import Node
from moveit_commander import RobotCommander, MoveGroupCommander
import math 
import time
import numpy as np
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from sensor_msgs.msg import JointState


# 40 degrees for each arm, and they move 2 each
# swap arms when they reach 40 degrees (the tracking)

#using UR5 robot
class UR5CirciularMotion(Node): 
    def __init__(self):
        super().__init__('ur5_circular_motion')
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander("manipulator")
        self.joint_publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.joint_state = JointState()
        self.joint_state.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_state.position = [0.0] * 6
        self.angle = 0.0

    def move_in_cirlce(self):
        radius = 0.5  # Radius of the circular path
        center_x = 0.5  # Center of the circle in x
        center_y = 0.0  # Center of the circle in y
        num_points = 100  # Number of points to generate

        for i in range(num_points):
            theta = (2 * math.pi / num_points) * i
            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta)
            z = 0.5  # Fixed height

            pose_target = Pose()
            pose_target.position.x = x
            pose_target.position.y = y
            pose_target.position.z = z
            pose_target.orientation.w = 1.0

            self.group.set_pose_target(pose_target)
            self.group.go(wait=True)
            self.group.stop()
            self.group.clear_pose_targets()
            time.sleep(0.1)

    def timer_callback(self):
        # Update joint angles for circular motion
        self.angle += math.radians(5)  # Increment angle by 5 degrees
        if self.angle >= 2 * math.pi:
            self.angle = 0.0

        # Calculate new joint positions for circular motion
        radius = 0.5  # Radius of the circular path
        x = radius * math.cos(self.angle)
        y = radius * math.sin(self.angle)
        
        # Inverse kinematics to find joint angles (simplified example)
        self.joint_state.position[0] = math.atan2(y, x)  # Base rotation
        self.joint_state.position[1] = math.radians(45)   # Shoulder lift
        self.joint_state.position[2] = math.radians(-90)  # Elbow
        self.joint_state.position[3] = math.radians(45)   # Wrist 1
        self.joint_state.position[4] = 0.0                 # Wrist 2
        self.joint_state.position[5] = 0.0                 # Wrist 3

        # Publish the joint states
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_publisher.publish(self.joint_state)


    def main(args=None):
        rclpy.init(args=args)
        ur5_circular_motion = UR5CirciularMotion()
        ur5_circular_motion.move_in_cirlce()
        rclpy.spin(ur5_circular_motion)
        ur5_circular_motion.destroy_node()
        rclpy.shutdown()

    if __name__ == '__main__':
        main()
        

