#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import math
from tf_transformations import (quaternion_from_euler, concatenate_matrices, 
                                 translation_matrix, quaternion_matrix)
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint

class RoverRoutineBase(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.move_client = ActionClient(self, MoveGroup, "move_action")
        self.ee_link = "link_6"

    def get_tcp_matrix(self, tx, ty, tz, r, p, y):
        """Standard 4x4 Homogeneous Transformation"""
        T = translation_matrix([tx, ty, tz])
        R = quaternion_matrix(quaternion_from_euler(r, p, y))
        return concatenate_matrices(T, R)

    def execute_move(self, T_offset, x, y, z, pitch_deg, v_scale=0.15):
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action Server not found!")
            return

        pitch_rad = math.radians(pitch_deg)
        tool_length = T_offset[2, 3] # Extracting the length from the matrix

        # Calculate link_6 position to place tip at (x,y,z)
        adj_x = x - (tool_length * math.sin(pitch_rad))
        adj_z = z + (tool_length * math.cos(pitch_rad))

        c = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = "world"
        pc.link_name = self.ee_link
        
        target_pose = Pose()
        target_pose.position.x, target_pose.position.y, target_pose.position.z = adj_x, y, adj_z
        target_pose.orientation.y = math.sin(pitch_rad / 2.0)
        target_pose.orientation.w = math.cos(pitch_rad / 2.0)

        pc.constraint_region.primitive_poses.append(target_pose)
        # (Assuming you add the Sphere constraint here as in previous working versions)
        
        c.position_constraints.append(pc)
        goal = MoveGroup.Goal()
        goal.request.group_name = "arm_controller"
        goal.request.goal_constraints.append(c)
        goal.request.max_velocity_scaling_factor = v_scale
        
        self.get_logger().info(f"Executing move for tip target: {x, y, z}")
        return self.move_client.send_goal_async(goal)