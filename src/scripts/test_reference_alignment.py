#!/usr/bin/env python3
"""
Simple test: publish a PoseStamped to '/ee_reference_pose' with a 45° yaw rotation
and set teleop to use 'ee_ref' as reference frame by publishing to '/reference_frame'.

Run while the workspace is up (start_complete_stack.py) to see markers in RViz and
verify the teleop GUI can switch to the new frame.
"""
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf_transformations import quaternion_from_euler


class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_reference_alignment')
        self.pose_pub = self.create_publisher(PoseStamped, '/ee_reference_pose', 10)
        self.ref_pub = self.create_publisher(String, '/reference_frame', 10)

    def run_once(self):
        ps = PoseStamped()
        ps.header.frame_id = 'world'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = 0.3
        ps.pose.position.y = 0.0
        ps.pose.position.z = 0.3
        q = quaternion_from_euler(0.0, 0.0, math.radians(45.0))
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        self.pose_pub.publish(ps)
        self.get_logger().info('Published ee_reference_pose (45 deg yaw)')
        # Tell teleop (or other nodes) to use ee_ref
        self.ref_pub.publish(String(data='ee_ref'))
        self.get_logger().info("Requested reference frame 'ee_ref'")


def main():
    rclpy.init()
    node = TestPublisher()
    # publish several times to ensure subscribers receive it
    for _ in range(5):
        node.run_once()
        time.sleep(0.2)
    node.get_logger().info('Done publishing test transform. Check RViz /ee_reference_marker and teleop reference.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
