#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, PoseStamped, Point, Quaternion
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Return (x, y, z, w) quaternion from roll, pitch, yaw (radians)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


class ReferenceFrameBroadcaster(Node):
    """Broadcasts a dynamic reference frame 'ee_ref' relative to 'world'.

    - Subscribe to '/ee_reference_pose' (geometry_msgs/PoseStamped) to set the frame pose.
    - Subscribe to '/ee_reference_reset' (std_msgs/Empty) to reset to identity (aligned with world).
    - Publishes TF from 'world' -> 'ee_ref' at 10Hz.
    - Publishes a Marker on '/ee_reference_marker' for RViz visualization.
    """

    def __init__(self):
        super().__init__('reference_frame_broadcaster')
        self.br = TransformBroadcaster(self)
        self.pose: Optional[PoseStamped] = None
        self.timer = self.create_timer(0.1, self._on_timer)
        self.create_subscription(PoseStamped, '/ee_reference_pose', self._on_pose, 10)
        self.create_subscription(Empty, '/ee_reference_reset', self._on_reset, 10)
        self.marker_pub = self.create_publisher(Marker, '/ee_reference_marker', 10)
        # default: identity at origin
        self.pose = PoseStamped()
        self.pose.header.frame_id = 'world'
        self.pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
        q = quaternion_from_euler(0.0, 0.0, 0.0)
        self.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.get_logger().info('Reference frame broadcaster ready (frame: ee_ref)')

    def _on_pose(self, msg: PoseStamped):
        # Accept pose in any frame; we'll broadcast relative to msg.header.frame_id
        self.pose = msg
        self.get_logger().info(f'Set ee_ref <- {msg.header.frame_id} pose ({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})')

    def _on_reset(self, msg: Empty):
        self.pose = PoseStamped()
        self.pose.header.frame_id = 'world'
        self.pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
        q = quaternion_from_euler(0.0, 0.0, 0.0)
        self.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.get_logger().info('Reset ee_ref -> identity (world)')

    def _on_timer(self):
        if self.pose is None:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        # parent frame is the pose header frame (usually 'world')
        t.header.frame_id = self.pose.header.frame_id or 'world'
        t.child_frame_id = 'ee_ref'
        p = self.pose.pose.position
        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = p.z
        q = self.pose.pose.orientation
        t.transform.rotation = q
        self.br.sendTransform(t)

        # Publish a single dynamic marker in the world frame that shows the ee_ref orientation
        # Only publish if pose was provided in 'world' for simplicity
        try:
            if (self.pose.header.frame_id or 'world') == 'world':
                world_marker = Marker()
                world_marker.header.frame_id = 'world'
                world_marker.header.stamp = t.header.stamp
                world_marker.ns = 'ee_ref_world'
                world_marker.id = 10
                world_marker.type = Marker.ARROW
                world_marker.action = Marker.ADD
                # place arrow at the ee_ref position with orientation matching ee_ref
                world_marker.pose = self.pose.pose
                world_marker.scale.x = 0.20
                world_marker.scale.y = 0.03
                world_marker.scale.z = 0.05
                world_marker.color.r = 0.9
                world_marker.color.g = 0.1
                world_marker.color.b = 0.6
                world_marker.color.a = 0.95
                self.marker_pub.publish(world_marker)

                # text label in world frame above the arrow
                text = Marker()
                text.header = world_marker.header
                text.ns = 'ee_ref_world'
                text.id = 11
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.position.x = self.pose.pose.position.x
                text.pose.position.y = self.pose.pose.position.y
                text.pose.position.z = self.pose.pose.position.z + 0.12
                text.scale.z = 0.05
                text.color.r = 1.0
                text.color.g = 1.0
                text.color.b = 1.0
                text.color.a = 1.0
                text.text = 'ee_ref'
                self.marker_pub.publish(text)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ReferenceFrameBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
