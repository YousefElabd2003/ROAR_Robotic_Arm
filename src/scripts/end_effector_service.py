#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import TransformStamped, PoseStamped
from std_srvs.srv import Trigger
import tf_transformations
from tf2_ros import StaticTransformBroadcaster
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState

LINK_NAME = "link_6"
GROUP_NAME = "arm_controller"

class EndEffectorService(Node):
    def __init__(self):
        super().__init__('end_effector_service')
        self.br = StaticTransformBroadcaster(self)
        self.move_client = ActionClient(self, MoveGroup, "move_action")
        self.current_joints = {}
        self.js_received = False
        self.create_subscription(JointState, "joint_states", self._js_cb, 10)

        # Services
        self.create_service(Trigger, 'set_sampling', self.handle_sampling)
        self.create_service(Trigger, 'set_probing', self.handle_probing)
        self.create_service(Trigger, 'set_gripping', self.handle_gripping)
        self.create_service(Trigger, 'set_maintenance', self.handle_maintenance)

    def _js_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos
        self.js_received = True

    def wait_for_joint_states(self, timeout: float = 5.0) -> bool:
        import time
        t0 = time.time()
        while not self.js_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - t0 > timeout:
                return False
        return True

    def wait_for_move_server(self, timeout: float = 5.0) -> bool:
        return self.move_client.wait_for_server(timeout_sec=timeout)

    def broadcast_tip(self, tool_name, xyz, rpy):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "end_effector_frame"
        t.child_frame_id = f"{tool_name}_tip"

        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
        q = tf_transformations.quaternion_from_euler(*rpy)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q

        self.br.sendTransform([t])
        self.get_logger().info(f"Broadcasted {tool_name}_tip under end_effector_frame")

    def execute_motion(self, tool_name):
        c = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = f"{tool_name}_tip"
        pc.link_name = LINK_NAME
        pc.weight = 1.0
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 0.01, 0.01]
        pc.constraint_region.primitives.append(box)
        pose = PoseStamped()
        pose.header.frame_id = f"{tool_name}_tip"
        pose.pose.position.x = 0.1
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.2
        pose.pose.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(pose.pose)
        c.position_constraints.append(pc)

        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.allowed_planning_time = 3.0
        goal.request.num_planning_attempts = 5
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3
        goal.request.goal_constraints.append(c)

        self.move_client.wait_for_server()
        goal_future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=5.0)
        if not goal_future.done() or goal_future.result() is None:
            self.get_logger().error("Goal not accepted")
            return False

        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("No result from MoveGroup")
            return False

        code = int(result_future.result().result.error_code.val)
        self.get_logger().info(f"Motion result code={code}")
        return code == 1

    # Service handlers
    def handle_sampling(self, request, response):
        self.broadcast_tip("sampling", (0.0, 0.0, 0.20), (0.0, 0.0, 0.0))
        ok = self.execute_motion("sampling")
        response.success = ok
        response.message = "Sampling executed" if ok else "Sampling failed"
        return response

    def handle_probing(self, request, response):
        self.broadcast_tip("probing", (0.0, 0.0, 0.30), (0.0, 0.0, math.pi/2))
        ok = self.execute_motion("probing")
        response.success = ok
        response.message = "Probing executed" if ok else "Probing failed"
        return response

    def handle_gripping(self, request, response):
        self.broadcast_tip("gripping", (0.05, 0.0, 0.10), (0.0, -math.pi/4, 0.0))
        ok = self.execute_motion("gripping")
        response.success = ok
        response.message = "Gripping executed" if ok else "Gripping failed"
        return response

    def handle_maintenance(self, request, response):
        self.broadcast_tip("maintenance", (0.0, 0.05, 0.15), (math.pi/6, 0.0, 0.0))
        ok = self.execute_motion("maintenance")
        response.success = ok
        response.message = "Maintenance executed" if ok else "Maintenance failed"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = EndEffectorService()
    if not node.wait_for_joint_states():
        node.get_logger().error("No joint states received")
    if not node.wait_for_move_server():
        node.get_logger().error("Move action server unavailable")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
