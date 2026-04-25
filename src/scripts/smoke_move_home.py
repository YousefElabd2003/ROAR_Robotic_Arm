#!/usr/bin/env python3
import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint

HOME_JOINTS = {
    'joint_0': 0.0,
    'joint_1': 0.0,
    'joint_2': 0.0,
    'joint_3': 0.0,
    'joint_4': 0.0,
    'joint_5': 0.0,
}

class SmokeMover(Node):
    def __init__(self):
        super().__init__('smoke_move_home')
        self.client = ActionClient(self, MoveGroup, 'move_action')

    def move_home(self):
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('move_action server unavailable')
            return False
        constraints = Constraints()
        for name, val in HOME_JOINTS.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            jc.tolerance_above = 0.03
            jc.tolerance_below = 0.03
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal = MoveGroup.Goal()
        goal.request.group_name = 'arm_controller'
        goal.request.allowed_planning_time = 5.0
        goal.request.num_planning_attempts = 10
        goal.request.goal_constraints.append(constraints)
        fut = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if not fut.done():
            self.get_logger().error('Goal send timeout')
            return False
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('Goal rejected')
            return False
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=20.0)
        if not res_fut.done() or res_fut.result() is None:
            self.get_logger().error('Result timeout')
            return False
        code = int(res_fut.result().result.error_code.val)
        self.get_logger().info(f'Move finished with code {code}')
        return code == 1


def main():
    rclpy.init()
    node = SmokeMover()
    ok = node.move_home()
    node.destroy_node()
    rclpy.shutdown()
    print('SMOKE TEST OK' if ok else 'SMOKE TEST FAILED')

if __name__ == '__main__':
    main()
