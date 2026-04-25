#!/usr/bin/env python3

from __future__ import annotations

from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity

import rclpy


class CollisionGuard:
    def __init__(self, node, group_name: str, service_name: str = "/check_state_validity"):
        self.node = node
        self.group_name = group_name
        self.client = node.create_client(GetStateValidity, service_name)

    def check_current_state(self, joints: dict[str, float], timeout_sec: float = 0.5) -> tuple[bool, str]:
        if not joints:
            return False, "No joint state available for collision check."

        if not self.client.wait_for_service(timeout_sec=timeout_sec):
            return False, "Collision service /check_state_validity unavailable."

        req = GetStateValidity.Request()
        req.group_name = self.group_name

        rs = RobotState()
        names = sorted(joints.keys())
        rs.joint_state.name = names
        rs.joint_state.position = [float(joints[name]) for name in names]
        req.robot_state = rs

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)

        if not future.done() or future.result() is None:
            return False, "Collision check timed out."

        result = future.result()
        if result.valid:
            return True, ""

        if result.contacts:
            contact = result.contacts[0]
            return False, f"Collision: {contact.contact_body_1} vs {contact.contact_body_2}"
        return False, "Current state is in collision."
