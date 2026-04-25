#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

LEFT_GRIPPER = 'left_gripper'
RIGHT_GRIPPER = 'right_gripper'
RAW_TOPIC = '/joint_states'
CORRECTED_TOPIC = '/joint_states_corrected'


class GripperJointStateRepublisher(Node):
    def __init__(self):
        super().__init__('gripper_joint_state_republisher')
        self.publisher = self.create_publisher(JointState, CORRECTED_TOPIC, 10)
        self.create_subscription(JointState, RAW_TOPIC, self._on_joint_state, 10)
        self.get_logger().info(
            f'Republishing {RAW_TOPIC} -> {CORRECTED_TOPIC} with mirrored gripper state from {RIGHT_GRIPPER}'
        )

    def _on_joint_state(self, msg: JointState):
        corrected = JointState()
        corrected.header = msg.header
        corrected.name = list(msg.name)
        corrected.position = list(msg.position)
        corrected.velocity = list(msg.velocity)
        corrected.effort = list(msg.effort)

        try:
            right_index = corrected.name.index(RIGHT_GRIPPER)
        except ValueError:
            self.publisher.publish(corrected)
            return

        right_position = corrected.position[right_index] if right_index < len(corrected.position) else 0.0
        right_velocity = corrected.velocity[right_index] if right_index < len(corrected.velocity) else 0.0
        right_effort = corrected.effort[right_index] if right_index < len(corrected.effort) else 0.0

        try:
            left_index = corrected.name.index(LEFT_GRIPPER)
        except ValueError:
            corrected.name.append(LEFT_GRIPPER)
            corrected.position.append(right_position)
            if corrected.velocity:
                corrected.velocity.append(right_velocity)
            if corrected.effort:
                corrected.effort.append(right_effort)
            self.publisher.publish(corrected)
            return

        if left_index < len(corrected.position):
            corrected.position[left_index] = right_position
        if left_index < len(corrected.velocity):
            corrected.velocity[left_index] = right_velocity
        if left_index < len(corrected.effort):
            corrected.effort[left_index] = right_effort
        self.publisher.publish(corrected)


def main():
    rclpy.init()
    node = GripperJointStateRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()