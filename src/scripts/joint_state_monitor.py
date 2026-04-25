#!/usr/bin/env python3

import argparse
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM_JOINT_ORDER = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]


class JointStateMonitor(Node):
    def __init__(self, hz: float):
        super().__init__("joint_state_monitor")
        self.joints: dict[str, float] = {}
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        period = 1.0 / hz if hz > 0 else 1.0
        self.create_timer(period, self._print_once)
        self.get_logger().info(f"Monitoring /joint_states at {hz:.2f} Hz")

    def _js_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.joints[name] = pos

    def _print_once(self):
        names = [n for n in ARM_JOINT_ORDER if n in self.joints]
        if not names:
            self.get_logger().info("No arm joints received yet")
            return
        parts = []
        for name in names:
            rad = self.joints[name]
            deg = math.degrees(rad)
            parts.append(f"{name}={rad:+.4f} rad ({deg:+.1f} deg)")
        self.get_logger().info(" | ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Low-frequency joint state monitor")
    parser.add_argument("--hz", type=float, default=1.0, help="Print frequency (default: 1.0 Hz)")
    args = parser.parse_args()

    hz = args.hz if args.hz > 0 else 1.0

    rclpy.init()
    node = JointStateMonitor(hz=hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
