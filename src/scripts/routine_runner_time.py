#!/usr/bin/env python3

import re
import threading

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, RobotState
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from builtin_interfaces.msg import Duration as BuiltinDuration

GROUP_NAME = "arm_controller"
FRAME_CANDIDATES = ["world", "base_link", "base_footprint", "map", "odom"]
EE_LINK_CANDIDATES = ["link_6", "left_gripper_link", "right_gripper_link", "ee_link", "tool0"]
FRAME_RESOLVE_TIMEOUT = 30.0
WAIT_SECONDS = 5.0
ARM_JOINT_ORDER = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]

DIRECTION_MAP = {
    "w": (+1, 0, 0),
    "s": (-1, 0, 0),
    "a": (0, +1, 0),
    "d": (0, -1, 0),
    "q": (0, 0, +1),
    "e": (0, 0, -1),
}

AXIS_LABEL = {
    "w": "+X forward",
    "s": "-X backward",
    "a": "+Y left",
    "d": "-Y right",
    "q": "+Z up",
    "e": "-Z down",
}

MOTION_ALIASES = {
    "forward": "w",
    "backward": "s",
    "back": "s",
    "left": "a",
    "right": "d",
    "up": "q",
    "down": "e",
    "w": "w",
    "s": "s",
    "a": "a",
    "d": "d",
    "q": "q",
    "e": "e",
}


def parse_motion_token(token: str):
    return MOTION_ALIASES.get(token.strip().lower())


def parse_routine_input():
    while True:
        try:
            count = int(input("Number of motions: ").strip())
            if count <= 0:
                print("Count must be > 0")
                continue
        except ValueError:
            print("Enter a valid integer count")
            continue

        raw = input("Routine motions (space/comma separated, e.g. up down forward): ").strip()
        tokens = [t for t in re.split(r"[\s,]+", raw) if t]
        if len(tokens) != count:
            print(f"Expected {count} motions, got {len(tokens)}")
            continue

        motions = []
        bad = []
        for tok in tokens:
            key = parse_motion_token(tok)
            if key is None:
                bad.append(tok)
            else:
                motions.append(key)
        if bad:
            print(f"Unsupported motions: {bad}")
            print("Supported: up down left right forward backward (or w/s/a/d/q/e)")
            continue

        try:
            distance_cm = float(input("Distance for each motion (cm): ").strip())
            if distance_cm <= 0.0:
                print("Distance must be > 0")
                continue
        except ValueError:
            print("Enter a valid number for distance")
            continue

        return motions, distance_cm


class RoutineRunnerTime(Node):
    def __init__(self, motions: list[str], distance_cm: float):
        super().__init__("routine_runner_time")
        self.move_client = ActionClient(self, MoveGroup, "move_action")
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, "joint_states", self._js_cb, 10)
        self.final_joint_pub = self.create_publisher(JointState, "/final_joints_state", 10)

        self.motions = motions
        self.distance_cm = distance_cm
        self.distance_m = distance_cm / 100.0
        self.index = 0
        self.ok_count = 0
        self.fail_count = 0
        self.finished = threading.Event()
        self.wait_timer = None
        self.frame_id = None
        self.link_name = None
        self.current_joints: dict[str, float] = {}
        self.latest_joint_state = JointState()
        self._published_for_step = False

    def _js_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos
        self.latest_joint_state = msg

    def _wait_for_joint_states(self, timeout_sec: float = 5.0):
        start = self.get_clock().now()
        timeout_ns = int(timeout_sec * 1e9)
        while rclpy.ok():
            if self.current_joints:
                return True
            if (self.get_clock().now() - start).nanoseconds > timeout_ns:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)

    def _format_joint_line(self, source: dict[str, float], label: str):
        values = {j: source.get(j) for j in ARM_JOINT_ORDER}
        if all(v is None for v in values.values()):
            self.get_logger().info(f"{label}: no arm joint data")
            return

        lines = [f"{label} (deg)", "-----------------------------", "Joint      Angle"]
        for joint in ARM_JOINT_ORDER:
            val = values[joint]
            angle = "n/a" if val is None else f"{val * 57.2958:+7.2f}"
            lines.append(f"{joint:<8}   {angle}")
        self.get_logger().info("\n".join(lines))

    def _latest_velocity_map(self):
        out = {}
        if not self.latest_joint_state.name:
            return out
        vels = list(self.latest_joint_state.velocity)
        for idx, name in enumerate(self.latest_joint_state.name):
            out[name] = vels[idx] if idx < len(vels) else 0.0
        return out

    def _latest_effort_map(self):
        out = {}
        if not self.latest_joint_state.name:
            return out
        efforts = list(self.latest_joint_state.effort)
        for idx, name in enumerate(self.latest_joint_state.name):
            out[name] = efforts[idx] if idx < len(efforts) else 0.0
        return out

    def _compute_target_joints(self, tx: float, ty: float, tz: float):
        if not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("/compute_ik service unavailable")
            return None

        req = GetPositionIK.Request()
        req.ik_request.group_name = GROUP_NAME
        req.ik_request.ik_link_name = self.link_name
        req.ik_request.pose_stamped.header.frame_id = self.frame_id
        req.ik_request.pose_stamped.pose.position.x = tx
        req.ik_request.pose_stamped.pose.position.y = ty
        req.ik_request.pose_stamped.pose.position.z = tz
        req.ik_request.pose_stamped.pose.orientation.w = 1.0
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = BuiltinDuration(sec=2, nanosec=0)

        rs = RobotState()
        rs.joint_state = self.latest_joint_state
        req.ik_request.robot_state = rs

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("IK request timed out")
            return None

        resp = future.result()
        if int(resp.error_code.val) != 1:
            self.get_logger().error(f"IK failed with code {int(resp.error_code.val)}")
            return None

        return {n: p for n, p in zip(resp.solution.joint_state.name, resp.solution.joint_state.position)}

    def _publish_target_joints(self, target_joints: dict[str, float]):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id or ""
        ordered = [j for j in ARM_JOINT_ORDER if j in target_joints]
        if not ordered:
            ordered = list(target_joints.keys())
        vel_map = self._latest_velocity_map()
        effort_map = self._latest_effort_map()
        msg.name = ordered
        msg.position = [target_joints[name] for name in ordered]
        msg.velocity = [vel_map.get(name, 0.0) for name in ordered]
        msg.effort = [effort_map.get(name, 0.0) for name in ordered]
        self.final_joint_pub.publish(msg)

    def _resolve_transform_pair(self):
        start = self.get_clock().now()
        timeout_ns = int(FRAME_RESOLVE_TIMEOUT * 1e9)

        while rclpy.ok():
            now = self.get_clock().now()
            if (now - start).nanoseconds > timeout_ns:
                break

            for frame in FRAME_CANDIDATES:
                for link in EE_LINK_CANDIDATES:
                    try:
                        if self.tf_buffer.can_transform(
                            frame,
                            link,
                            rclpy.time.Time(),
                            timeout=Duration(seconds=0.3),
                        ):
                            self.frame_id = frame
                            self.link_name = link
                            self.get_logger().info(
                                f"Using TF/planning frame '{self.frame_id}' and EE link '{self.link_name}'"
                            )
                            return True
                    except Exception:
                        continue

            rclpy.spin_once(self, timeout_sec=0.1)

        try:
            frames_yaml = self.tf_buffer.all_frames_as_yaml()
            if not frames_yaml.strip():
                frames_yaml = "<no TF frames available>"
        except Exception:
            frames_yaml = "<unable to query TF frame list>"

        self.get_logger().error(
            "Could not resolve frame/link pair. Tried frames: "
            + ", ".join(FRAME_CANDIDATES)
            + " | links: "
            + ", ".join(EE_LINK_CANDIDATES)
        )
        self.get_logger().error("TF snapshot: " + frames_yaml[:800])
        return False

    def _tf_ee(self):
        if self.frame_id is None or self.link_name is None:
            return None
        try:
            t = self.tf_buffer.lookup_transform(
                self.frame_id,
                self.link_name,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )
            p = t.transform.translation
            return p.x, p.y, p.z
        except Exception as exc:
            self.get_logger().error(f"TF lookup failed: {exc}")
            return None

    def _pos_constraint(self, x: float, y: float, z: float, tol: float = 0.01):
        pc = PositionConstraint()
        pc.header.frame_id = self.frame_id
        pc.link_name = self.link_name
        pc.weight = 1.0

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [tol, tol, tol]
        pc.constraint_region.primitives.append(box)

        target = PoseStamped()
        target.header.frame_id = self.frame_id
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        target.pose.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(target.pose)
        return pc

    def start(self):
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("move_action server unavailable")
            self.finished.set()
            return
        if not self._wait_for_joint_states(timeout_sec=5.0):
            self.get_logger().error("No /joint_states received")
            self.finished.set()
            return
        if not self._resolve_transform_pair():
            self.finished.set()
            return
        self.get_logger().info(
            f"Starting time-based routine: steps={len(self.motions)} distance={self.distance_cm:.2f} cm wait={WAIT_SECONDS:.0f}s"
        )
        self._send_next_move()

    def _send_next_move(self):
        if self.index >= len(self.motions):
            self.get_logger().info(f"Routine complete: OK={self.ok_count} FAIL={self.fail_count}")
            self.finished.set()
            return

        direction = self.motions[self.index]
        self._published_for_step = False
        axis = DIRECTION_MAP[direction]
        ee = self._tf_ee()
        if ee is None:
            self.fail_count += 1
            self.index += 1
            self._schedule_next_after_wait()
            return

        tx = ee[0] + axis[0] * self.distance_m
        ty = ee[1] + axis[1] * self.distance_m
        tz = ee[2] + axis[2] * self.distance_m

        self._format_joint_line(self.current_joints, "CURRENT")
        target_joints = self._compute_target_joints(tx, ty, tz)
        if target_joints is None:
            self.get_logger().warn("IK preview unavailable; executing move using teleop-style planning")
            self._publish_target_joints(self.current_joints)
        else:
            self._format_joint_line(target_joints, "GOAL")
            self._publish_target_joints(target_joints)
        self._published_for_step = True
        self.get_logger().info(f"Step {self.index + 1}/{len(self.motions)} | {AXIS_LABEL[direction]} {self.distance_cm:.2f} cm")

        c = Constraints()
        c.position_constraints.append(self._pos_constraint(tx, ty, tz, tol=0.01))

        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.allowed_planning_time = 3.0
        goal.request.num_planning_attempts = 10
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3
        goal.request.goal_constraints.append(c)

        fut = self.move_client.send_goal_async(goal)
        fut.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(f"Step {self.index + 1}: goal rejected")
            self.fail_count += 1
            self.index += 1
            self._schedule_next_after_wait()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        code = int(future.result().result.error_code.val)
        if code == 1:
            self.ok_count += 1
            self._format_joint_line(self.current_joints, "REACHED")
            self.get_logger().info(f"Step {self.index + 1}: reached")
        else:
            self.fail_count += 1
            self.get_logger().warn(f"Step {self.index + 1}: failed (code {code})")

        self.index += 1
        self._schedule_next_after_wait()

    def _schedule_next_after_wait(self):
        self.get_logger().info(f"Waiting {WAIT_SECONDS:.0f}s before next move")

        if self.wait_timer is not None:
            self.wait_timer.cancel()
            self.wait_timer = None

        def _timer_cb():
            if self.wait_timer is not None:
                self.wait_timer.cancel()
                self.wait_timer = None
            self._send_next_move()

        self.wait_timer = self.create_timer(WAIT_SECONDS, _timer_cb)


def main():
    motions, distance_cm = parse_routine_input()

    rclpy.init()
    node = RoutineRunnerTime(motions=motions, distance_cm=distance_cm)
    node.start()

    try:
        while rclpy.ok() and not node.finished.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node.wait_timer is not None:
            node.wait_timer.cancel()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()