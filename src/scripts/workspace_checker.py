#!/usr/bin/env python3

"""
workspace_checker.py — Reports how far the EE can move from its current position
---------------------------------------------------------------------------------
Uses the current sixdof URDF chain and MoveIt FK to report:
  - Total reachable workspace bounds
  - Remaining teleop rotation headroom
  - Planning-validated translation headroom from the current EE pose

Also publishes RViz markers showing the current EE pose and validated reach in
the +/-X, +/-Y, and +/-Z directions.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, RobotState
from moveit_msgs.srv import GetPositionFK
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from collision_guard import CollisionGuard

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


DEFAULT_JOINT_LIMITS = {
    "joint_0": (-0.0170, 3.1590),
    "joint_1": (-2.6350, 0.0170),
    "joint_2": (-0.0170, 3.1590),
    "joint_3": (-1.6581, 1.6581),
    "joint_4": (-1.6581, 1.6581),
    "joint_5": (-1.6581, 1.6581),
}
DEFAULT_JOINT_NAMES = list(DEFAULT_JOINT_LIMITS.keys())
EE_LINK = "link_6"
BASE_FRAME = "base_link"
WORLD_FRAME = "world"
GROUP_NAME = "arm_controller"
LOCK_TOL = 0.05
MARKER_TOPIC = "/workspace_checker_markers"


def resolve_urdf_path() -> str | None:
    if get_package_share_directory is not None:
        try:
            return os.path.join(get_package_share_directory("sixdof_pkg"), "urdf", "roar.urdf")
        except Exception:
            pass
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sixdof_pkg", "urdf", "roar.urdf"))


def load_chain_joint_limits_from_urdf(base_link: str, ee_link: str) -> tuple[list[str], dict[str, tuple[float, float]], str | None]:
    urdf_path = resolve_urdf_path()
    if not urdf_path or not os.path.exists(urdf_path):
        return DEFAULT_JOINT_NAMES, DEFAULT_JOINT_LIMITS.copy(), None

    try:
        xml_text = subprocess.check_output(["xacro", urdf_path], text=True)
        root = ET.fromstring(xml_text)
    except Exception:
        return DEFAULT_JOINT_NAMES, DEFAULT_JOINT_LIMITS.copy(), urdf_path

    joints_by_child = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link"):
            joints_by_child[child.get("link")] = joint

    chain = []
    link = ee_link
    while link != base_link and link in joints_by_child:
        joint = joints_by_child[link]
        chain.append(joint)
        parent = joint.find("parent")
        if parent is None or not parent.get("link"):
            break
        link = parent.get("link")

    chain.reverse()
    joint_names = []
    joint_limits: dict[str, tuple[float, float]] = {}
    for joint in chain:
        j_type = (joint.get("type") or "").lower()
        if j_type not in ("revolute", "prismatic", "continuous"):
            continue

        j_name = joint.get("name")
        if not j_name:
            continue

        if j_type == "continuous":
            lo, hi = -math.pi, math.pi
        else:
            limit = joint.find("limit")
            if limit is None:
                continue
            lo = float(limit.get("lower", "0.0"))
            hi = float(limit.get("upper", "0.0"))

        joint_names.append(j_name)
        joint_limits[j_name] = (lo, hi)

    if len(joint_names) < 3:
        return DEFAULT_JOINT_NAMES, DEFAULT_JOINT_LIMITS.copy(), urdf_path
    return joint_names, joint_limits, urdf_path


class WorkspaceChecker(Node):
    def __init__(self):
        super().__init__("workspace_checker")
        self.joint_names, self.joint_limits, self.urdf_path = load_chain_joint_limits_from_urdf(BASE_FRAME, EE_LINK)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.move_client = ActionClient(self, MoveGroup, "move_action")
        self.current_joints: dict[str, float] = {}
        self.js_received = False
        self.collision_guard = CollisionGuard(self, GROUP_NAME)
        self.cached_markers: MarkerArray | None = None

        marker_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_pub = self.create_publisher(MarkerArray, MARKER_TOPIC, marker_qos)
        self.create_timer(2.0, self._republish_markers)
        self.create_subscription(JointState, "joint_states", self._js_cb, 10)

    def _republish_markers(self):
        if self.cached_markers is not None:
            self.marker_pub.publish(self.cached_markers)

    def _js_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos
        self.js_received = True

    def wait_for_joints(self, timeout=5.0):
        import time

        start = time.time()
        while not self.js_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout:
                return False
        return True

    def compute_fk_batch(self, configs: list[list[float]]) -> list[tuple[float, float, float]]:
        if not self.fk_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/compute_fk service not available")
            return []

        positions = []
        for config in configs:
            req = GetPositionFK.Request()
            req.header.frame_id = WORLD_FRAME
            req.fk_link_names = [EE_LINK]

            robot_state = RobotState()
            robot_state.joint_state.name = self.joint_names
            robot_state.joint_state.position = config
            req.robot_state = robot_state

            future = self.fk_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

            if future.done() and future.result() is not None:
                result = future.result()
                if result.error_code.val == 1 and result.pose_stamped:
                    pose = result.pose_stamped[0].pose.position
                    positions.append((pose.x, pose.y, pose.z))

        return positions

    def get_current_ee_pos(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                WORLD_FRAME,
                EE_LINK,
                rclpy.time.Time(),
                timeout=Duration(seconds=2.0),
            )
            point = transform.transform.translation
            return (point.x, point.y, point.z)
        except Exception as exc:
            self.get_logger().error(f"TF failed: {exc}")
            return None

    def _pos_constraint(self, x: float, y: float, z: float, tol: float = 0.01):
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = WORLD_FRAME
        position_constraint.link_name = EE_LINK
        position_constraint.weight = 1.0

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [tol, tol, tol]
        position_constraint.constraint_region.primitives.append(box)

        pose = PoseStamped()
        pose.header.frame_id = WORLD_FRAME
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        position_constraint.constraint_region.primitive_poses.append(pose.pose)
        return position_constraint

    def _lock_constraints(self, lock: bool):
        if not lock:
            return []
        constraints = []
        for joint_name in ["joint_4", "joint_5"]:
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = self.current_joints.get(joint_name, 0.0)
            joint_constraint.tolerance_above = LOCK_TOL
            joint_constraint.tolerance_below = LOCK_TOL
            joint_constraint.weight = 1.0
            constraints.append(joint_constraint)
        return constraints

    def _plan_only_reachable(self, tx: float, ty: float, tz: float, lock: bool) -> bool:
        if not self.move_client.wait_for_server(timeout_sec=2.0):
            return False

        ok, _ = self.collision_guard.check_current_state(self.current_joints)
        if not ok:
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.allowed_planning_time = 1.5
        goal.request.num_planning_attempts = 5
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        constraints = Constraints()
        constraints.position_constraints.append(self._pos_constraint(tx, ty, tz, tol=0.01))
        constraints.joint_constraints.extend(self._lock_constraints(lock))
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = True

        goal_future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=3.0)
        if not goal_future.done() or goal_future.result() is None:
            return False

        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)
        if not result_future.done() or result_future.result() is None:
            return False
        return result_future.result().result.error_code.val == 1

    def _max_plan_valid_translation(self, cx: float, cy: float, cz: float, axis: str, sign: float, fk_cap_m: float, lock: bool) -> float:
        if fk_cap_m <= 0.0:
            return 0.0

        def target_at(distance):
            if axis == "x":
                return (cx + sign * distance, cy, cz)
            if axis == "y":
                return (cx, cy + sign * distance, cz)
            return (cx, cy, cz + sign * distance)

        low, high = 0.0, fk_cap_m
        tiny = min(0.01, fk_cap_m)
        tx, ty, tz = target_at(tiny)
        if not self._plan_only_reachable(tx, ty, tz, lock):
            return 0.0

        tx, ty, tz = target_at(high)
        if self._plan_only_reachable(tx, ty, tz, lock):
            return high

        for _ in range(8):
            mid = 0.5 * (low + high)
            tx, ty, tz = target_at(mid)
            if self._plan_only_reachable(tx, ty, tz, lock):
                low = mid
            else:
                high = mid
        return low

    def _build_summary_markers(self, current_pos, distances: dict[str, float]) -> MarkerArray:
        cx, cy, cz = current_pos
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        ee_marker = Marker()
        ee_marker.header.frame_id = WORLD_FRAME
        ee_marker.header.stamp = now
        ee_marker.ns = "workspace_checker"
        ee_marker.id = 0
        ee_marker.type = Marker.SPHERE
        ee_marker.action = Marker.ADD
        ee_marker.pose.position.x = cx
        ee_marker.pose.position.y = cy
        ee_marker.pose.position.z = cz
        ee_marker.pose.orientation.w = 1.0
        ee_marker.scale.x = 0.035
        ee_marker.scale.y = 0.035
        ee_marker.scale.z = 0.035
        ee_marker.color.r = 1.0
        ee_marker.color.g = 0.9
        ee_marker.color.b = 0.1
        ee_marker.color.a = 1.0
        markers.markers.append(ee_marker)

        axis_specs = [
            ("+X", (1.0, 0.2, 0.2), (1.0, 0.0, 0.0)),
            ("-X", (0.6, 0.2, 0.2), (-1.0, 0.0, 0.0)),
            ("+Y", (0.2, 1.0, 0.2), (0.0, 1.0, 0.0)),
            ("-Y", (0.2, 0.6, 0.2), (0.0, -1.0, 0.0)),
            ("+Z", (0.2, 0.5, 1.0), (0.0, 0.0, 1.0)),
            ("-Z", (0.2, 0.2, 0.7), (0.0, 0.0, -1.0)),
        ]

        marker_id = 10
        for label, color, direction in axis_specs:
            dist = distances.get(label, 0.0)

            arrow = Marker()
            arrow.header.frame_id = WORLD_FRAME
            arrow.header.stamp = now
            arrow.ns = "workspace_checker"
            arrow.id = marker_id
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.008
            arrow.scale.y = 0.015
            arrow.scale.z = 0.02
            arrow.color.r = color[0]
            arrow.color.g = color[1]
            arrow.color.b = color[2]
            arrow.color.a = 0.95

            start = Point()
            start.x = cx
            start.y = cy
            start.z = cz
            end = Point()
            end.x = cx + direction[0] * dist
            end.y = cy + direction[1] * dist
            end.z = cz + direction[2] * dist
            arrow.points = [start, end]
            markers.markers.append(arrow)
            marker_id += 1

            text = Marker()
            text.header.frame_id = WORLD_FRAME
            text.header.stamp = now
            text.ns = "workspace_checker"
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = end.x
            text.pose.position.y = end.y
            text.pose.position.z = end.z + 0.03
            text.pose.orientation.w = 1.0
            text.scale.z = 0.025
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.95
            text.text = f"{label} {dist * 100:.1f} cm"
            markers.markers.append(text)
            marker_id += 1

        return markers

    def run(self, n_samples: int, lock=False, minimal=False):
        if not minimal:
            print("\n" + "═" * 60)
            print("  SixDOF Arm Workspace Checker")
            print("═" * 60)
            print(f"  URDF: {self.urdf_path or 'fallback defaults'}")
            print(f"  Chain joints: {', '.join(self.joint_names)}")

        if not minimal:
            print("\n  Waiting for joint states...")
        if not self.wait_for_joints():
            print("  ERROR: No joint states received. Is the robot running?")
            return

        if not minimal:
            print("  Getting current EE position...")
        rclpy.spin_once(self, timeout_sec=1.0)
        current_pos = self.get_current_ee_pos()

        if current_pos:
            cx, cy, cz = current_pos
            if not minimal:
                print("\n  Current EE position (world frame):")
                print(f"    x = {cx:.4f} m")
                print(f"    y = {cy:.4f} m")
                print(f"    z = {cz:.4f} m")
        else:
            cx = cy = cz = 0.0
            print("  WARNING: Could not get current EE position")

        print("\n" + "─" * 60)
        print("  KEY RESULT 1: ROTATION HEADROOM")
        print("─" * 60)
        if not minimal:
            print("  Joint state + remaining rotation:")
            for joint_name in self.joint_names:
                value = self.current_joints.get(joint_name, 0.0)
                lo, hi = self.joint_limits[joint_name]
                pct_lo = 100 * (value - lo) / (hi - lo) if hi != lo else 0
                room_neg = math.degrees(value - lo)
                room_pos = math.degrees(hi - value)
                print(
                    f"    {joint_name}: {value:+.4f} rad ({math.degrees(value):+.1f}°)  "
                    f"[{math.degrees(lo):.0f}° to {math.degrees(hi):.0f}°]  "
                    f"at {pct_lo:.0f}% | remaining: -{room_neg:.1f}° / +{room_pos:.1f}°"
                )

        print("  Teleop rotation remaining (your controls):")
        for axis, joint_name in [("rz", "joint_0"), ("ry", "joint_1"), ("rx", "joint_2")]:
            value = self.current_joints.get(joint_name, 0.0)
            lo, hi = self.joint_limits[joint_name]
            print(f"    {axis}: {joint_name} -> -{math.degrees(value - lo):.1f}° / +{math.degrees(hi - value):.1f}°")

        if not minimal:
            print(f"\n  Sampling {n_samples} random joint configurations...")
        np.random.seed(42)
        configs = []
        current_cfg = [self.current_joints.get(joint_name, 0.0) for joint_name in self.joint_names]
        configs.append(current_cfg)
        if lock:
            if not minimal:
                print("  Orientation lock enabled: joint_4 and joint_5 fixed to current values.")
            j4_val = self.current_joints.get("joint_4", 0.0)
            j5_val = self.current_joints.get("joint_5", 0.0)
            for _ in range(max(0, n_samples - 1)):
                cfg = []
                for joint_name in self.joint_names:
                    if joint_name == "joint_4":
                        cfg.append(j4_val)
                    elif joint_name == "joint_5":
                        cfg.append(j5_val)
                    else:
                        lo, hi = self.joint_limits[joint_name]
                        cfg.append(np.random.uniform(lo, hi))
                configs.append(cfg)
        else:
            for _ in range(max(0, n_samples - 1)):
                configs.append([np.random.uniform(lo, hi) for lo, hi in self.joint_limits.values()])

        if not minimal:
            print(f"  Running FK for {len(configs)} configurations...")
        positions = self.compute_fk_batch(configs)

        if not positions:
            print("  ERROR: FK computation returned no results.")
            return

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]

        if not minimal:
            print(f"\n  Successfully computed {len(positions)} FK samples")
            print("\n" + "─" * 60)
            print("  REFERENCE: TOTAL REACHABLE WORKSPACE (world frame):")
            print("─" * 60)
            print(f"    X: {min(xs):.4f} to {max(xs):.4f} m  (span: {max(xs) - min(xs):.4f} m)")
            print(f"    Y: {min(ys):.4f} to {max(ys):.4f} m  (span: {max(ys) - min(ys):.4f} m)")
            print(f"    Z: {min(zs):.4f} to {max(zs):.4f} m  (span: {max(zs) - min(zs):.4f} m)")

        if current_pos:
            print("\n" + "─" * 60)
            mode_label = "(J4/J5 LOCKED)" if lock else "(ALL JOINTS FREE)"
            print(
                f"  KEY RESULT 2: REACHABLE TRANSLATION FROM CURRENT EE POSITION {mode_label} "
                f"({cx:.3f}, {cy:.3f}, {cz:.3f}):"
            )
            print("─" * 60)

            dxs = [x - cx for x, _, _ in positions]
            dys = [y - cy for _, y, _ in positions]
            dzs = [z - cz for _, _, z in positions]
            caps = {
                "+X": max(0.0, max(dxs)),
                "-X": max(0.0, -min(dxs)),
                "+Y": max(0.0, max(dys)),
                "-Y": max(0.0, -min(dys)),
                "+Z": max(0.0, max(dzs)),
                "-Z": max(0.0, -min(dzs)),
            }

            print("  (planning-validated, plan-only checks)")
            p_x = self._max_plan_valid_translation(cx, cy, cz, "x", +1.0, caps["+X"], lock)
            n_x = self._max_plan_valid_translation(cx, cy, cz, "x", -1.0, caps["-X"], lock)
            p_y = self._max_plan_valid_translation(cx, cy, cz, "y", +1.0, caps["+Y"], lock)
            n_y = self._max_plan_valid_translation(cx, cy, cz, "y", -1.0, caps["-Y"], lock)
            p_z = self._max_plan_valid_translation(cx, cy, cz, "z", +1.0, caps["+Z"], lock)
            n_z = self._max_plan_valid_translation(cx, cy, cz, "z", -1.0, caps["-Z"], lock)

            print(f"    +X (forward): up to {p_x * 100:.1f} cm")
            print(f"    -X (back):    up to {n_x * 100:.1f} cm")
            print(f"    +Y (left):    up to {p_y * 100:.1f} cm")
            print(f"    -Y (right):   up to {n_y * 100:.1f} cm")
            print(f"    +Z (up):      up to {p_z * 100:.1f} cm")
            print(f"    -Z (down):    up to {n_z * 100:.1f} cm")

            self.cached_markers = self._build_summary_markers(
                current_pos,
                {
                    "+X": p_x,
                    "-X": n_x,
                    "+Y": p_y,
                    "-Y": n_y,
                    "+Z": p_z,
                    "-Z": n_z,
                },
            )
            self.marker_pub.publish(self.cached_markers)
            if not minimal:
                print(f"\n  Published RViz markers on {MARKER_TOPIC}")
                print("  Add MarkerArray by topic in RViz to see current EE and directional reach.")

        if not minimal:
            print("═" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000, help="Number of FK samples (default: 2000)")
    parser.add_argument("--lock", action="store_true", help="Lock joint_4 and joint_5 to current values")
    parser.add_argument("--minimal", action="store_true", help="Show only key numeric results")
    args = parser.parse_args()

    rclpy.init()
    node = WorkspaceChecker()
    try:
        node.run(args.samples, lock=args.lock, minimal=args.minimal)
        if node.cached_markers is not None:
            print("  Workspace checker markers will stay live in RViz. Ctrl+C to stop.\n")
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
