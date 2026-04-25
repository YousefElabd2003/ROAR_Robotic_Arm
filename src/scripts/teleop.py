#!/usr/bin/env python3
"""
teleop.py — GUI teleop for 6DOF pose control with gripper control.

Features:
- World-frame XYZ jogging
- Optional fixed-orientation pose solving
- Orientation presets and editable RPY targets
- Single-slider gripper control with open/close shortcuts
- No joint locking; orientation is enforced through full pose IK
"""

import argparse
import math
import threading
import time
from queue import Empty, Queue
import tkinter as tk
from tkinter import scrolledtext

import rclpy
from builtin_interfaces.msg import Duration as BuiltinDuration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, OrientationConstraint, PositionConstraint
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

LINK_NAME = "link_6"
GROUP_NAME = "arm_controller"
FRAME_ID = "world"
BASE_FRAME = "base_link"
HAND_CONTROLLER = "/hand_controller_controller/follow_joint_trajectory"
JOINT_STATE_TOPIC = "/joint_states_corrected"
DEFAULT_CM = 1.0
POSITION_TOL = 0.01
ORIENTATION_TOL = 0.20
DISPLAY_JOINTS = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
HOME_JOINTS = {
    "joint_0": 0.0,
    "joint_1": 0.0,
    "joint_2": 0.0,
    "joint_3": 0.0,
    "joint_4": 0.0,
    "joint_5": 0.0,
}
GRIPPER_MIN = 0.0
GRIPPER_MAX = 0.07
GRIPPER_OPEN_BUTTON = 0.0
GRIPPER_CLOSE_BUTTON = 0.07
GRIPPER_TRAJECTORY_SECONDS = 0.35
ORIENTATION_PRESETS_DEG = {
    "Look Forward": (180.0, 0.0, 0.0),
    "Look Down": (173.0, 0.3, -90.0),
    "Look Up": (7.0, -0.3, -90.0),
    "Look Right": (179.0, -2.0, -90.0),
    "Look Left": (179.0, 2.0, 90.0),
}


def quat_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
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


def euler_from_quat(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def joint_constraint(joint_name: str, position: float, tolerance: float) -> JointConstraint:
    constraint = JointConstraint()
    constraint.joint_name = joint_name
    constraint.position = position
    constraint.tolerance_above = tolerance
    constraint.tolerance_below = tolerance
    constraint.weight = 1.0
    return constraint


class Teleop(Node):
    def __init__(self, log_callback=None):
        super().__init__("sixdof_pose_teleop")
        self._client = ActionClient(self, MoveGroup, "move_action")
        self._hand_client = ActionClient(self, FollowJointTrajectory, HAND_CONTROLLER)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.done = threading.Event()
        self.done.set()
        self.hand_done = threading.Event()
        self.hand_done.set()
        self.joints: dict[str, float] = {}
        self.log_callback = log_callback
        self.maintain_orientation = True
        self.target_orientation = quat_from_euler(0.0, 0.0, 0.0)
        self.target_orientation_rpy_deg = [0.0, 0.0, 0.0]
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._js, 10)
        # Reference frame (can be changed at runtime)
        self.reference_frame = FRAME_ID
        self.create_subscription(String, '/reference_frame', self._on_ref_frame, 10)
        self._ref_capture_timer = None
        # Publisher to request reference frame pose changes
        self._ref_pose_pub = self.create_publisher(PoseStamped, '/ee_reference_pose', 10)

    def _log(self, level, text):
        logger = self.get_logger()
        if level == "error":
            logger.error(text)
        elif level == "warn":
            logger.warning(text)
        else:
            logger.info(text)
        if self.log_callback is not None:
            self.log_callback(level, text)

    def rotate_reference(self, axis: str, degrees: float):
        # Rotate the ee_ref frame about a world axis ('X','Y','Z') by degrees, keeping position
        # Find current ee_ref pose in world (fallback to EE link)
        transform = self._tf_transform('world', 'ee_ref')
        if transform is None:
            transform = self._tf_transform('world', LINK_NAME)
            if transform is None:
                self._log('error', 'No ee_ref or EE transform available to rotate')
                return

        tx = transform.translation.x
        ty = transform.translation.y
        tz = transform.translation.z
        qx = transform.rotation.x
        qy = transform.rotation.y
        qz = transform.rotation.z
        qw = transform.rotation.w

        # rotation quaternion about specified world axis
        a = axis.upper()
        rad = math.radians(degrees)
        if a == 'X':
            qrot = quat_from_euler(rad, 0.0, 0.0)
        elif a == 'Y':
            qrot = quat_from_euler(0.0, rad, 0.0)
        else:
            qrot = quat_from_euler(0.0, 0.0, rad)

        # q_new = qrot * q_current (apply rotation in world frame)
        def quat_mult(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            qy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            qz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return (qx, qy, qz, qw)

        q_current = (qx, qy, qz, qw)
        q_new = quat_mult(qrot, q_current)

        ps = PoseStamped()
        ps.header.frame_id = 'world'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = tx
        ps.pose.position.y = ty
        ps.pose.position.z = tz
        ps.pose.orientation.x = q_new[0]
        ps.pose.orientation.y = q_new[1]
        ps.pose.orientation.z = q_new[2]
        ps.pose.orientation.w = q_new[3]
        self._ref_pose_pub.publish(ps)
        self._log('info', f'Rotated ee_ref around world {a} by {degrees:.1f} deg')
        # done

    def _on_ref_frame(self, msg: String):
        try:
            self.set_reference_frame(msg.data)
        except Exception:
            pass

    def _js(self, msg):
        for name, position in zip(msg.name, msg.position):
            self.joints[name] = position

    def _tf_transform(self, parent, child):
        # Try direct lookup first
        try:
            if self.tf_buffer.can_transform(parent, child, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5)):
                return self.tf_buffer.lookup_transform(
                    parent,
                    child,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5),
                ).transform
        except Exception:
            pass

        # Fallback: attempt to compute parent->child via world frame if possible
        try:
            if parent == 'world' or child == 'world':
                return None
            # need T_world_parent and T_world_child
            if not self.tf_buffer.can_transform('world', parent, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5)):
                return None
            if not self.tf_buffer.can_transform('world', child, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5)):
                return None
            t_world_parent = self.tf_buffer.lookup_transform('world', parent, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
            t_world_child = self.tf_buffer.lookup_transform('world', child, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
            # compute T_parent_child = inv(T_world_parent) * T_world_child
            inv_r, inv_t = self._invert_transform(t_world_parent.transform)
            comp = self._compose_transform(inv_r, inv_t, t_world_child.transform)
            # return a Transform-like object with translation and rotation
            class SimpleTransform:
                pass
            st = SimpleTransform()
            st.translation = type('T', (), {})()
            st.rotation = type('Q', (), {})()
            st.translation.x, st.translation.y, st.translation.z = comp[1]
            st.rotation.x, st.rotation.y, st.rotation.z, st.rotation.w = comp[0]
            self._log('info', f'Used world-based TF fallback for {parent}->{child}')
            return st
        except Exception as exc:
            self._log('warn', f'Fallback TF {parent}->{child} failed: {exc}')
            return None

    def _invert_transform(self, transform):
        # transform: geometry_msgs/Transform
        q = (transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
        t = (transform.translation.x, transform.translation.y, transform.translation.z)
        # inverse rotation is conjugate
        qx, qy, qz, qw = q
        inv_q = (-qx, -qy, -qz, qw)
        # rotate -t by inv_q
        rt = self._rotate_vector(inv_q, (-t[0], -t[1], -t[2]))
        return inv_q, rt

    def _compose_transform(self, q1, t1, transform2):
        # q1: (x,y,z,w) rotation, t1: (x,y,z) translation; transform2 has rotation & translation
        q2 = (transform2.rotation.x, transform2.rotation.y, transform2.rotation.z, transform2.rotation.w)
        t2 = (transform2.translation.x, transform2.translation.y, transform2.translation.z)
        # composed rotation q = q1 * q2
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        qy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        qz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        # rotated translation: t = t1 + rotate(q1, t2)
        rt2 = self._rotate_vector(q1, t2)
        tx = t1[0] + rt2[0]
        ty = t1[1] + rt2[1]
        tz = t1[2] + rt2[2]
        return (qx, qy, qz, qw), (tx, ty, tz)

    def _rotate_vector(self, q, v):
        # rotate vector v by quaternion q
        x, y, z, w = q
        vx, vy, vz = v
        # q * v * q_conj
        # compute q*v
        ix =  w * vx + y * vz - z * vy
        iy =  w * vy + z * vx - x * vz
        iz =  w * vz + x * vy - y * vx
        iw = -x * vx - y * vy - z * vz
        # result = (qv) * q_conj
        rx = ix * w + iw * -x + iy * -z - iz * -y
        ry = iy * w + iw * -y + iz * -x - ix * -z
        rz = iz * w + iw * -z + ix * -y - iy * -x
        return (rx, ry, rz)

    def capture_current_orientation(self) -> bool:
        # Capture orientation of the end-effector expressed in the current reference frame
        transform = self._tf_transform(self.reference_frame, LINK_NAME)
        if transform is None:
            return False
        rotation = transform.rotation
        self.target_orientation = (rotation.x, rotation.y, rotation.z, rotation.w)
        roll, pitch, yaw = euler_from_quat(rotation.x, rotation.y, rotation.z, rotation.w)
        self.target_orientation_rpy_deg = [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]
        self._log(
            "info",
            "Captured current orientation -> "
            f"roll={self.target_orientation_rpy_deg[0]:.1f} deg, "
            f"pitch={self.target_orientation_rpy_deg[1]:.1f} deg, "
            f"yaw={self.target_orientation_rpy_deg[2]:.1f} deg",
        )
        return True

    def set_orientation_from_rpy_deg(self, roll_deg: float, pitch_deg: float, yaw_deg: float):
        self.target_orientation_rpy_deg = [roll_deg, pitch_deg, yaw_deg]
        self.target_orientation = quat_from_euler(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg),
        )
        self._log(
            "info",
            f"Orientation target set -> roll={roll_deg:.1f} deg, pitch={pitch_deg:.1f} deg, yaw={yaw_deg:.1f} deg",
        )

    def set_maintain_orientation(self, enabled: bool):
        self.maintain_orientation = enabled
        self._log("info", f"Maintain orientation: {'ON' if enabled else 'OFF'}")

    def set_reference_frame(self, frame: str):
        self.reference_frame = frame
        self._log('info', f'Set reference frame -> {frame}')
        # Try immediate capture; if unavailable, schedule a short retry
        ok = False
        try:
            ok = self.capture_current_orientation()
        except Exception:
            ok = False
        if not ok:
            if self._ref_capture_timer is not None:
                try:
                    self._ref_capture_timer.cancel()
                except Exception:
                    pass
            def _retry():
                try:
                    if self.capture_current_orientation():
                        self._log('info', 'Captured orientation after retry')
                except Exception:
                    pass
            self._ref_capture_timer = threading.Timer(0.5, _retry)
            self._ref_capture_timer.daemon = True
            self._ref_capture_timer.start()

    def status_text(self):
        mode = "fixed orientation" if self.maintain_orientation else "position only"
        lines = [f"Mode: {mode}", "Joints:"]
        for name in DISPLAY_JOINTS:
            value = self.joints.get(name, float("nan"))
            lines.append(f"  {name}  {value:+.4f} rad  ({math.degrees(value):+.1f} deg)")
        for name in ["left_gripper", "right_gripper"]:
            if name in self.joints:
                lines.append(f"  {name}  {self.joints[name]:+.4f} m")
        lines.append(
            "Target orientation: "
            f"roll={self.target_orientation_rpy_deg[0]:.1f} deg, "
            f"pitch={self.target_orientation_rpy_deg[1]:.1f} deg, "
            f"yaw={self.target_orientation_rpy_deg[2]:.1f} deg"
        )
        world_transform = self._tf_transform(self.reference_frame, LINK_NAME)
        if world_transform is not None:
            position = world_transform.translation
            lines.append(f"EE world: x={position.x:.4f} y={position.y:.4f} z={position.z:.4f}")
            roll, pitch, yaw = euler_from_quat(
                world_transform.rotation.x,
                world_transform.rotation.y,
                world_transform.rotation.z,
                world_transform.rotation.w,
            )
            lines.append(
                "EE orientation: "
                f"roll={math.degrees(roll):.1f} deg, pitch={math.degrees(pitch):.1f} deg, yaw={math.degrees(yaw):.1f} deg"
            )
        return "\n".join(lines)

    def print_status(self):
        self._log("info", self.status_text())

    def _position_constraint(self, x, y, z, tol=POSITION_TOL):
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self.reference_frame
        position_constraint.link_name = LINK_NAME
        position_constraint.weight = 1.0
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [tol, tol, tol]
        position_constraint.constraint_region.primitives.append(box)
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.reference_frame
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        target_pose.pose.orientation.w = 1.0
        position_constraint.constraint_region.primitive_poses.append(target_pose.pose)
        return position_constraint

    def _orientation_constraint(self):
        qx, qy, qz, qw = self.target_orientation
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.reference_frame
        orientation_constraint.link_name = LINK_NAME
        orientation_constraint.orientation.x = qx
        orientation_constraint.orientation.y = qy
        orientation_constraint.orientation.z = qz
        orientation_constraint.orientation.w = qw
        orientation_constraint.absolute_x_axis_tolerance = ORIENTATION_TOL
        orientation_constraint.absolute_y_axis_tolerance = ORIENTATION_TOL
        orientation_constraint.absolute_z_axis_tolerance = ORIENTATION_TOL
        orientation_constraint.weight = 1.0
        return orientation_constraint

    def move_xyz(self, dx, dy, dz, label):
        if not self.done.is_set():
            self._log("warn", "Still executing.")
            return
        transform = self._tf_transform(self.reference_frame, LINK_NAME)
        if transform is None:
            return
        target_x = transform.translation.x + dx
        target_y = transform.translation.y + dy
        target_z = transform.translation.z + dz
        self._log(
            "info",
            f"XYZ {math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) * 100.0:.1f}cm {label}"
            + (" [fixed orientation]" if self.maintain_orientation else " [position only]"),
        )
        self._send_pose_goal(target_x, target_y, target_z)

    def apply_orientation_here(self):
        transform = self._tf_transform(self.reference_frame, LINK_NAME)
        if transform is None:
            return
        self._log("info", "Applying orientation target at current position")
        self._send_pose_goal(transform.translation.x, transform.translation.y, transform.translation.z)

    def go_home(self):
        if not self.done.is_set():
            self._log("warn", "Still executing.")
            return
        self.done.clear()
        constraints = Constraints()
        for joint_name, value in HOME_JOINTS.items():
            constraints.joint_constraints.append(joint_constraint(joint_name, value, 0.03))
        self._log("info", "HOME -> all joints to 0 deg")
        self._send_constraints(constraints)

    def set_gripper(self, opening: float):
        opening = max(GRIPPER_MIN, min(GRIPPER_MAX, opening))
        if not self.hand_done.is_set():
            self._log("warn", "Gripper still executing.")
            return
        if not self._hand_client.wait_for_server(timeout_sec=1.0):
            self._log("error", "Hand controller action server unavailable")
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["left_gripper", "right_gripper"]
        point = JointTrajectoryPoint()
        point.positions = [opening, opening]
        point.time_from_start = BuiltinDuration(sec=0, nanosec=int(GRIPPER_TRAJECTORY_SECONDS * 1_000_000_000))
        goal.trajectory.points = [point]

        self._log("info", f"Gripper opening -> {opening:.3f} m")
        self.hand_done.clear()
        self._hand_client.send_goal_async(goal).add_done_callback(self._on_hand_goal)

    def _on_hand_goal(self, future):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._log("warn", "Gripper goal rejected.")
            self.hand_done.set()
            return
        goal_handle.get_result_async().add_done_callback(self._on_hand_result)

    def _on_hand_result(self, future):
        result = future.result()
        if result is None:
            self._log("warn", "Gripper command failed.")
            self.hand_done.set()
            return
        self._log("info", "Gripper command done.")
        self.hand_done.set()

    def _send_pose_goal(self, x: float, y: float, z: float):
        if not self.done.is_set():
            self._log("warn", "Still executing.")
            return
        self.done.clear()
        constraints = Constraints()
        constraints.position_constraints.append(self._position_constraint(x, y, z))
        if self.maintain_orientation:
            constraints.orientation_constraints.append(self._orientation_constraint())
        self._send_constraints(constraints)

    def _send_constraints(self, constraints: Constraints):
        # Debug: log constraint frames and basic info
        try:
            frame_info = []
            for pc in constraints.position_constraints:
                frame_info.append(f"pos(frame={pc.header.frame_id})")
            for oc in constraints.orientation_constraints:
                frame_info.append(f"orient(frame={oc.header.frame_id})")
            self._log("info", "Sending goal with constraints: " + ", ".join(frame_info))
        except Exception:
            pass
        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.allowed_planning_time = 5.0
        goal.request.num_planning_attempts = 10
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3
        goal.request.goal_constraints.append(constraints)
        # Wait for action server (log if unavailable)
        if not self._client.wait_for_server(timeout_sec=2.0):
            self._log("error", "Move action server unavailable when sending goal")
            self.done.set()
            return
        self._client.send_goal_async(goal).add_done_callback(self._on_goal)

    def _on_goal(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._log("warn", "Goal rejected.")
            self.done.set()
            return
        goal_handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        value = future.result().result.error_code.val
        self._log("info", "Done." if value == 1 else f"Failed (code {value})")
        self.done.set()


class TeleopGui:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.log_queue = Queue()
        self.ref_frame_var = tk.StringVar(value=self.node.reference_frame)
        self.xyz_step_var = tk.StringVar(value=str(DEFAULT_CM))
        self.hold_var = tk.BooleanVar(value=True)
        self.roll_var = tk.StringVar(value="0.0")
        self.pitch_var = tk.StringVar(value="0.0")
        self.yaw_var = tk.StringVar(value="0.0")
        self.gripper_var = tk.DoubleVar(value=GRIPPER_OPEN_BUTTON)
        self.mode_var = tk.StringVar(value="Mode: fixed orientation")
        self.exec_var = tk.StringVar(value="Planner: idle")
        self.joint_var = tk.StringVar(value="Waiting for joint states...")

        self.node.log_callback = self.enqueue_log

        self.root.title("SixDOF Pose Teleop")
        self.root.geometry("1020x820")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_ui()
        self.root.after(100, self._drain_logs)
        self.root.after(250, self._refresh_status)

    def enqueue_log(self, level, text):
        self.log_queue.put((level, text))

    def _build_ui(self):
        main = tk.Frame(self.root, padx=12, pady=12)
        main.pack(fill=tk.BOTH, expand=True)

        header = tk.Label(main, text="SixDOF Pose Teleop", font=("TkDefaultFont", 16, "bold"))
        header.pack(anchor=tk.W)

        info = tk.Label(
            main,
            text="Jog XYZ in the current reference frame. Toggle fixed orientation on or off. Use presets or custom RPY to test full 6DOF pose IK.",
            justify=tk.LEFT,
        )
        info.pack(anchor=tk.W, pady=(4, 10))

        status = tk.Frame(main)
        status.pack(fill=tk.X, pady=(0, 12))
        tk.Label(status, textvariable=self.mode_var, width=34, anchor=tk.W).pack(side=tk.LEFT)
        tk.Label(status, textvariable=self.exec_var, width=18, anchor=tk.W).pack(side=tk.LEFT, padx=(12, 0))
        # Reference frame selector
        ref_frame_frame = tk.Frame(status)
        ref_frame_frame.pack(side=tk.RIGHT)
        tk.Label(ref_frame_frame, text="Reference Frame:").pack(side=tk.LEFT)
        tk.OptionMenu(ref_frame_frame, self.ref_frame_var, "world", "ee_ref").pack(side=tk.LEFT)
        tk.Button(ref_frame_frame, text="Set", command=lambda: self.node.set_reference_frame(self.ref_frame_var.get())).pack(side=tk.LEFT, padx=(6,0))

        controls = tk.Frame(main)
        controls.pack(fill=tk.X)

        xyz = tk.LabelFrame(controls, text="XYZ Move (cm)", padx=10, pady=10)
        xyz.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        tk.Label(xyz, text="Step").grid(row=0, column=0, sticky="w")
        tk.Entry(xyz, textvariable=self.xyz_step_var, width=8).grid(row=0, column=1, sticky="w")
        tk.Button(xyz, text="+X", width=8, command=lambda: self._move_xyz(+1, 0, 0, "+X forward")).grid(row=1, column=1, pady=4)
        tk.Button(xyz, text="-X", width=8, command=lambda: self._move_xyz(-1, 0, 0, "-X back")).grid(row=3, column=1, pady=4)
        tk.Button(xyz, text="+Y", width=8, command=lambda: self._move_xyz(0, +1, 0, "+Y left")).grid(row=2, column=0, padx=4)
        tk.Button(xyz, text="-Y", width=8, command=lambda: self._move_xyz(0, -1, 0, "-Y right")).grid(row=2, column=2, padx=4)
        tk.Button(xyz, text="+Z", width=8, command=lambda: self._move_xyz(0, 0, +1, "+Z up")).grid(row=1, column=3, padx=(12, 0))
        tk.Button(xyz, text="-Z", width=8, command=lambda: self._move_xyz(0, 0, -1, "-Z down")).grid(row=3, column=3, padx=(12, 0))

        orient = tk.LabelFrame(controls, text="Orientation", padx=10, pady=10)
        orient.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        tk.Checkbutton(
            orient,
            text="Maintain fixed orientation",
            variable=self.hold_var,
            command=self._toggle_hold,
        ).grid(row=0, column=0, columnspan=4, sticky="w")
        tk.Button(orient, text="Capture Current", width=14, command=self._capture_current).grid(row=1, column=0, pady=4, sticky="w")
        tk.Button(orient, text="Apply Here", width=14, command=self.node.apply_orientation_here).grid(row=1, column=1, pady=4, sticky="w")
        tk.Button(orient, text="Look Forward", width=14, command=lambda: self._apply_preset("Look Forward")).grid(row=2, column=0, pady=4, sticky="w")
        tk.Button(orient, text="Look Down", width=14, command=lambda: self._apply_preset("Look Down")).grid(row=2, column=1, pady=4, sticky="w")
        tk.Button(orient, text="Look Up", width=14, command=lambda: self._apply_preset("Look Up")).grid(row=3, column=0, pady=4, sticky="w")
        tk.Button(orient, text="Look Right", width=14, command=lambda: self._apply_preset("Look Right")).grid(row=3, column=1, pady=4, sticky="w")
        tk.Button(orient, text="Look Left", width=14, command=lambda: self._apply_preset("Look Left")).grid(row=4, column=0, pady=4, sticky="w")
        tk.Button(orient, text="Apply RPY", width=14, command=self._apply_rpy).grid(row=4, column=1, pady=4, sticky="w")

        # Reference rotation controls: axis selector + custom degrees
        tk.Label(orient, text="Axis").grid(row=5, column=2, sticky="w")
        self.axis_var = tk.StringVar(value="Z")
        tk.OptionMenu(orient, self.axis_var, "X", "Y", "Z").grid(row=5, column=3, sticky="w")
        tk.Label(orient, text="Degrees").grid(row=6, column=2, sticky="w")
        self.deg_var = tk.StringVar(value="45")
        tk.Entry(orient, textvariable=self.deg_var, width=6).grid(row=6, column=3, sticky="w")
        tk.Button(orient, text="Rotate Ref", width=14, command=self._rotate_ref_custom).grid(row=7, column=2, columnspan=2, pady=4, sticky="w")

        tk.Label(orient, text="Roll").grid(row=5, column=0, sticky="w", pady=(8, 0))
        tk.Entry(orient, textvariable=self.roll_var, width=10).grid(row=5, column=1, sticky="w", pady=(8, 0))
        tk.Label(orient, text="Pitch").grid(row=6, column=0, sticky="w")
        tk.Entry(orient, textvariable=self.pitch_var, width=10).grid(row=6, column=1, sticky="w")
        tk.Label(orient, text="Yaw").grid(row=7, column=0, sticky="w")
        tk.Entry(orient, textvariable=self.yaw_var, width=10).grid(row=7, column=1, sticky="w")

        gripper = tk.LabelFrame(main, text="Gripper", padx=10, pady=10)
        gripper.pack(fill=tk.X, pady=(12, 12))
        tk.Scale(
            gripper,
            from_=GRIPPER_MIN,
            to=GRIPPER_MAX,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=400,
            variable=self.gripper_var,
            label="Opening (m)",
        ).pack(side=tk.LEFT, padx=(0, 12))
        buttons = tk.Frame(gripper)
        buttons.pack(side=tk.LEFT)
        tk.Button(buttons, text="Apply Slider", width=14, command=self._apply_gripper_slider).pack(pady=2)
        tk.Button(buttons, text="Open", width=14, command=self._open_gripper).pack(pady=2)
        tk.Button(buttons, text="Close", width=14, command=self._close_gripper).pack(pady=2)

        actions = tk.LabelFrame(main, text="Actions", padx=10, pady=10)
        actions.pack(fill=tk.X, pady=(0, 12))
        tk.Button(actions, text="Home", width=12, command=self.node.go_home).pack(side=tk.LEFT)
        tk.Button(actions, text="Refresh Status", width=14, command=self.node.print_status).pack(side=tk.LEFT, padx=6)
        tk.Button(actions, text="Quit", width=12, command=self.close).pack(side=tk.RIGHT)

        joint_frame = tk.LabelFrame(main, text="Current State", padx=10, pady=10)
        joint_frame.pack(fill=tk.X, pady=(0, 12))
        tk.Label(joint_frame, textvariable=self.joint_var, justify=tk.LEFT, anchor="w").pack(fill=tk.X)

        log_frame = tk.LabelFrame(main, text="Log", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_box = scrolledtext.ScrolledText(log_frame, height=18, state=tk.DISABLED, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def _append_log(self, line):
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, line + "\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _drain_logs(self):
        try:
            while True:
                level, text = self.log_queue.get_nowait()
                self._append_log(f"[{level.upper()}] {text}")
        except Empty:
            pass
        self.root.after(100, self._drain_logs)

    def _refresh_status(self):
        self.mode_var.set("Mode: fixed orientation" if self.node.maintain_orientation else "Mode: position only")
        self.exec_var.set("Planner: busy" if not self.node.done.is_set() else "Planner: idle")
        values = []
        for name in DISPLAY_JOINTS:
            if name in self.node.joints:
                values.append(f"{name}: {math.degrees(self.node.joints[name]):+.1f} deg")
            else:
                values.append(f"{name}: n/a")
        for name in ["left_gripper", "right_gripper"]:
            if name in self.node.joints:
                values.append(f"{name}: {self.node.joints[name]:+.3f} m")
        values.append(
            "Target RPY: "
            f"{self.node.target_orientation_rpy_deg[0]:+.1f}, "
            f"{self.node.target_orientation_rpy_deg[1]:+.1f}, "
            f"{self.node.target_orientation_rpy_deg[2]:+.1f} deg"
        )
        self.joint_var.set("\n".join(values))
        self.root.after(250, self._refresh_status)

    def _parse_float(self, value, label):
        try:
            return float(value)
        except ValueError:
            self.enqueue_log("error", f"Invalid {label}: {value}")
            return None

    def _move_xyz(self, x_sign, y_sign, z_sign, label):
        cm = self._parse_float(self.xyz_step_var.get(), "XYZ step")
        if cm is None:
            return
        metres = cm / 100.0
        self.node.move_xyz(x_sign * metres, y_sign * metres, z_sign * metres, label)

    def _toggle_hold(self):
        enabled = self.hold_var.get()
        self.node.set_maintain_orientation(enabled)
        if enabled and self.node.capture_current_orientation():
            self.roll_var.set(f"{self.node.target_orientation_rpy_deg[0]:.1f}")
            self.pitch_var.set(f"{self.node.target_orientation_rpy_deg[1]:.1f}")
            self.yaw_var.set(f"{self.node.target_orientation_rpy_deg[2]:.1f}")

    def _capture_current(self):
        if self.node.capture_current_orientation():
            self.roll_var.set(f"{self.node.target_orientation_rpy_deg[0]:.1f}")
            self.pitch_var.set(f"{self.node.target_orientation_rpy_deg[1]:.1f}")
            self.yaw_var.set(f"{self.node.target_orientation_rpy_deg[2]:.1f}")
            self.enqueue_log("info", "Captured current end-effector orientation into the target fields.")

    def _apply_preset(self, preset_name):
        roll_deg, pitch_deg, yaw_deg = ORIENTATION_PRESETS_DEG[preset_name]
        self.roll_var.set(f"{roll_deg:.1f}")
        self.pitch_var.set(f"{pitch_deg:.1f}")
        self.yaw_var.set(f"{yaw_deg:.1f}")
        self.node.set_orientation_from_rpy_deg(roll_deg, pitch_deg, yaw_deg)
        self.enqueue_log("info", f"Applied preset: {preset_name}")

    def _apply_rpy(self):
        roll_deg = self._parse_float(self.roll_var.get(), "roll")
        pitch_deg = self._parse_float(self.pitch_var.get(), "pitch")
        yaw_deg = self._parse_float(self.yaw_var.get(), "yaw")
        if None in (roll_deg, pitch_deg, yaw_deg):
            return
        self.node.set_orientation_from_rpy_deg(roll_deg, pitch_deg, yaw_deg)

    def _rotate_ref_custom(self):
        try:
            deg = float(self.deg_var.get())
        except ValueError:
            self.enqueue_log("error", f"Invalid degrees: {self.deg_var.get()}")
            return
        axis = self.axis_var.get()
        self.node.rotate_reference(axis, deg)

    def _apply_gripper_slider(self):
        self.node.set_gripper(self.gripper_var.get())

    def _open_gripper(self):
        self.gripper_var.set(GRIPPER_OPEN_BUTTON)
        self.node.set_gripper(GRIPPER_OPEN_BUTTON)

    def _close_gripper(self):
        self.gripper_var.set(GRIPPER_CLOSE_BUTTON)
        self.node.set_gripper(GRIPPER_CLOSE_BUTTON)

    def close(self):
        self.root.quit()


def run_headless_check():
    rclpy.init()
    node = Teleop()
    node._log("info", "Headless teleop check passed.")
    node.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless-check", action="store_true", help="Initialize the ROS node without starting the GUI")
    args = parser.parse_args()

    if args.headless_check:
        run_headless_check()
        return

    rclpy.init()
    node = Teleop()
    root = tk.Tk()
    app = TeleopGui(root, node)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        app._capture_current()
        app._append_log("[INFO] 6DOF pose teleop GUI ready.")
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
