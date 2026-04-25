#!/usr/bin/env python3
"""
wrist_lock_teleop.py — Lock joint_4 (pitch) + joint_5 (twist), move XYZ
-------------------------------------------------------------------------
kinematics.yaml: KDL, position_only_ik: true  (no changes needed)

INPUT MODES:
  Single letter        → move 1cm in that direction
  "w 5"  + Enter       → move 5cm forward
  "s 3.5" + Enter      → move 3.5cm back
  "c"    + Enter       → lock current pitch + twist
  "u"    + Enter       → unlock
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, JointConstraint
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from collision_guard import CollisionGuard
import sys, select, termios, tty, threading

LINK_NAME   = "link_6"
GROUP_NAME  = "arm_controller"
FRAME_ID    = "world"
DEFAULT_CM  = 1.0

PITCH_JOINT = "joint_4"
TWIST_JOINT = "joint_5"
PITCH_TOL   = 0.05
TWIST_TOL   = 0.05

DIRECTION_MAP = {
    "w": ( 1,  0,  0),
    "s": (-1,  0,  0),
    "a": ( 0,  1,  0),
    "d": ( 0, -1,  0),
    "q": ( 0,  0,  1),
    "e": ( 0,  0, -1),
}

msg = """
┌─────────────────────────────────────────────┐
│     ROVER Teleop — Wrist Lock Mode          │
├─────────────────────────────────────────────┤
│  MOVEMENT COMMANDS:                         │
│    w / s  → X axis  (forward / back)        │
│    a / d  → Y axis  (left / right)          │
│    q / e  → Z axis  (up / down)             │
│                                             │
│  DISTANCE (optional, in cm):                │
│    "w"         → move 1 cm (default)        │
│    "w 5"       → move 5 cm forward          │
│    "s 3.5"     → move 3.5 cm back           │
│    "q 10"      → move 10 cm up              │
│                                             │
│  ORIENTATION:                               │
│    c  → LOCK current joint_4 + joint_5      │
│    u  → UNLOCK wrist                        │
│                                             │
│    x  → Quit                                │
└─────────────────────────────────────────────┘
"""


class WristLockedTeleop(Node):
    def __init__(self):
        super().__init__("wrist_locked_teleop")
        self._action_client = ActionClient(self, MoveGroup, "move_action")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_done = threading.Event()
        self.goal_done.set()

        self.joint_positions = {}
        self.joint_sub = self.create_subscription(
            JointState, "joint_states", self._joint_state_cb, 10
        )
        self.collision_guard = CollisionGuard(self, GROUP_NAME)
        self.wrist_locked = False
        self.locked_pitch = 0.0
        self.locked_twist = 0.0

    def _joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def get_current_pos(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                FRAME_ID, LINK_NAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            return trans.transform.translation
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None

    def lock_wrist(self):
        missing = [j for j in [PITCH_JOINT, TWIST_JOINT] if j not in self.joint_positions]
        if missing:
            self.get_logger().error(f"Missing joint states for: {missing}")
            return
        self.locked_pitch = self.joint_positions[PITCH_JOINT]
        self.locked_twist = self.joint_positions[TWIST_JOINT]
        self.wrist_locked = True
        self.get_logger().info(
            f"Wrist LOCKED:\n"
            f"  {PITCH_JOINT} (pitch) = {self.locked_pitch:.4f} rad\n"
            f"  {TWIST_JOINT} (twist) = {self.locked_twist:.4f} rad"
        )

    def unlock_wrist(self):
        self.wrist_locked = False
        self.get_logger().info("Wrist UNLOCKED")

    def send_relative_goal(self, dx, dy, dz):
        if not self.goal_done.is_set():
            self.get_logger().warn("Still executing, ignoring.")
            return
        self.goal_done.clear()
        current = self.get_current_pos()
        if current is None:
            self.goal_done.set()
            return

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = GROUP_NAME
        goal_msg.request.allowed_planning_time = 3.0
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.max_velocity_scaling_factor = 0.3
        goal_msg.request.max_acceleration_scaling_factor = 0.3

        constraints = Constraints()

        pos_c = PositionConstraint()
        pos_c.header.frame_id = FRAME_ID
        pos_c.link_name = LINK_NAME
        pos_c.weight = 1.0
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 0.01, 0.01]
        pos_c.constraint_region.primitives.append(box)
        pose = PoseStamped()
        pose.header.frame_id = FRAME_ID
        pose.pose.position.x = current.x + dx
        pose.pose.position.y = current.y + dy
        pose.pose.position.z = current.z + dz
        pose.pose.orientation.w = 1.0
        pos_c.constraint_region.primitive_poses.append(pose.pose)
        constraints.position_constraints.append(pos_c)

        if self.wrist_locked:
            for joint_name, locked_val, tol in [
                (PITCH_JOINT, self.locked_pitch, PITCH_TOL),
                (TWIST_JOINT, self.locked_twist, TWIST_TOL),
            ]:
                jc = JointConstraint()
                jc.joint_name = joint_name
                jc.position = locked_val
                jc.tolerance_above = tol
                jc.tolerance_below = tol
                jc.weight = 1.0
                constraints.joint_constraints.append(jc)

        goal_msg.request.goal_constraints.append(constraints)

        ok, reason = self.collision_guard.check_current_state(self.joint_positions)
        if not ok:
            self.get_logger().warn(f"Blocked by collision guard: {reason}")
            self.goal_done.set()
            return

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected.")
            self.goal_done.set()
            return
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result().result
        if result.error_code.val != 1:
            self.get_logger().warn(f"Move failed (code {result.error_code.val})")
        else:
            self.get_logger().info("Move succeeded.")
        self.goal_done.set()


def parse_command(raw: str):
    raw = raw.strip().lower()
    if not raw:
        return None
    direction = raw[0]
    if direction not in DIRECTION_MAP:
        return None
    rest = raw[1:].strip()
    cm = DEFAULT_CM if rest == "" else float(rest)
    metres = cm / 100.0
    sx, sy, sz = DIRECTION_MAP[direction]
    return sx * metres, sy * metres, sz * metres


def user_input_thread(node: WristLockedTeleop):
    settings = termios.tcgetattr(sys.stdin)
    print(msg)

    try:
        while rclpy.ok():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            status = f"LOCKED(J4={node.locked_pitch:.2f} J5={node.locked_twist:.2f})" \
                     if node.wrist_locked else "free"
            sys.stdout.write(f"[wrist={status}] cmd> ")
            sys.stdout.flush()

            try:
                line = sys.stdin.readline()
            except EOFError:
                break
            if not line:
                continue

            line = line.strip()

            if line == "x":
                print("Exiting...")
                rclpy.shutdown()
                break
            elif line == "c":
                node.lock_wrist()
                continue
            elif line == "u":
                node.unlock_wrist()
                continue

            try:
                result = parse_command(line)
            except ValueError:
                print(f"  Invalid distance. Use: w/a/s/d/q/e [cm]  e.g. 'w 5'")
                continue

            if result is None:
                print(f"  Unknown command '{line}'. Use: w/a/s/d/q/e [cm]  or c/u/x")
                continue

            dx, dy, dz = result
            cm_moved = (dx**2 + dy**2 + dz**2) ** 0.5 * 100
            node.get_logger().info(
                f"Moving '{line[0].upper()}' {cm_moved:.1f} cm  "
                f"Δ({dx:+.4f}, {dy:+.4f}, {dz:+.4f}) m  wrist={status}"
            )
            node.send_relative_goal(dx, dy, dz)

    except Exception as e:
        print(e)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main():
    rclpy.init()
    node = WristLockedTeleop()
    thread = threading.Thread(target=user_input_thread, args=(node,), daemon=True)
    thread.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()