#!/usr/bin/env python3
"""
macro_orientation_teleop.py — Save/apply orientation macros + XYZ teleop
-------------------------------------------------------------------------
kinematics.yaml: KDL, position_only_ik: true  (no changes needed)

INPUT MODES:
  "w 5"   + Enter  → move 5cm forward
  "s 3.5" + Enter  → move 3.5cm back
  "w"     + Enter  → move 1cm forward (default)
  "save 1"         → save current orientation to slot 1
  "save 2"         → save current orientation to slot 2
  "save 3"         → save current orientation to slot 3
  "lock 1"         → lock to slot 1
  "lock 2"         → lock to slot 2
  "lock 3"         → lock to slot 3
  "u"              → unlock
  "p"              → print all saved macros
  "x"              → quit
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
import sys, termios, threading

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
┌──────────────────────────────────────────────┐
│   ROVER Teleop — Macro Orientation Mode      │
├──────────────────────────────────────────────┤
│  MOVEMENT (all distances in cm):             │
│    "w"        → forward 1 cm (default)       │
│    "w 5"      → forward 5 cm                 │
│    "s 3.5"    → back 3.5 cm                  │
│    "a / d"    → Y axis left / right          │
│    "q / e"    → Z axis up / down             │
│                                              │
│  SAVE current orientation:                   │
│    "save 1"   → save to slot 1               │
│    "save 2"   → save to slot 2               │
│    "save 3"   → save to slot 3               │
│                                              │
│  LOCK to saved orientation:                  │
│    "lock 1"   → lock slot 1                  │
│    "lock 2"   → lock slot 2                  │
│    "lock 3"   → lock slot 3                  │
│                                              │
│    "u"        → unlock (free movement)       │
│    "p"        → print saved macros           │
│    "x"        → quit                         │
├──────────────────────────────────────────────┤
│  WORKFLOW:                                   │
│    1. Move arm to desired orientation        │
│    2. Type "save 1" and Enter                │
│    3. Type "lock 1" and Enter                │
│    4. Teleop with e.g. "w 5", "q 2"         │
└──────────────────────────────────────────────┘
"""


class Macro:
    def __init__(self, pitch: float, twist: float):
        self.pitch = pitch
        self.twist = twist

    def __str__(self):
        return (
            f"{PITCH_JOINT}(pitch)={self.pitch:.4f} rad  "
            f"{TWIST_JOINT}(twist)={self.twist:.4f} rad"
        )


class MacroOrientationTeleop(Node):
    def __init__(self):
        super().__init__("macro_orientation_teleop")
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

        self.macros: dict[int, Macro | None] = {1: None, 2: None, 3: None}
        self.locked = False
        self.active_macro: Macro | None = None
        self.active_slot: int = 0

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

    def save_macro(self, slot: int):
        missing = [j for j in [PITCH_JOINT, TWIST_JOINT] if j not in self.joint_positions]
        if missing:
            self.get_logger().error(f"Missing joint states for: {missing}")
            return
        macro = Macro(
            self.joint_positions[PITCH_JOINT],
            self.joint_positions[TWIST_JOINT],
        )
        self.macros[slot] = macro
        self.get_logger().info(
            f"✓ Saved slot {slot}: {macro}\n"
            f"  Type 'lock {slot}' to activate it."
        )

    def lock_macro(self, slot: int):
        macro = self.macros.get(slot)
        if macro is None:
            self.get_logger().warn(
                f"Slot {slot} is empty. Type 'save {slot}' first."
            )
            return
        self.active_macro = macro
        self.active_slot = slot
        self.locked = True
        self.get_logger().info(f"✓ LOCKED to slot {slot}: {macro}")

    def unlock(self):
        self.locked = False
        self.active_macro = None
        self.active_slot = 0
        self.get_logger().info("Orientation UNLOCKED")

    def print_macros(self):
        lines = ["", "═══ Saved Macros ═══"]
        for slot in range(1, 4):
            macro = self.macros[slot]
            active = "  ← ACTIVE" if self.locked and self.active_slot == slot else ""
            if macro:
                lines.append(f"  Slot {slot}: {macro}{active}")
            else:
                lines.append(f"  Slot {slot}: (empty — type 'save {slot}' to fill)")
        lines.append("════════════════════")
        self.get_logger().info("\n".join(lines))

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

        if self.locked and self.active_macro is not None:
            for joint_name, locked_val, tol in [
                (PITCH_JOINT, self.active_macro.pitch, PITCH_TOL),
                (TWIST_JOINT, self.active_macro.twist, TWIST_TOL),
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


def parse_move(raw: str):
    """Parse "w", "w5", "w 5" into (dx, dy, dz) in metres."""
    raw = raw.strip().lower()
    if not raw or raw[0] not in DIRECTION_MAP:
        return None
    rest = raw[1:].strip()
    try:
        cm = DEFAULT_CM if rest == "" else float(rest)
    except ValueError:
        return None
    metres = cm / 100.0
    sx, sy, sz = DIRECTION_MAP[raw[0]]
    return sx * metres, sy * metres, sz * metres


def user_input_thread(node: MacroOrientationTeleop):
    settings = termios.tcgetattr(sys.stdin)
    print(msg)

    try:
        while rclpy.ok():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            slot_str = f"slot {node.active_slot}" if node.locked else "free"
            sys.stdout.write(f"[orient={slot_str}] cmd> ")
            sys.stdout.flush()

            try:
                line = sys.stdin.readline()
            except EOFError:
                break
            if not line:
                continue

            line = line.strip()
            parts = line.split()
            if not parts:
                continue

            cmd = parts[0].lower()

            # Quit
            if cmd == "x":
                print("Exiting...")
                rclpy.shutdown()
                break

            # Unlock
            elif cmd == "u":
                node.unlock()

            # Print macros
            elif cmd == "p":
                node.print_macros()

            # Save macro: "save 1"
            elif cmd == "save":
                if len(parts) < 2 or parts[1] not in ("1", "2", "3"):
                    print("  Usage: save 1  /  save 2  /  save 3")
                else:
                    node.save_macro(int(parts[1]))

            # Lock macro: "lock 1"
            elif cmd == "lock":
                if len(parts) < 2 or parts[1] not in ("1", "2", "3"):
                    print("  Usage: lock 1  /  lock 2  /  lock 3")
                else:
                    node.lock_macro(int(parts[1]))

            # Movement: "w", "w 5", "w5"
            elif cmd[0] in DIRECTION_MAP:
                # Reassemble in case user typed "w 5" (split into ["w", "5"])
                combined = cmd + ("" if len(parts) == 1 else " " + parts[1])
                result = parse_move(combined)
                if result is None:
                    print(f"  Invalid command. Example: 'w 5' or 'q 2.5'")
                    continue
                dx, dy, dz = result
                cm_moved = (dx**2 + dy**2 + dz**2) ** 0.5 * 100
                node.get_logger().info(
                    f"Moving '{cmd[0].upper()}' {cm_moved:.1f} cm  "
                    f"Δ({dx:+.4f}, {dy:+.4f}, {dz:+.4f}) m  orient={slot_str}"
                )
                node.send_relative_goal(dx, dy, dz)

            else:
                print(f"  Unknown command '{line}'. Type 'p' to see macros or 'x' to quit.")

    except Exception as e:
        print(e)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main():
    rclpy.init()
    node = MacroOrientationTeleop()
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