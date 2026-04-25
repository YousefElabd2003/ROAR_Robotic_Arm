#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from collision_guard import CollisionGuard
import sys
import threading

# === CONFIGURATION ===
# The tip link of the arm chain before the grippers.
LINK_NAME = "link_6"  

# The MoveIt planning group for the arm joints.
GROUP_NAME = "arm_controller"  

# The fixed world frame
FRAME_ID = "world"

class MoveRobotInteractive(Node):
    def __init__(self):
        super().__init__('move_robot_interactive')
        
        # 1. Action Client (To Send Commands to MoveIt)
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        
        # 2. TF Listener (To Know Where the Arm Is)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.current_joints = {}
        self.create_subscription(JointState, "joint_states", self._joint_state_cb, 10)
        self.collision_guard = CollisionGuard(self, GROUP_NAME)
        
        self.goal_done = threading.Event()
        self.goal_done.set() # Ready to start

    def _joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos

    def get_current_pos(self):
        try:
            # Wait up to 1 second for the transform
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                FRAME_ID, 
                LINK_NAME, 
                now, 
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return trans.transform.translation
        except Exception as e:
            self.get_logger().error(f"Could not find robot position: {e}")
            return None

    def send_relative_goal(self, dx, dy, dz):
        self.goal_done.clear()
        
        # 1. Get Current Position
        current = self.get_current_pos()
        if current is None:
            self.goal_done.set()
            return

        target_x = current.x + dx
        target_y = current.y + dy
        target_z = current.z + dz
        
        print(f"Moving from ({current.x:.2f}, {current.y:.2f}, {current.z:.2f}) -> ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")

        # 2. Construct MoveGroup Goal
        goal_msg = MoveGroup.Goal()
        goal_msg.request.workspace_parameters.header.frame_id = FRAME_ID
        goal_msg.request.group_name = GROUP_NAME
        goal_msg.request.allowed_planning_time = 10.0  # 5 seconds to plan
        goal_msg.request.num_planning_attempts = 100   
        
        # Create Constraints
        constraints = Constraints()
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = FRAME_ID
        pos_constraint.link_name = LINK_NAME
        
        # Create a small target box for the end effector to reach
        target_box = SolidPrimitive()
        target_box.type = SolidPrimitive.BOX
        target_box.dimensions = [0.01, 0.01, 0.01] # 1cm tolerance box
        pos_constraint.constraint_region.primitives.append(target_box)
        
        target_pose = PoseStamped()
        target_pose.header.frame_id = FRAME_ID
        target_pose.pose.position.x = target_x
        target_pose.pose.position.y = target_y
        target_pose.pose.position.z = target_z
        
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)
        pos_constraint.weight = 1.0
        constraints.position_constraints.append(pos_constraint)
        goal_msg.request.goal_constraints.append(constraints)

        ok, reason = self.collision_guard.check_current_state(self.current_joints)
        if not ok:
            self.get_logger().warn(f"Blocked by collision guard: {reason}")
            self.goal_done.set()
            return

        # 3. Send Goal
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("Goal rejected (Check if MoveIt is running/Group name is correct).")
            self.goal_done.set()
            return

        print("Planning & Executing...")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            print(">>> SUCCESS <<<")
        else:
            print(f">>> FAILED (Error Code: {result.error_code.val}) <<<")
        self.goal_done.set()

def user_input_thread(node):
    print("-------------------------------------------------")
    print(" Interactive Controller Active")
    print(" Commands: 'x 0.05'  (Move X by 5cm)")
    print("           'z -0.1'  (Move Z down 10cm)")
    print("           'y 0.01'  (Move Y by 1cm)")
    print("           'q'       (Quit)")
    print("-------------------------------------------------")
    
    # Give ROS time to initialize
    import time
    time.sleep(2)
    
    while rclpy.ok():
        try:
            cmd = input("Enter Command: ").strip().split()
            if not cmd: continue
            
            axis = cmd[0].lower()
            if axis == 'q':
                print("Quitting...")
                rclpy.shutdown()
                break
            
            if len(cmd) < 2:
                print("Invalid format. Use: axis amount (e.g., 'x 0.1')")
                continue

            val = float(cmd[1])
            dx, dy, dz = 0.0, 0.0, 0.0
            
            if axis == 'x': dx = val
            elif axis == 'y': dy = val
            elif axis == 'z': dz = val
            else:
                print("Unknown axis. Use x, y, or z.")
                continue
                
            # Send the command
            node.send_relative_goal(dx, dy, dz)
            
            # Wait for completion
            node.goal_done.wait()
            node.goal_done.clear()
            
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error: {e}")
            break

def main():
    rclpy.init()
    node = MoveRobotInteractive()
    
    # Run user input in a separate thread so ROS callbacks can keep spinning
    thread = threading.Thread(target=user_input_thread, args=(node,), daemon=True)
    thread.start()
    
    # Spin the ROS node in the main thread
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()