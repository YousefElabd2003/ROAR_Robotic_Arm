import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class WristDebugMonitor(Node):
    def __init__(self):
        super().__init__('wrist_debug_monitor')
        
        # --- MOTOR SPECIFICATIONS (JGB37-520 37RPM) ---
        # 11 PPR x 4 (Quadrature) x 270 (Gear Ratio) = ~11,880 ticks/rev
        self.TICKS_PER_REV = 11880
        self.TICKS_PER_RAD = self.TICKS_PER_REV / (2 * math.pi)  # approx 1890.7 ticks/rad
        self.MAX_RPM = 37.0
        
        # Subscribe to the Joint States (from RViz Sliders)
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',  # Topic name usually published by joint_state_publisher_gui
            self.listener_callback,
            10
        )
        
        self.pitch_joint_name = 'joint_4'
        self.roll_joint_name = 'joint_5'

        self.get_logger().info("Wrist Monitor Started. Waiting for RViz...")

    def listener_callback(self, msg):
        try:
            # 1. FIND THE JOINTS
            if self.pitch_joint_name in msg.name and self.roll_joint_name in msg.name:
                pitch_index = msg.name.index(self.pitch_joint_name)
                roll_index = msg.name.index(self.roll_joint_name)
                
                # 2. GET INPUT (Radians from RViz)
                # Note: 'position' gives us Angle, but motors need Velocity to get there.
                # For this specific debug script, we act as if Position = Target Position.
                pitch_rad = msg.position[pitch_index]
                roll_rad = msg.position[roll_index]
                
                # 3. CONVERT TO TICKS (The "Virtual" Command)
                # How many ticks away from Zero is this position?
                cmd_pitch_ticks = pitch_rad * self.TICKS_PER_RAD
                cmd_roll_ticks = roll_rad * self.TICKS_PER_RAD

                # 4. APPLY DIFFERENTIAL LOGIC (Face-to-Face)
                # Right Motor = Roll + Pitch
                # Left Motor  = Roll - Pitch
                target_ticks_right = cmd_roll_ticks + cmd_pitch_ticks
                target_ticks_left  = cmd_roll_ticks - cmd_pitch_ticks
                
                # 5. CONVERT TO RPM (For visualization)
                # (Ticks / Ticks_Per_Rev) * 60 seconds (assuming this was a speed cmd)
                # Just for display, let's show "Rotations from Zero"
                rotations_right = target_ticks_right / self.TICKS_PER_REV
                rotations_left = target_ticks_left / self.TICKS_PER_REV

                # 6. OUTPUT TO TERMINAL
                print(f"\n--- RVIZ INPUT (Simulated) ---")
                print(f"Pitch: {math.degrees(pitch_rad):.1f}°  |  Roll: {math.degrees(roll_rad):.1f}°")
                
                print(f"--- HARDWARE COMMANDS ---")
                print(f"LEFT MOTOR:  {int(target_ticks_left)} ticks  ({rotations_left:.2f} revs)")
                print(f"RIGHT MOTOR: {int(target_ticks_right)} ticks  ({rotations_right:.2f} revs)")
                
                # Check for "Impossible" ranges (Software Limit Switch)
                if abs(rotations_right) > 5.0 or abs(rotations_left) > 5.0:
                    print("!! WARNING: COMMAND EXCEEDS CABLE LIMITS !!")
                    
        except ValueError:
            pass 

def main(args=None):
    rclpy.init(args=args)
    monitor = WristDebugMonitor()
    rclpy.spin(monitor)
    monitor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()