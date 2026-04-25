import rclpy
from rover_arm_base import RoverRoutineBase
from std_msgs.msg import Float32MultiArray

class ProbingNode(RoverRoutineBase):
    def __init__(self):
        super().__init__("probing_routine")
        # 22cm offset for the long probe
        self.T_ee = self.get_tcp_matrix(0, 0, 0.22, 0, 0, 0)
        self.create_subscription(Float32MultiArray, "/rover/probing/target", self.cb, 10)
        self.get_logger().info("Probing Node: ACTIVE. Listening on /rover/probing/target")

    def cb(self, msg):
        # Slower v_scale for probing safety
        self.execute_move(self.T_ee, msg.data[0], msg.data[1], msg.data[2], msg.data[3], v_scale=0.05)

def main(args=None):
    rclpy.init(args=args)
    node = ProbingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()