import rclpy
from rover_arm_base import RoverRoutineBase
from std_msgs.msg import Float32MultiArray

class GrippingNode(RoverRoutineBase):
    def __init__(self):
        super().__init__("gripping_routine")
        # 5cm offset on Z-axis
        self.T_ee = self.get_tcp_matrix(0, 0, 0.05, 0, 0, 0)
        self.create_subscription(Float32MultiArray, "/rover/gripping/target", self.cb, 10)
        self.get_logger().info("Gripping Node: ACTIVE. Listening on /rover/gripping/target")

    def cb(self, msg):
        self.execute_move(self.T_ee, msg.data[0], msg.data[1], msg.data[2], msg.data[3])

def main(args=None):
    rclpy.init(args=args)
    node = GrippingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()