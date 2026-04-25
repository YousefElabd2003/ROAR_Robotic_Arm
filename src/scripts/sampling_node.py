import rclpy
from rover_arm_base import RoverRoutineBase

class SamplingNode(RoverRoutineBase):
    def __init__(self):
        super().__init__("sampling_routine")
        # 15cm Tool Transformation + 10deg mount tilt
        self.T_ee = self.get_tcp_matrix(0, 0, 0.15, 0, 0.174, 0)

    def start_routine(self, x, y, z, pitch):
        self.execute_move(self.T_ee, x, y, z, pitch)

def main():
    rclpy.init()
    node = SamplingNode()
    node.start_routine(0.5, 0.1, 0.3, 15.0)
    rclpy.spin(node)