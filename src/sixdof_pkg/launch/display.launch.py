from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # 1. Set the correct package and file names
    pkg_path = get_package_share_directory('sixdof_pkg')
    urdf_file = os.path.join(pkg_path, 'urdf', 'roar.urdf')

    # 2. Read the URDF file directly
    with open(urdf_file, 'r') as infp:
        robot_description_content = infp.read()

    # 3. Robot State Publisher (Publishes TF tree)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}]
    )

    # 4. Joint State Publisher GUI (Publishes Joint States for the TF tree)
    # CRITICAL FIX: We must pass the robot_description here too!
    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        parameters=[{'robot_description': robot_description_content}]
    )

    # 5. RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node
    ])