import os
import tempfile
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def software_gl_actions():
    if os.environ.get('SIXDOF_FORCE_SOFTWARE_GL', '1').lower() in {'0', 'false', 'no', 'off'}:
        return []
    return [
        SetEnvironmentVariable(name='LIBGL_ALWAYS_SOFTWARE', value='1'),
        SetEnvironmentVariable(name='QT_OPENGL', value='software'),
        SetEnvironmentVariable(name='QT_X11_NO_MITSHM', value='1'),
    ]

def generate_launch_description():
    # --- 1. GET DYNAMIC PACKAGE PATHS ---
    pkg_sixdof_pkg = get_package_share_directory('sixdof_pkg')
    pkg_sixdof_moveit = get_package_share_directory('sixdof_moveit')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # --- 2. DEFINE PATHS ---
    urdf_file_path = os.path.join(pkg_sixdof_pkg, 'urdf', 'roar.urdf')
    meshes_path = os.path.join(pkg_sixdof_pkg, 'meshes')
    controllers_yaml_path = os.path.join(pkg_sixdof_moveit, 'config', 'ros2_controllers.yaml')

    # --- 3. PROCESS URDF ---
    with open(urdf_file_path, 'r') as file:
        robot_desc_content = file.read()
    
    # Replace package:// with file:// for Gazebo
    robot_desc_content = robot_desc_content.replace('package://sixdof_pkg/meshes', 'file://' + meshes_path)
    robot_desc_content = robot_desc_content.replace('package://sixdof_moveit/config/ros2_controllers.yaml', controllers_yaml_path)

    # --- CRITICAL FIX: Write to Temp File ---
    # This prevents the "Syntax error: newline unexpected" crash
    tmp_urdf = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.urdf')
    tmp_urdf.write(robot_desc_content)
    tmp_urdf.close()
    tmp_urdf_path = tmp_urdf.name

    # --- 4. GAZEBO SIMULATION (FORTRESS MODE) ---
    # We use ros_gz_sim, but with GZ_VERSION=fortress set in terminal, it runs Fortress.
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # --- 5. SPAWN THE ROBOT ---
    spawn_entity = Node(
        package='ros_gz_sim', 
        executable='create',
        arguments=['-name', 'sixdof_pkg',
                   '-file', tmp_urdf_path,
                   '-x', '0', '-y', '0', '-z', '0.02'],
        output='screen',
    )

    # --- 6. ROBOT STATE PUBLISHER ---
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc_content, 'use_sim_time': True}],
        remappings=[('/joint_states', '/joint_states_corrected')],
    )

    gripper_state_republisher = Node(
        package='sixdof_pkg',
        executable='gripper_joint_state_republisher.py',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # --- 7. BRIDGE (ROS <-> FORTRESS) ---
    # Since we are in Fortress mode, we map to ign.msgs
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen'
    )

    # --- 8. CONTROLLERS ---
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller_controller"], # Matches ros2_controllers.yaml
    )

    ee_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["hand_controller_controller"],
    )

    # Delay controllers to make sure robot is spawned first
    delay_controllers = TimerAction(
        period=5.0, 
        actions=[joint_state_broadcaster, arm_controller, ee_controller],
    )

    return LaunchDescription([
        *software_gl_actions(),
        gazebo,
        gripper_state_republisher,
        node_robot_state_publisher,
        spawn_entity,
        bridge,
        delay_controllers,
    ])