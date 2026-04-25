# import the packages
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import SetParameter
from ament_index_python.packages import get_package_share_directory
import os


def software_gl_actions():
    if os.environ.get('SIXDOF_FORCE_SOFTWARE_GL', '1').lower() in {'0', 'false', 'no', 'off'}:
        return []
    return [
        SetEnvironmentVariable(name='LIBGL_ALWAYS_SOFTWARE', value='1'),
        SetEnvironmentVariable(name='QT_OPENGL', value='software'),
        SetEnvironmentVariable(name='QT_X11_NO_MITSHM', value='1'),
    ]


def generate_launch_description():
    moveit_pkg_path = get_package_share_directory('sixdof_moveit')
    sixdof_pkg_path = get_package_share_directory('sixdof_pkg')

    cleanup_gazebo = ExecuteProcess(
        cmd=['bash', '-lc', "pkill -f 'ign gazebo' || true"],
        output='screen'
    )

    lab_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(sixdof_pkg_path, 'launch', 'gazebo.launch.py')
        ])
    )

    move_group_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(moveit_pkg_path, 'launch', 'move_group.launch.py')
        ]),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    moveit_rviz_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(moveit_pkg_path, 'launch', 'moveit_rviz.launch.py')
        ]),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    delay_move_group = TimerAction(
        period=8.0,
        actions=[move_group_node]
    )

    delay_rviz = TimerAction(
        period=11.0,
        actions=[moveit_rviz_node]
    )

    return LaunchDescription([
        *software_gl_actions(),
        SetParameter(name='use_sim_time', value=True),
        cleanup_gazebo,
        TimerAction(period=1.0, actions=[lab_gazebo]),
        delay_move_group,
        delay_rviz,
    ])