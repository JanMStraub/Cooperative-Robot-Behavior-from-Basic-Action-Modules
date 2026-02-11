"""Full demo launch: robot_state_publisher + move_group for AR4 mk3."""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate demo launch combining robot state publisher and move_group."""
    pkg_dir = get_package_share_directory("ar4_moveit_config")

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, "launch", "robot_state_publisher.launch.py")
            ),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, "launch", "move_group.launch.py")
            ),
        ),
    ])
