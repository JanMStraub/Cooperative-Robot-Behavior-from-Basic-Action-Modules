"""Launch robot_state_publisher for AR4 mk3.

Parameterized by robot_id for multi-robot support.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for robot_state_publisher.

    Args:
        robot_id: Robot namespace (default: "Robot1"). Example: "Robot1", "Robot2"
    """
    pkg_dir = get_package_share_directory("ar4_moveit_config")

    # Declare launch arguments
    robot_id_arg = DeclareLaunchArgument(
        "robot_id",
        default_value="Robot1",
        description="Robot namespace for topic routing (e.g., Robot1, Robot2)"
    )

    # Get launch configuration
    robot_id = LaunchConfiguration("robot_id")

    urdf_path = os.path.join(pkg_dir, "urdf", "ar4.urdf")
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    return LaunchDescription([
        robot_id_arg,
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=robot_id,  # Namespace all topics with robot ID
            output="screen",
            parameters=[
                {"robot_description": robot_description},
                {"use_sim_time": False},
            ],
        ),
    ])
