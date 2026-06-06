"""Launch robot_state_publisher for dual AR4 mk3 (shared planning scene).

Uses ar4_dual.urdf and subscribes to /joint_states (aggregated from both robots).
No robot namespace — publishes to /robot_description directly.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("ar4_moveit_config")

    urdf_path = os.path.join(pkg_dir, "urdf", "ar4_dual.urdf")
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                # No namespace: robot_description at /robot_description
                output="screen",
                parameters=[
                    {"robot_description": robot_description},
                    {"use_sim_time": False},
                ],
            ),
        ]
    )
