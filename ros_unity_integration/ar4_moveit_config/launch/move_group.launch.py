"""Launch MoveIt 2 move_group node for AR4 mk3 (Plan-Only mode).

This launch file configures move_group for PLANNING ONLY.
We do not load controller configurations because execution is handled
by publishing planned trajectories directly to Unity via ROS topics.
MoveIt's FollowJointTrajectory execution pipeline is not used.

Parameterized by robot_id for multi-robot support.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for MoveIt move_group (plan-only).

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

    # Load URDF
    urdf_path = os.path.join(pkg_dir, "urdf", "ar4.urdf")
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    # Load SRDF
    srdf_path = os.path.join(pkg_dir, "config", "ar4.srdf")
    with open(srdf_path, "r") as f:
        robot_description_semantic = f.read()

    # MoveIt configuration (planning only - no controller config)
    kinematics_yaml = os.path.join(pkg_dir, "config", "kinematics.yaml")
    ompl_planning_yaml = os.path.join(pkg_dir, "config", "ompl_planning.yaml")
    joint_limits_yaml = os.path.join(pkg_dir, "config", "joint_limits.yaml")

    return LaunchDescription([
        robot_id_arg,
        # Move Group node (plan-only: no controller manager loaded)
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            namespace=robot_id,  # Namespace all topics with robot ID
            output="screen",
            parameters=[
                {"robot_description": robot_description},
                {"robot_description_semantic": robot_description_semantic},
                kinematics_yaml,
                ompl_planning_yaml,
                joint_limits_yaml,
                {
                    "planning_scene_monitor_options": {
                        # Subscribe to joint states from Unity's ROSJointStatePublisher
                        # Using relative topic (joint_states) within robot namespace
                        "joint_state_topic": "joint_states",
                        "attached_collision_object_topic": "/planning_scene",
                        "publish_planning_scene_topic": "/planning_scene",
                        "monitored_planning_scene_topic": "/planning_scene_monitored",
                        # Don't wait for initial state at startup - accept whenever Unity connects
                        # This allows MoveIt to start before Unity is running
                        "wait_for_initial_state_timeout": 0.0,
                    },
                    "use_sim_time": False,
                    "publish_robot_description": True,
                    "publish_robot_description_semantic": True,
                    # Force OMPL planner for move_group pipeline (supports pose goals, unlike CHOMP)
                    "move_group": {
                        "planning_plugin": "ompl_interface/OMPLPlanner",
                    },
                    # Capabilities to load (minimal set for plan-only)
                    "capabilities": "move_group/MoveGroupCartesianPathService "
                                     "move_group/MoveGroupKinematicsService "
                                     "move_group/MoveGroupMoveAction "
                                     "move_group/MoveGroupPlanService",
                    # No controller manager - we use plan-only mode.
                    # Trajectories are published directly to Unity via topic.
                },
            ],
        ),
    ])
