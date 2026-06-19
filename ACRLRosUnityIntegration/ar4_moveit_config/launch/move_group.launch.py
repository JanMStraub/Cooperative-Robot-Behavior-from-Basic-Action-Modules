"""Launch MoveIt 2 move_group node for AR4 mk3 (Plan-Only mode).

This launch file configures move_group for PLANNING ONLY.
We do not load controller configurations because execution is handled
by publishing planned trajectories directly to Unity via ROS topics.
MoveIt's FollowJointTrajectory execution pipeline is not used.

Parameterized by robot_id for multi-robot support.
"""

import os
from launch import LaunchDescription # type: ignore[import-not-found]
from launch.actions import DeclareLaunchArgument # type: ignore[import-not-found]
from launch.substitutions import LaunchConfiguration # type: ignore[import-not-found]
from launch_ros.actions import Node # type: ignore[import-not-found]
from ament_index_python.packages import get_package_share_directory # type: ignore[import-not-found]


def generate_launch_description():
    pkg_dir = get_package_share_directory("ar4_moveit_config")

    robot_id_arg = DeclareLaunchArgument(
        "robot_id",
        default_value="Robot1",
        description="Robot namespace for topic routing (e.g., Robot1, Robot2)",
    )

    robot_id = LaunchConfiguration("robot_id")

    urdf_path = os.path.join(pkg_dir, "urdf", "ar4.urdf")
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    srdf_path = os.path.join(pkg_dir, "config", "ar4.srdf")
    with open(srdf_path, "r") as f:
        robot_description_semantic = f.read()

    # Planning plugin/adapters inlined as scalars - YAML namespace prefix breaks nested keys
    kinematics_yaml = os.path.join(pkg_dir, "config", "kinematics.yaml")
    joint_limits_yaml = os.path.join(pkg_dir, "config", "joint_limits.yaml")
    ompl_planning_yaml = os.path.join(pkg_dir, "config", "ompl_planning.yaml")
    moveit_controllers_yaml = os.path.join(pkg_dir, "config", "moveit_controllers.yaml")

    return LaunchDescription(
        [
            robot_id_arg,
            # Move Group node (plan-only: no controller manager loaded)
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                namespace=robot_id,
                output="screen",
                parameters=[
                    {"robot_description": robot_description},
                    {"robot_description_semantic": robot_description_semantic},
                    kinematics_yaml,
                    joint_limits_yaml,
                    ompl_planning_yaml,
                    moveit_controllers_yaml,
                    {
                        "default_planning_pipeline": "ompl",
                        "ompl.planning_plugin": "ompl_interface/OMPLPlanner",
                        # Humble uses single-adapter-list API: prefix is
                        # "default_planner_request_adapters/" (not "default_planning_request_adapters/"),
                        # no separate response_adapters param.
                        "ompl.request_adapters": (
                            "default_planner_request_adapters/ResolveConstraintFrames "
                            "default_planner_request_adapters/FixWorkspaceBounds "
                            "default_planner_request_adapters/FixStartStateBounds "
                            "default_planner_request_adapters/FixStartStateCollision "
                            "default_planner_request_adapters/FixStartStatePathConstraints "
                            "default_planner_request_adapters/AddTimeOptimalParameterization"
                        ),
                        # Tolerance for clamping start state joint positions within URDF limits.
                        "ompl.start_state_max_bounds_error": 0.5,
                        "planning_scene_monitor_options.joint_state_topic": "joint_states",
                        "planning_scene_monitor_options.attached_collision_object_topic": "/planning_scene",
                        "planning_scene_monitor_options.publish_planning_scene_topic": "/planning_scene",
                        "planning_scene_monitor_options.monitored_planning_scene_topic": "/planning_scene_monitored",
                        # Allow MoveIt to start before Unity sends joint states
                        "planning_scene_monitor_options.wait_for_initial_state_timeout": 0.0,
                        "use_sim_time": False,
                        "publish_robot_description": True,
                        "publish_robot_description_semantic": True,
                        "capabilities": (
                            "move_group/MoveGroupCartesianPathService "
                            "move_group/MoveGroupKinematicsService "
                            "move_group/MoveGroupMoveAction "
                            "move_group/MoveGroupPlanService"
                        ),
                        # controller_names/moveit_controller_manager need string array - set
                        # via moveit_controllers.yaml; octomap params suppress sensor plugin ERROR.
                        "octomap_resolution": 0.1,
                        "max_range": 5.0,
                    },
                ],
            ),
        ]
    )
