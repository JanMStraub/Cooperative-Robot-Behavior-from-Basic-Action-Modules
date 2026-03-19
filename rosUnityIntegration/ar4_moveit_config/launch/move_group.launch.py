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
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
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

    # MoveIt configuration (planning only)
    # kinematics.yaml, joint_limits.yaml, and ompl_planning.yaml all use /**:
    # so they apply correctly under any robot namespace (e.g. /Robot1, /Robot2).
    # Planning plugin/adapters are inlined below (scalars only) because the
    # move_group: namespace prefix in YAML would create a nested sub-key that
    # the namespaced node cannot find. Nested planner_configs dicts come from
    # ompl_planning.yaml since ROS 2 launch params only support scalar values.
    kinematics_yaml = os.path.join(pkg_dir, "config", "kinematics.yaml")
    joint_limits_yaml = os.path.join(pkg_dir, "config", "joint_limits.yaml")
    ompl_planning_yaml = os.path.join(pkg_dir, "config", "ompl_planning.yaml")

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
                joint_limits_yaml,
                ompl_planning_yaml,
                {
                    # planning_pipelines is a string_array — it must come from the YAML
                    # file below (ompl_planning.yaml), not from this inline dict.
                    "default_planning_pipeline": "ompl",
                    # Plugin and adapters declared under the pipeline name prefix.
                    "ompl.planning_plugin": "ompl_interface/OMPLPlanner",
                    # This Humble build uses the older single-adapter-list API:
                    # prefix is "default_planner_request_adapters/" (not
                    # "default_planning_request_adapters/"), and there is no separate
                    # response_adapters param — AddTimeOptimalParameterization runs as
                    # a post-planning request adapter (last in the list).
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

                    # --- Planning scene monitor ---
                    # Subscribe to joint states from Unity's ROSJointStatePublisher
                    "planning_scene_monitor_options.joint_state_topic": "joint_states",
                    "planning_scene_monitor_options.attached_collision_object_topic": "/planning_scene",
                    "planning_scene_monitor_options.publish_planning_scene_topic": "/planning_scene",
                    "planning_scene_monitor_options.monitored_planning_scene_topic": "/planning_scene_monitored",
                    # Don't wait for initial state - allow MoveIt to start before Unity
                    "planning_scene_monitor_options.wait_for_initial_state_timeout": 0.0,

                    # --- General ---
                    "use_sim_time": False,
                    "publish_robot_description": True,
                    "publish_robot_description_semantic": True,

                    # Capabilities to load (minimal set for plan-only)
                    "capabilities": (
                        "move_group/MoveGroupCartesianPathService "
                        "move_group/MoveGroupKinematicsService "
                        "move_group/MoveGroupMoveAction "
                        "move_group/MoveGroupPlanService"
                    ),

                    # No controller manager - plan-only mode; trajectories go directly to Unity.
                    # Set controller_names explicitly to suppress the "No controller_names
                    # specified" ERROR from moveit_simple_controller_manager at startup.
                    # Empty string is the only scalar representation of an empty list
                    # accepted by the ROS 2 launch parameter API.
                    "moveit_simple_controller_manager.controller_names": "",

                    # Explicitly declare no 3D sensors so MoveIt does not attempt to
                    # load an octomap sensor plugin and log an ERROR on startup.
                    "octomap_resolution": 0.1,
                    "max_range": 5.0,
                },
            ],
        ),
    ])
