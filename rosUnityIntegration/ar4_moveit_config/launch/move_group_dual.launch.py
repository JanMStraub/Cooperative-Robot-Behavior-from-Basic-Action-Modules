"""Launch MoveIt 2 move_group node for dual AR4 mk3 (shared planning scene, plan-only).

Single move_group instance with both robots in one planning scene.
Planning groups: robot1_arm, robot2_arm.
Requires joint_state_aggregator and robot_state_publisher_dual to be running first.
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

    srdf_path = os.path.join(pkg_dir, "config", "ar4_dual.srdf")
    with open(srdf_path, "r") as f:
        robot_description_semantic = f.read()

    kinematics_yaml = os.path.join(pkg_dir, "config", "kinematics_dual.yaml")
    joint_limits_yaml = os.path.join(pkg_dir, "config", "joint_limits.yaml")
    ompl_planning_yaml = os.path.join(pkg_dir, "config", "ompl_planning.yaml")
    moveit_controllers_yaml = os.path.join(pkg_dir, "config", "moveit_controllers.yaml")

    return LaunchDescription(
        [
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                # No namespace: action server at /move_action (not /{robot_id}/move_action)
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
                        "ompl.request_adapters": (
                            "default_planner_request_adapters/ResolveConstraintFrames "
                            "default_planner_request_adapters/FixWorkspaceBounds "
                            "default_planner_request_adapters/FixStartStateBounds "
                            "default_planner_request_adapters/FixStartStateCollision "
                            "default_planner_request_adapters/FixStartStatePathConstraints "
                            "default_planner_request_adapters/AddTimeOptimalParameterization"
                        ),
                        "ompl.start_state_max_bounds_error": 0.5,
                        # Aggregated joint states topic (all 16 joints, prefixed)
                        "planning_scene_monitor_options.joint_state_topic": "/joint_states",
                        "planning_scene_monitor_options.attached_collision_object_topic": "/planning_scene",
                        "planning_scene_monitor_options.publish_planning_scene_topic": "/planning_scene",
                        "planning_scene_monitor_options.monitored_planning_scene_topic": "/planning_scene_monitored",
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
                        "octomap_resolution": 0.1,
                        "max_range": 5.0,
                    },
                ],
            ),
        ]
    )
