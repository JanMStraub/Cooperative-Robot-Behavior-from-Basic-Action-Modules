using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Configuration options for grasp planning and gripper control
    /// </summary>
    public struct GraspOptions
    {
        public bool useGraspPlanning; // Use GraspPlanner to adjust target pose
        public bool openGripperOnSet; // Open gripper when setting target
        public bool closeGripperOnReach; // Close gripper when target reached
        public GraspApproach? approach; // Optional override for grasp approach (null = auto-determine)

        /// <summary>
        /// Default: intelligent grasping enabled
        /// </summary>
        public static GraspOptions Default =>
            new()
            {
                useGraspPlanning = true,
                openGripperOnSet = true,
                closeGripperOnReach = true,
                approach = null,
            };

        /// <summary>
        /// Preset: just move, no gripper
        /// </summary>
        public static GraspOptions MoveOnly =>
            new()
            {
                useGraspPlanning = false,
                openGripperOnSet = false,
                closeGripperOnReach = false,
                approach = null,
            };
    }
}
