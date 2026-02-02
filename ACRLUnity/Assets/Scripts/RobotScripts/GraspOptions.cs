using Configuration;
using Robotics.Grasp;
using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Configuration options for grasp planning and gripper control
    /// </summary>
    public struct GraspOptions
    {
        public bool useGraspPlanning;
        public bool openGripperOnSet;
        public bool closeGripperOnReach;
        public GraspApproach? approach;

        public bool useAdvancedPlanning;
        public GraspConfig graspConfig;
        public float overridePreGraspDistance;
        public Vector3? customApproachVector;

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
                useAdvancedPlanning = false,
                graspConfig = null,
                overridePreGraspDistance = 0f,
                customApproachVector = null,
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
                useAdvancedPlanning = false,
                graspConfig = null,
                overridePreGraspDistance = 0f,
                customApproachVector = null,
            };

        /// <summary>
        /// Preset: Advanced planning with full MoveIt2-inspired pipeline
        /// </summary>
        public static GraspOptions Advanced =>
            new()
            {
                useGraspPlanning = true,
                openGripperOnSet = true,
                closeGripperOnReach = true,
                approach = null, // Auto-determine
                useAdvancedPlanning = true,
                graspConfig = null, // Use default config
                overridePreGraspDistance = 0f,
                customApproachVector = null,
            };
    }
}
