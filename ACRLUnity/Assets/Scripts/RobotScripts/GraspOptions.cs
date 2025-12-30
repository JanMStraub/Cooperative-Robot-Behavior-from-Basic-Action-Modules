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
        public bool useGraspPlanning; // Use GraspPlanner to adjust target pose
        public bool openGripperOnSet; // Open gripper when setting target
        public bool closeGripperOnReach; // Close gripper when target reached
        public GraspApproach? approach; // Optional override for grasp approach (null = auto-determine)

        // Advanced planning options (MoveIt2-inspired pipeline)
        public bool useAdvancedPlanning; // Use full GraspPlanningPipeline
        public GraspConfig graspConfig; // Configuration for advanced planning (null = use default)
        public float overridePreGraspDistance; // Custom pre-grasp distance (0 = use config default)
        public Vector3? customApproachVector; // Custom approach direction (null = use approach type)

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
