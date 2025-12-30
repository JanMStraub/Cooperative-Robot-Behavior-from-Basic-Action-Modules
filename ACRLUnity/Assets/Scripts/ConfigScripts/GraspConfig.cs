using UnityEngine;
using Robotics;
using Robotics.Grasp;

namespace Configuration
{
    /// <summary>
    /// Settings for enabling/disabling specific grasp approach types.
    /// </summary>
    [System.Serializable]
    public class GraspApproachSettings
    {
        public GraspApproach approachType;
        public bool enabled = true;

        [Tooltip("Preference weight for this approach (higher = more preferred)")]
        [Range(0f, 2f)]
        public float preferenceWeight = 1.0f;

        public GraspApproachSettings(GraspApproach type, bool isEnabled = true, float weight = 1.0f)
        {
            approachType = type;
            enabled = isEnabled;
            preferenceWeight = weight;
        }
    }

    /// <summary>
    /// Configuration for MoveIt2-inspired grasp planning pipeline.
    /// Controls candidate generation, filtering, scoring, and execution.
    /// </summary>
    [CreateAssetMenu(fileName = "GraspConfig", menuName = "Robotics/Grasp Config")]
    public class GraspConfig : ScriptableObject
    {
        [Header("Candidate Generation")]
        [Tooltip("Number of candidate poses to generate per approach type")]
        [Range(1, 20)]
        public int candidatesPerApproach = 5;

        [Tooltip("Enabled approach directions with preference weights")]
        public GraspApproachSettings[] enabledApproaches = new GraspApproachSettings[]
        {
            new GraspApproachSettings(GraspApproach.Top, true, 1.2f),
            new GraspApproachSettings(GraspApproach.Front, true, 1.0f),
            new GraspApproachSettings(GraspApproach.Side, true, 0.8f)
        };

        [Header("Approach Distances")]
        [Tooltip("Multiplier for object size to determine pre-grasp distance")]
        [Range(1.0f, 3.0f)]
        public float preGraspDistanceFactor = 1.5f;

        [Tooltip("Minimum pre-grasp distance (meters)")]
        [Range(0.01f, 0.2f)]
        public float minPreGraspDistance = 0.05f;

        [Tooltip("Maximum pre-grasp distance (meters)")]
        [Range(0.05f, 0.3f)]
        public float maxPreGraspDistance = 0.15f;

        [Header("Retreat Motion")]
        [Tooltip("Enable post-grasp retreat motion")]
        public bool enableRetreat = true;

        [Tooltip("Multiplier for object size to determine retreat distance")]
        [Range(1.0f, 5.0f)]
        public float retreatDistanceFactor = 2.0f;

        [Tooltip("Direction for retreat motion (typically upward)")]
        public Vector3 retreatDirection = Vector3.up;

        [Header("Gripper Settings")]
        [Tooltip("Gripper geometry for validation")]
        public GripperGeometry gripperGeometry = GripperGeometry.Default();

        [Tooltip("Target grasp depth as fraction of object size (0-1)")]
        [Range(0.0f, 1.0f)]
        public float targetGraspDepth = 0.5f;

        [Header("Scoring Weights")]
        [Tooltip("Weight for IK quality score")]
        [Range(0f, 2f)]
        public float ikScoreWeight = 1.0f;

        [Tooltip("Weight for approach preference score")]
        [Range(0f, 2f)]
        public float approachScoreWeight = 0.8f;

        [Tooltip("Weight for grasp depth score")]
        [Range(0f, 2f)]
        public float depthScoreWeight = 0.6f;

        [Tooltip("Weight for stability score")]
        [Range(0f, 2f)]
        public float stabilityScoreWeight = 1.2f;

        [Header("Collision Checking")]
        [Tooltip("Enable collision checking along approach path")]
        public bool enableCollisionChecking = true;

        [Tooltip("Number of waypoints to check along approach path")]
        [Range(3, 20)]
        public int collisionCheckWaypoints = 5;

        [Tooltip("Radius of sphere for collision checking (meters)")]
        [Range(0.01f, 0.1f)]
        public float collisionCheckRadius = 0.03f;

        [Tooltip("Layer mask for collision checking")]
        public LayerMask collisionLayerMask = -1;

        [Header("IK Validation")]
        [Tooltip("Enable IK validation for candidate filtering")]
        public bool enableIKValidation = true;

        [Tooltip("Maximum IK iterations for validation")]
        [Range(5, 50)]
        public int maxIKValidationIterations = 10;

        [Tooltip("Distance threshold for IK convergence (meters)")]
        [Range(0.001f, 0.1f)]
        public float ikValidationThreshold = 0.01f;

        [Tooltip("Maximum reach distance for quick rejection (meters)")]
        [Range(0.3f, 1.0f)]
        public float maxReachDistance = 0.6f;

        [Header("Performance")]
        [Tooltip("Maximum time budget for full pipeline (milliseconds)")]
        [Range(50, 1000)]
        public int maxPipelineTimeMs = 200;

        /// <summary>
        /// Initialize with default AR4 grasp configuration.
        /// </summary>
        public void InitializeDefaultConfig()
        {
            candidatesPerApproach = 5;
            preGraspDistanceFactor = 1.5f;
            minPreGraspDistance = 0.05f;
            maxPreGraspDistance = 0.15f;
            enableRetreat = true;
            retreatDistanceFactor = 2.0f;
            retreatDirection = Vector3.up;
            gripperGeometry = GripperGeometry.Default();
            targetGraspDepth = 0.5f;

            ikScoreWeight = 1.0f;
            approachScoreWeight = 0.8f;
            depthScoreWeight = 0.6f;
            stabilityScoreWeight = 1.2f;

            enableCollisionChecking = true;
            collisionCheckWaypoints = 5;
            collisionCheckRadius = 0.03f;
            collisionLayerMask = -1;

            enableIKValidation = true;
            maxIKValidationIterations = 10;
            ikValidationThreshold = 0.01f;
            maxReachDistance = 0.6f;

            maxPipelineTimeMs = 200;

            enabledApproaches = new GraspApproachSettings[]
            {
                new GraspApproachSettings(GraspApproach.Top, true, 1.2f),
                new GraspApproachSettings(GraspApproach.Front, true, 1.0f),
                new GraspApproachSettings(GraspApproach.Side, true, 0.8f)
            };
        }

        /// <summary>
        /// Get preference weight for a specific approach type.
        /// </summary>
        /// <param name="approach">Approach type to query</param>
        /// <returns>Preference weight (0 if disabled)</returns>
        public float GetApproachWeight(GraspApproach approach)
        {
            foreach (var setting in enabledApproaches)
            {
                if (setting.approachType == approach && setting.enabled)
                {
                    return setting.preferenceWeight;
                }
            }
            return 0f;
        }

        /// <summary>
        /// Check if an approach type is enabled.
        /// </summary>
        /// <param name="approach">Approach type to check</param>
        /// <returns>True if enabled</returns>
        public bool IsApproachEnabled(GraspApproach approach)
        {
            foreach (var setting in enabledApproaches)
            {
                if (setting.approachType == approach)
                {
                    return setting.enabled;
                }
            }
            return false;
        }

        /// <summary>
        /// Calculate adaptive pre-grasp distance based on object size.
        /// </summary>
        /// <param name="objectSize">Size of target object</param>
        /// <returns>Pre-grasp distance clamped to min/max bounds</returns>
        public float CalculatePreGraspDistance(Vector3 objectSize)
        {
            float avgSize = (objectSize.x + objectSize.y + objectSize.z) / 3f;
            float distance = avgSize * preGraspDistanceFactor;
            return Mathf.Clamp(distance, minPreGraspDistance, maxPreGraspDistance);
        }

        /// <summary>
        /// Calculate retreat distance based on object size.
        /// </summary>
        /// <param name="objectSize">Size of target object</param>
        /// <returns>Retreat distance</returns>
        public float CalculateRetreatDistance(Vector3 objectSize)
        {
            float avgSize = (objectSize.x + objectSize.y + objectSize.z) / 3f;
            return avgSize * retreatDistanceFactor;
        }
    }
}
