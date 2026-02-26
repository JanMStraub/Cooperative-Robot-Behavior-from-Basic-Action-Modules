using UnityEngine;

namespace Configuration
{
    /// <summary>
    /// Configuration for the IK solver: convergence, timeouts, motion limits, and object detection.
    /// </summary>
    [CreateAssetMenu(fileName = "IKConfig", menuName = "Robotics/IK Config")]
    public class IKConfig : ScriptableObject
    {
        [Header("IK Convergence")]
        [Range(0.001f, 0.1f)]
        public float convergenceThreshold = 0.02f;

        [Range(0.1f, 0.5f)]
        public float dampingFactor = 0.2f;

        [Range(0.02f, 0.3f)]
        public float maxJointStepRad = 0.2f;

        [Range(0.1f, 1f)]
        public float adjustmentSpeed = 0.5f;

        [Header("Orientation Control")]
        [Range(5f, 45f)]
        public float orientationThresholdDegrees = 10f;

        [Range(0.1f, 1f)]
        public float orientationRampStartDistance = 0.30f;

        [Header("Timeouts")]
        [Range(5f, 60f)]
        public float graspTimeoutSeconds = 30f;

        [Range(5f, 60f)]
        public float movementTimeoutSeconds = 15f;

        [Header("Motion Limits")]
        [Range(0.1f, 1f)]
        public float maxVelocity = 0.2f;

        [Range(0.3f, 2f)]
        public float maxAcceleration = 0.7f;

        [Header("Convergence Multipliers")]
        [Range(0.1f, 1f)]
        public float graspConvergenceMultiplier = 0.33f;

        [Range(1f, 5f)]
        public float preGraspConvergenceMultiplier = 2.0f;

        [Header("Advanced IK Solver Parameters")]
        [Range(1f, 10f)]
        [Tooltip("Maximum joint velocity in rad/sec to prevent singularity spikes")]
        public float maxJointVelocity = 5.0f;

        [Range(0.1f, 5f)]
        [Tooltip("Maximum error magnitude to prevent matrix instability (meters)")]
        public float maxErrorMagnitude = 1.0f;

        [Header("Object Detection")]
        [Range(0.05f, 0.5f)]
        [Tooltip("Search radius for finding real objects when setting target by coordinate")]
        public float objectFindingRadius = 0.15f;

        [Range(0.05f, 0.3f)]
        [Tooltip("Distance threshold for matching coordinate to real object")]
        public float objectDistanceThreshold = 0.1f;

#if UNITY_EDITOR
        /// <summary>
        /// Validate configuration values to ensure consistency.
        /// </summary>
        private void OnValidate()
        {
            // IK convergence parameters
            convergenceThreshold = Mathf.Clamp(convergenceThreshold, 0.001f, 0.1f);
            dampingFactor = Mathf.Clamp(dampingFactor, 0.1f, 0.5f);
            maxJointStepRad = Mathf.Clamp(maxJointStepRad, 0.02f, 0.3f);
            adjustmentSpeed = Mathf.Clamp(adjustmentSpeed, 0.1f, 1f);

            // Orientation control
            orientationThresholdDegrees = Mathf.Clamp(orientationThresholdDegrees, 5f, 45f);
            orientationRampStartDistance = Mathf.Clamp(orientationRampStartDistance, 0.1f, 1f);

            // Timeouts
            graspTimeoutSeconds = Mathf.Clamp(graspTimeoutSeconds, 5f, 60f);
            movementTimeoutSeconds = Mathf.Clamp(movementTimeoutSeconds, 5f, 60f);

            // Motion limits
            maxVelocity = Mathf.Clamp(maxVelocity, 0.1f, 1f);
            maxAcceleration = Mathf.Clamp(maxAcceleration, 0.3f, 2f);

            // Convergence multipliers
            graspConvergenceMultiplier = Mathf.Clamp(graspConvergenceMultiplier, 0.1f, 1f);
            preGraspConvergenceMultiplier = Mathf.Clamp(preGraspConvergenceMultiplier, 1f, 5f);

            // Advanced IK solver parameters
            maxJointVelocity = Mathf.Clamp(maxJointVelocity, 1f, 10f);
            maxErrorMagnitude = Mathf.Clamp(maxErrorMagnitude, 0.1f, 5f);

            // Ensure convergence threshold < max error magnitude
            if (convergenceThreshold >= maxErrorMagnitude)
            {
                convergenceThreshold = maxErrorMagnitude * 0.1f;
            }

            // Object detection
            objectFindingRadius = Mathf.Clamp(objectFindingRadius, 0.05f, 0.5f);
            objectDistanceThreshold = Mathf.Clamp(objectDistanceThreshold, 0.05f, 0.3f);

            // Ensure distance threshold <= finding radius
            if (objectDistanceThreshold > objectFindingRadius)
            {
                objectDistanceThreshold = objectFindingRadius;
            }
        }
#endif
    }
}
