using UnityEngine;

namespace Configuration
{
    [System.Serializable]
    public class JointConfiguration
    {
        [Header("Joint Parameters")]
        public float stiffness = 5000f;
        public float damping = 1500f;
        public float forceLimit = 2000f;
        public float upperLimit = 170f;
        public float lowerLimit = -170f;

        public JointConfiguration(float stiff, float damp, float force, float upper, float lower)
        {
            stiffness = stiff;
            damping = damp;
            forceLimit = force;
            upperLimit = upper;
            lowerLimit = lower;
        }
    }

    [CreateAssetMenu(fileName = "RobotProfile", menuName = "Robotics/RobotProfile")]
    public class RobotConfig : ScriptableObject
    {
        [Header("Robot Identity")]
        public string profileName = "AR4_Default";
        public string description = "Standard AR4 robotic arm configuration";

        [Header("Joint Configurations")]
        public JointConfiguration[] joints = new JointConfiguration[6];

        [Header("IK Settings")]
        [Range(0.1f, 1f)]
        public float adjustmentSpeed = 0.5f;

        public void InitializeDefaultAR4Profile()
        {
            joints = new JointConfiguration[6]
            {
                new JointConfiguration(5000f, 1500f, 2000f, 170f, -170f), // Base
                new JointConfiguration(4000f, 1200f, 2000f, 90f, -42.5f), // Shoulder
                new JointConfiguration(3000f, 800f, 1000f, 125f, -90f), // Elbow
                new JointConfiguration(2500f, 500f, 500f, 105f, -105f), // Wrist 1
                new JointConfiguration(2000f, 400f, 500f, 100f, -100f), // Wrist 2
                new JointConfiguration(1500f, 300f, 500f, 180f, -180f), // Wrist 3
            };
        }

        /// <summary>
        /// Validate configuration values to ensure consistency.
        /// </summary>
        private void OnValidate()
        {
            // Validate adjustment speed
            adjustmentSpeed = Mathf.Clamp(adjustmentSpeed, 0.1f, 1f);

            // Validate each joint configuration
            if (joints != null)
            {
                foreach (var joint in joints)
                {
                    if (joint != null)
                    {
                        // Ensure positive values
                        joint.stiffness = Mathf.Max(0f, joint.stiffness);
                        joint.damping = Mathf.Max(0f, joint.damping);
                        joint.forceLimit = Mathf.Max(0f, joint.forceLimit);

                        // Ensure lower < upper limit
                        if (joint.lowerLimit >= joint.upperLimit)
                        {
                            joint.upperLimit = joint.lowerLimit + 10f;
                        }
                    }
                }
            }
        }
    }

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
    }
}
