using UnityEngine;

namespace Configuration
{
    /// <summary>
    /// Configuration for trajectory controller PD gains.
    /// Configurable via Unity Inspector for runtime tuning.
    /// </summary>
    [CreateAssetMenu(fileName = "TrajectoryConfig", menuName = "Robotics/Trajectory Config")]
    public class TrajectoryConfig : ScriptableObject
    {
        [Header("PD Control Gains")]
        [Tooltip("Position gains (Kp) for trajectory tracking (XYZ)")]
        public Vector3 positionGains = new Vector3(10f, 10f, 10f);

        [Tooltip("Velocity gains (Kd) for damping (XYZ)")]
        public Vector3 velocityGains = new Vector3(2f, 2f, 2f);

        [Header("Motion Limits")]
        [Range(0.1f, 1f)]
        [Tooltip("Maximum velocity for trajectory generation (m/s)")]
        public float maxVelocity = 0.5f;

        [Range(0.3f, 2f)]
        [Tooltip("Maximum acceleration for trajectory generation (m/s²)")]
        public float maxAcceleration = 1.0f;

        /// <summary>
        /// Validate configuration values to ensure consistency.
        /// </summary>
        private void OnValidate()
        {
            // Ensure all gain components are positive
            positionGains.x = Mathf.Max(0.1f, positionGains.x);
            positionGains.y = Mathf.Max(0.1f, positionGains.y);
            positionGains.z = Mathf.Max(0.1f, positionGains.z);

            velocityGains.x = Mathf.Max(0.1f, velocityGains.x);
            velocityGains.y = Mathf.Max(0.1f, velocityGains.y);
            velocityGains.z = Mathf.Max(0.1f, velocityGains.z);

            // Motion limits
            maxVelocity = Mathf.Clamp(maxVelocity, 0.1f, 1f);
            maxAcceleration = Mathf.Clamp(maxAcceleration, 0.3f, 2f);
        }
    }
}
