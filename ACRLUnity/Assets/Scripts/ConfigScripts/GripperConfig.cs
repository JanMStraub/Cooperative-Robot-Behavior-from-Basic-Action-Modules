using UnityEngine;

namespace Configuration
{
    /// <summary>
    /// Configuration for gripper control and contact sensing.
    /// Configurable via Unity Inspector for runtime tuning.
    /// </summary>
    [CreateAssetMenu(fileName = "GripperConfig", menuName = "Robotics/Gripper Config")]
    public class GripperConfig : ScriptableObject
    {
        [Header("Contact Detection")]
        [Range(3, 10)]
        [Tooltip("Number of frames to average force readings over (handles Unity physics noise)")]
        public int forceWindowSize = 5;

        [Range(0.01f, 1f)]
        [Tooltip("Minimum force threshold to register contact (Newtons)")]
        public float minForceThreshold = 0.1f;

        [Range(0.05f, 0.5f)]
        [Tooltip("Minimum contact duration to confirm stable grasp (seconds)")]
        public float minContactDuration = 0.1f;

        [Range(1f, 20f)]
        [Tooltip("Minimum grasp force for stable grip verification (Newtons)")]
        public float minGraspForce = 5f;

        [Header("Gripper Control")]
        [Range(0.1f, 2f)]
        [Tooltip("Smooth time for gripper position interpolation (seconds)")]
        public float smoothTime = 0.5f;

#if UNITY_EDITOR
        /// <summary>
        /// Validate configuration values to ensure consistency.
        /// </summary>
        private void OnValidate()
        {
            // Force window size
            forceWindowSize = Mathf.Clamp(forceWindowSize, 3, 10);

            // Force thresholds
            minForceThreshold = Mathf.Clamp(minForceThreshold, 0.01f, 1f);
            minGraspForce = Mathf.Clamp(minGraspForce, 1f, 20f);

            // Ensure minForceThreshold < minGraspForce
            if (minForceThreshold >= minGraspForce)
            {
                minForceThreshold = minGraspForce * 0.1f;
            }

            // Contact duration
            minContactDuration = Mathf.Clamp(minContactDuration, 0.05f, 0.5f);

            // Smooth time
            smoothTime = Mathf.Clamp(smoothTime, 0.1f, 2f);
        }
#endif
    }
}
