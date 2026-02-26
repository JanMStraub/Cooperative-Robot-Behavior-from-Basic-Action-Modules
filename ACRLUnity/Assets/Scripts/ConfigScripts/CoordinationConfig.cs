using UnityEngine;

namespace Configuration
{
    /// <summary>
    /// Verification mode for coordination checks.
    /// </summary>
    public enum VerificationMode
    {
        /// <summary>
        /// Fast Unity-only verification without Python backend.
        /// Best for performance-critical scenarios.
        /// </summary>
        UnityOnly,

        /// <summary>
        /// Use Python CoordinationVerifier for verification.
        /// Slower but more accurate with workspace state tracking.
        /// </summary>
        PythonVerified,

        /// <summary>
        /// Hybrid mode: Unity verification first, Python verification on conflicts.
        /// Balances performance and accuracy.
        /// </summary>
        Hybrid,
    }

    /// <summary>
    /// Configuration for robot coordination verification.
    /// Controls collision detection, path replanning, and Python verification integration.
    /// </summary>
    [CreateAssetMenu(fileName = "CoordinationConfig", menuName = "Robotics/Coordination Config")]
    public class CoordinationConfig : ScriptableObject
    {
        [Header("Verification Settings")]
        [Tooltip("Verification mode for coordination checks")]
        public VerificationMode verificationMode = VerificationMode.UnityOnly;

        [Tooltip("Fallback to Unity verification if Python times out")]
        public bool fallbackToUnityOnTimeout = true;

        [Tooltip("Timeout for Python verification in seconds")]
        [Range(0.1f, 5f)]
        public float pythonVerificationTimeout = 1f;

        [Header("Collision Detection")]
        [Tooltip("Minimum safe separation distance between robots in meters")]
        [Range(0.05f, 1f)]
        public float minSafeSeparation = 0.2f;

        [Tooltip("Enable path replanning for collision avoidance")]
        public bool enablePathReplanning = true;

        [Header("Timeout Settings")]
        [Tooltip("Timeout in seconds before switching to next robot (Sequential mode)")]
        [Range(5f, 120f)]
        public float robotTimeout = 30f;

        [Header("Path Replanning Parameters")]
        [Tooltip("Vertical offset for waypoint planning in meters")]
        [Range(0.05f, 0.5f)]
        public float verticalOffset = 0.15f;

        [Tooltip("Lateral offset for waypoint planning in meters")]
        [Range(0.05f, 0.3f)]
        public float lateralOffset = 0.1f;

        [Tooltip("Maximum number of waypoints per replanned path")]
        [Range(2, 10)]
        public int maxWaypoints = 5;

#if UNITY_EDITOR
        /// <summary>
        /// Validate configuration values.
        /// </summary>
        private void OnValidate()
        {
            minSafeSeparation = Mathf.Clamp(minSafeSeparation, 0.05f, 1f);
            robotTimeout = Mathf.Clamp(robotTimeout, 5f, 120f);
            pythonVerificationTimeout = Mathf.Clamp(pythonVerificationTimeout, 0.1f, 5f);
            verticalOffset = Mathf.Clamp(verticalOffset, 0.05f, 0.5f);
            lateralOffset = Mathf.Clamp(lateralOffset, 0.05f, 0.3f);
            maxWaypoints = Mathf.Clamp(maxWaypoints, 2, 10);
        }
#endif
    }
}
