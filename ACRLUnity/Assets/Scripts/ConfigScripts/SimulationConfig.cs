using UnityEngine;

namespace Configuration
{
    // All coordination is Python-driven via signal/wait operations.
    [CreateAssetMenu(fileName = "SimulationConfig", menuName = "Robotics/Simulation Config")]
    public class SimulationConfig : ScriptableObject
    {
        [Header("Simulation Settings")]
        [Tooltip("Speed multiplier for the entire simulation")]
        public float timeScale = 1f;

        [Tooltip("Start simulation automatically when scene loads")]
        public bool autoStart = false;

        [Tooltip("Reset simulation when an error occurs")]
        public bool resetOnError = true;

        [Header("Performance")]
        [Range(10, 120)]
        [Tooltip("Target FPS for the simulation")]
        public int targetFrameRate = 30;

        [Tooltip("Enable vertical sync to prevent screen tearing")]
        public bool enableVSync = true;

#if UNITY_EDITOR
        /// <summary>
        /// Validate configuration values.
        /// </summary>
        private void OnValidate()
        {
            timeScale = Mathf.Max(0.1f, timeScale);
            targetFrameRate = Mathf.Clamp(targetFrameRate, 10, 120);
        }
#endif
    }
}
