using UnityEngine;

namespace Configuration
{
    public enum RobotCoordinationMode
    {
        Independent, // Each robot works alone
        Collaborative, // Robots work together on shared tasks
        MasterSlave, // One robot leads, others follow
        Distributed, // Decentralized decision making
        Sequential, // Robots take turns
        Negotiated, // Python-side LLM negotiation determines coordination
    }

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

        [Header("Robot Coordination")]
        [Tooltip("How robots should coordinate with each other")]
        public RobotCoordinationMode coordinationMode = RobotCoordinationMode.Independent;

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
