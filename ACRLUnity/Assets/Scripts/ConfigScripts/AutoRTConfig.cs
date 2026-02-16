using PythonCommunication.DataModels;
using UnityEngine;

namespace ConfigScripts
{
    /// <summary>
    /// Configuration for AutoRT (Autonomous Robot Task generation).
    /// Stores settings for task generation, continuous loop mode, and robot assignment.
    /// </summary>
    [CreateAssetMenu(fileName = "AutoRTConfig", menuName = "Robotics/AutoRT Config")]
    public class AutoRTConfig : ScriptableObject
    {
        [Header("Task Generation")]
        [Tooltip("Maximum number of task candidates to generate per request")]
        [Range(1, 5)]
        public int maxTaskCandidates = 3;

        [Tooltip("Task selection strategy")]
        public TaskSelectionStrategy strategy = TaskSelectionStrategy.Balanced;

        [Header("Continuous Loop")]
        [Tooltip("Enable continuous task generation loop (toggleable at runtime)")]
        public bool enableContinuousLoop = false;

        [Tooltip("Delay between task generations in continuous loop mode (seconds)")]
        [Range(1f, 60f)]
        public float loopDelaySeconds = 5f;

        [Header("Robot Assignment")]
        [Tooltip("Robot IDs to use for task generation (e.g., Robot1, Robot2)")]
        public string[] robotIds = new[] { "Robot1", "Robot2" };

        [Tooltip("Enable collaborative tasks requiring multiple robots")]
        public bool enableCollaborativeTasks = true;

        [Header("UI Settings")]
        [Tooltip("Maximum number of tasks to display in inspector")]
        [Range(5, 20)]
        public int maxDisplayTasks = 10;

        [Tooltip("Auto-refresh task list in play mode")]
        public bool autoRefresh = true;

        [Tooltip("UI refresh rate (seconds)")]
        [Range(0.1f, 2f)]
        public float uiRefreshRate = 0.5f;

        /// <summary>
        /// Validate configuration settings.
        /// </summary>
        public void OnValidate()
        {
            // Ensure at least one robot is configured
            if (robotIds == null || robotIds.Length == 0)
            {
                Debug.LogWarning("[AutoRTConfig] No robots configured. Adding default Robot1.");
                robotIds = new[] { "Robot1" };
            }

            // Validate loop delay
            if (loopDelaySeconds < 1f)
            {
                Debug.LogWarning("[AutoRTConfig] Loop delay too short, setting to 1 second.");
                loopDelaySeconds = 1f;
            }
        }

        /// <summary>
        /// Get robot IDs as comma-separated string.
        /// </summary>
        public string GetRobotIdsString()
        {
            return robotIds != null ? string.Join(", ", robotIds) : "None";
        }
    }
}
