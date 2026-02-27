using System;
using System.Collections.Generic;

namespace PythonCommunication.DataModels
{
    /// <summary>
    /// Data models for AutoRT (Autonomous Robot Task generation) integration.
    /// Matches Python AutoRTIntegration.py serialization format.
    /// </summary>

    [Serializable]
    public class TaskOperation
    {
        public string type;
        public string robot_id;
        public Dictionary<string, object> parameters;

        public TaskOperation()
        {
            parameters = new Dictionary<string, object>();
        }
    }

    [Serializable]
    public class ProposedTask
    {
        public string task_id;
        public string description;
        public List<TaskOperation> operations;
        public List<string> required_robots;
        public int estimated_complexity;
        public string reasoning;

        public ProposedTask()
        {
            operations = new List<TaskOperation>();
            required_robots = new List<string>();
        }

        /// <summary>
        /// Get a display-friendly summary of the task.
        /// </summary>
        public string GetSummary()
        {
            return $"[Complexity {estimated_complexity}] {description}";
        }

        /// <summary>
        /// Get robot count required for this task.
        /// </summary>
        public int RobotCount
        {
            get { return required_robots?.Count ?? 0; }
        }

        /// <summary>
        /// Get operation count for this task.
        /// </summary>
        public int OperationCount
        {
            get { return operations?.Count ?? 0; }
        }
    }

    [Serializable]
    public class AutoRTResponse
    {
        public bool success;
        public List<ProposedTask> tasks;
        public bool loop_running;
        public string error;
        public string status;  // Execution status: "started", "not_found", "error", etc.
        public uint request_id;

        public AutoRTResponse()
        {
            tasks = new List<ProposedTask>();
        }

        /// <summary>
        /// Check if the response indicates an error state.
        /// </summary>
        public bool HasError
        {
            get { return !success || !string.IsNullOrEmpty(error); }
        }

        /// <summary>
        /// Get error message or "No error" if successful.
        /// </summary>
        public string ErrorMessage
        {
            get { return HasError ? error : "No error"; }
        }
    }

    [Serializable]
    public class AutoRTLoopConfig
    {
        public float delay;
        public List<string> robot_ids;
        public string strategy;

        public AutoRTLoopConfig()
        {
            robot_ids = new List<string>();
        }
    }

    [Serializable]
    public class AutoRTStatus
    {
        public bool success;
        public bool loop_running;
        public int pending_tasks_count;
        public AutoRTLoopConfig loop_config;
        public string error;

        public AutoRTStatus()
        {
            loop_config = new AutoRTLoopConfig();
        }
    }

    /// <summary>
    /// Task selection strategies for AutoRT generation.
    /// </summary>
    public enum TaskSelectionStrategy
    {
        Balanced,
        Explore,
        Exploit,
        Random,
    }
}
