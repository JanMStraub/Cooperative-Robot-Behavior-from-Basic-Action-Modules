using System.Collections.Generic;
using UnityEngine;

namespace Logging
{
    public enum ActionType
    {
        Task, // High-level task
        Movement, // Robot movement
        Manipulation, // Gripper/object manipulation
        Coordination, // Multi-robot coordination
        Observation, // Sensing/detection
    }

    public enum ActionStatus
    {
        Started,
        InProgress,
        Completed,
        Failed,
    }

    /// <summary>
    /// Unified action that can represent tasks, operations, or events
    /// Replaces: TaskContext, SemanticOperation, CoordinationEvent
    /// </summary>
    [System.Serializable]
    public class RobotAction
    {
        // Core identification
        public string actionId;
        public string actionName;
        public string description;
        public ActionType type;
        public ActionStatus status;

        // Participants
        public string[] robotIds;
        public string[] objectIds;

        // Timing
        public string timestamp;
        public float gameTime;
        public float duration;

        // Spatial data
        public Vector3 startPosition;
        public Vector3 targetPosition;
        public Vector3[] trajectoryPoints; // Optional:  trajectory

        // Outcomes and metrics
        public bool success;
        public string errorMessage;
        public float qualityScore; // 0-1 rating
        public Dictionary<string, float> metrics;

        // LLM training essentials
        public string humanReadable; // Natural language description
        public string[] capabilities; // Required skills: ["movement", "manipulation"]
        public int complexityLevel; // 1-4: simple to expert

        // Hierarchical relationships
        public string parentActionId; // For subtasks
        public string[] childActionIds; // Child operations

        public RobotAction()
        {
            robotIds = new string[0];
            objectIds = new string[0];
            trajectoryPoints = new Vector3[0];
            metrics = new Dictionary<string, float>();
            capabilities = new string[0];
            childActionIds = new string[0];
        }
    }

    /// <summary>
    ///  environment snapshot
    /// Replaces: EnvironmentState, DetectedObject, RobotState
    /// </summary>
    [System.Serializable]
    public class SceneSnapshot
    {
        public string snapshotId;
        public string timestamp;
        public float gameTime;

        // Objects in scene
        public Object[] objects;

        // Robot states
        public RobotState[] robots;

        // Scene metadata
        public int totalObjects;
        public int graspableObjects;
        public string sceneDescription; // Human-readable scene summary

        public SceneSnapshot()
        {
            objects = new Object[0];
            robots = new RobotState[0];
        }
    }

    [System.Serializable]
    public class Object
    {
        public string id;
        public string name;
        public string type; // "cube", "sphere", "target", etc.
        public Vector3 position;
        public Quaternion rotation;
        public bool isGraspable;
        public bool isMovable;
        public float mass;
    }

    [System.Serializable]
    public class RobotState
    {
        public string robotId;
        public Vector3 position; // End effector position
        public Quaternion rotation; // End effector rotation
        public float[] jointAngles; // Current joint configuration
        public Vector3 targetPosition; // Where robot is moving to
        public float distanceToTarget;
        public bool isMoving;
        public string currentAction; // What robot is currently doing

        public RobotState()
        {
            jointAngles = new float[0];
        }
    }

    /// <summary>
    /// Final log entry combining action and environment
    /// Replaces: EnhancedLogEntry, LearningMetadata
    /// </summary>
    [System.Serializable]
    public class LogEntry
    {
        public string logId;
        public string timestamp;
        public float gameTime;
        public string logType; // "action", "scene", "session"

        // Core data
        public RobotAction action;
        public SceneSnapshot scene;

        // LLM training metadata
        public string trainingPrompt; // "Task: Move robot to target..."
        public string trainingResponse; // "Robot successfully moved..."
        public string[] learningPoints; // Key takeaways for LLM
        public string difficultyLevel; // "simple", "moderate", "complex"

        public LogEntry()
        {
            learningPoints = new string[0];
        }
    }
}
