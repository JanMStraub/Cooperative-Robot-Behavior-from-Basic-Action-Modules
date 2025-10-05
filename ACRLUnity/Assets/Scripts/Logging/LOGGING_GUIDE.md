# Robot Logging System - Usage Guide

This guide explains how to use the unified logging system in the `Logging/` folder for robot action tracking, LLM training data collection, and simulation debugging.

---

## Table of Contents
1. [System Overview](#system-overview)
2. [MainLogger - Core Logging Engine](#mainlogger---core-logging-engine)
3. [AutoLogger - Automatic Action Tracking](#autologger---automatic-action-tracking)
4. [LLMExporter - Training Data Export](#llmexporter---training-data-export)
5. [DataModels - Log Data Structures](#datamodels---log-data-structures)
6. [Common Usage Patterns](#common-usage-patterns)
7. [Best Practices](#best-practices)

---

## System Overview

The logging system consists of **4 main components**:

| Component | Purpose | Location |
|-----------|---------|----------|
| **MainLogger** | Core unified logger (actions, console, simulation state) | `MainLogger.cs` |
| **AutoLogger** | Automatic robot action tracking (optional) | `AutoLogger.cs` |
| **LLMExporter** | Export logs for LLM training | `LLMExporter.cs` |
| **DataModels** | Data structures (RobotAction, SceneSnapshot, LogEntry) | `DataModels.cs` |

### Key Features
- ✅ **Unified logging**: Actions, console logs, and simulation state in one place
- ✅ **Per-robot or session files**: Choose between separate files per robot or single session log
- ✅ **Automatic log rotation**: Files rotate when size limit reached (default 10MB)
- ✅ **LLM training ready**: Rich metadata for training language models
- ✅ **Trajectory tracking**: Automatically capture robot movement paths
- ✅ **Environment snapshots**: Periodic scene state captures

---

## MainLogger - Core Logging Engine

**MainLogger** is the main singleton that handles all logging operations.

### Setup

1. **Add MainLogger to your scene**:
   ```
   GameObject → Create Empty → Name: "MainLogger"
   Add Component → MainLogger
   ```

2. **Configure in Inspector**:
   ```csharp
   // Configuration
   enableLogging = true              // Master switch
   logDirectory = ""                 // Leave empty for default (Application.persistentDataPath/RobotLogs)
   operationType = "navigation"      // Subfolder name (e.g., "training", "testing")
   perRobotFiles = true              // true = separate file per robot, false = single session file

   // File Management
   maxFileSizeMB = 10                // File size before rotation
   maxRotatedFiles = 5               // Number of backup files to keep

   // Console Logging
   captureUnityLogs = true           // Capture Debug.Log/Error to separate console.log file
   logSimulationState = true         // Log periodic simulation state
   stateLogInterval = 10             // Seconds between state snapshots

   // Environment Tracking
   captureEnvironment = true         // Capture scene snapshots
   environmentSampleRate = 2         // Seconds between environment captures
   trackTrajectories = true          // Track robot movement paths
   trajectorySampleRate = 0.2        // Seconds between trajectory points
   ```

### API - Action Logging

#### Start an Action
```csharp
using Logging;

// Start tracking a robot action
string actionId = MainLogger.Instance.StartAction(
    actionName: "move_to_target",
    type: ActionType.Movement,
    robotIds: new[] { "AR4_Robot_1" },
    startPos: robot.transform.position,
    targetPos: targetObject.transform.position,
    objectIds: new[] { "Target_Cube" },
    description: "Moving to pick up cube"
);
```

**Parameters**:
- `actionName`: Name of the action (e.g., "move_to_target", "grasp_object")
- `type`: ActionType enum (Task, Movement, Manipulation, Coordination, Observation)
- `robotIds`: Array of robot IDs involved
- `startPos`: Optional starting position
- `targetPos`: Optional target position
- `objectIds`: Optional array of object names
- `description`: Optional human-readable description

**Returns**: Unique `actionId` string to track this action

#### Complete an Action
```csharp
// Complete the action with results
MainLogger.Instance.CompleteAction(
    actionId: actionId,
    success: true,
    qualityScore: 0.95f,  // 0-1 quality rating
    errorMessage: null,   // Error if failed
    metrics: new Dictionary<string, float> {
        ["distance_traveled"] = 1.5f,
        ["time_taken"] = 2.3f,
        ["accuracy"] = 0.98f
    }
);
```

#### Quick Logging (Backward Compatible)
```csharp
// Simple logging (auto-starts and completes immediately)
MainLogger.Instance.LogAction(
    type: "move",
    robotId: "AR4_Robot_1",
    objectName: "Target_Cube",
    target: targetPosition,
    speed: 0.5f,
    success: true
);
```

### API - Coordination & Environment

#### Multi-Robot Coordination
```csharp
// Log coordination between multiple robots
string coordId = MainLogger.Instance.LogCoordination(
    coordinationName: "handoff_object",
    robotIds: new[] { "AR4_Left", "AR4_Right" },
    description: "Passing cube from left to right robot",
    objectIds: new[] { "Cube_1" }
);

// Complete coordination
MainLogger.Instance.CompleteAction(coordId, success: true, qualityScore: 0.9f);
```

#### Environment Snapshots
```csharp
// Manual environment capture
MainLogger.Instance.CaptureEnvironment(snapshotId: "before_task");

// Register objects for tracking
MainLogger.Instance.RegisterObject(
    obj: cubeGameObject,
    objectType: "cube",
    isGraspable: true
);
```

#### Simulation Events
```csharp
// Log simulation events (goes to console log file)
MainLogger.Instance.LogSimulationEvent(
    eventType: "robot_initialized",
    details: "AR4_Robot_1 initialized successfully",
    isActive: true
);
```

### API - Utility
```csharp
// Flush logs to disk immediately
MainLogger.Instance.FlushLogs();

// Flush specific robot's logs
MainLogger.Instance.FlushLogs("AR4_Robot_1");
```

### Output Files

MainLogger creates these files in `Application.persistentDataPath/RobotLogs/{operationType}/`:

```
RobotLogs/
└── navigation/
    ├── AR4_Robot_1_actions.json     # Per-robot action logs (if perRobotFiles=true)
    ├── AR4_Robot_2_actions.json
    ├── console_20250104_143022.log  # Unity console logs
    └── robot_actions_20250104_143022.json  # Session log (if perRobotFiles=false)
```

**Log rotation**: When a file exceeds `maxFileSizeMB`, it's renamed to `.1`, `.2`, etc.

---

## AutoLogger - Automatic Action Tracking

**AutoLogger** is an optional component that automatically tracks robot movements and gripper actions.

### Setup

1. **Attach to RobotController**:
   ```
   Select your robot GameObject
   Add Component → AutoLogger
   ```

2. **Configure**:
   ```csharp
   enableAutoLogging = true      // Enable automatic tracking
   robotId = ""                  // Leave empty to auto-detect
   logMovement = true            // Track robot movements
   logGripper = true             // Track gripper open/close
   autoRegisterObjects = true    // Auto-register scene objects
   ```

### How It Works

AutoLogger automatically detects and logs:
- **Movement**: When target changes, logs "move_to_target" action
- **Gripper**: When gripper opens/closes, logs "open_gripper"/"close_gripper" actions
- **Scene Objects**: Registers all graspable objects at startup

### Manual Custom Actions

```csharp
// Get AutoLogger reference
var autoLogger = GetComponent<AutoLogger>();

// Start custom action
string actionId = autoLogger.LogCustomAction(
    actionName: "inspect_object",
    type: ActionType.Observation,
    description: "Examining object with camera",
    targetPos: objectPosition,
    objectIds: new[] { "Cube_1" }
);

// Complete custom action
autoLogger.CompleteCustomAction(actionId, success: true, quality: 0.85f);
```

### Editor Helpers

Right-click component in Inspector:
- **Force Log Movement**: Manually log current movement
- **Force Complete Movement**: Complete active movement action

---

## LLMExporter - Training Data Export

**LLMExporter** converts raw logs into formats suitable for LLM training.

### Unity Editor Menu

Access via **Tools → Robot Logging** menu:
- **Export to JSONL**: Standard format for most LLM training
- **Export to Conversational**: Chat format (OpenAI, Anthropic)
- **Generate Statistics**: View training data stats
- **Open Log Directory**: Quick access to log files

### Programmatic Export

```csharp
using Logging;

// Export to JSONL (standard LLM format)
LLMExporter.ExportToJSONL(
    sourceLogFile: "/path/to/robot_actions.json",
    outputFile: "/path/to/training_data.jsonl"
);

// Export to conversational format
LLMExporter.ExportToConversational(
    sourceLogFile: "/path/to/robot_actions.json",
    outputFile: "/path/to/conversations.json"
);

// Generate statistics
var stats = LLMExporter.GenerateStatistics("/path/to/robot_actions.json");
// Output: total_entries, success_rate, action_types, complexity_distribution, etc.

// Filter logs by criteria
LLMExporter.FilterLogs(
    sourceFile: "/path/to/robot_actions.json",
    outputFile: "/path/to/filtered.json",
    typeFilter: ActionType.Manipulation,  // Only manipulation actions
    successFilter: true,                  // Only successful actions
    minComplexity: 2                      // Complexity level 2+
);

// Quick export from current session
LLMExporter.QuickExport("jsonl");          // or "conversational"
```

### Export Formats

#### JSONL Format (Standard)
```json
{"logId":"action_123","timestamp":"2025-01-04T14:30:22.123Z","logType":"action","action":{...},"trainingPrompt":"...","trainingResponse":"..."}
{"logId":"action_124","timestamp":"2025-01-04T14:30:25.456Z","logType":"action","action":{...},"trainingPrompt":"...","trainingResponse":"..."}
```

#### Conversational Format (Chat Models)
```json
[
  {
    "messages": [
      {"role": "user", "content": "Task: move_to_target with robot(s) AR4_Robot_1..."},
      {"role": "assistant", "content": "AR4_Robot_1 moving (successfully completed in 2.3s, quality: 0.95)"}
    ],
    "metadata": {
      "difficulty": "moderate",
      "timestamp": "2025-01-04T14:30:22.123Z",
      "learning_points": ["Action succeeded with high quality"]
    }
  }
]
```

---

## DataModels - Log Data Structures

### RobotAction
```csharp
public class RobotAction {
    // Core
    public string actionId;           // Unique identifier
    public string actionName;         // e.g., "move_to_target"
    public ActionType type;           // Task, Movement, Manipulation, Coordination, Observation
    public ActionStatus status;       // Started, InProgress, Completed, Failed

    // Participants
    public string[] robotIds;         // Robots involved
    public string[] objectIds;        // Objects involved

    // Timing
    public string timestamp;          // ISO 8601 timestamp
    public float gameTime;            // Unity Time.time
    public float duration;            // Total action duration

    // Spatial
    public Vector3 startPosition;     // Start position
    public Vector3 targetPosition;    // Target position
    public Vector3[] trajectoryPoints; // Path taken

    // Outcomes
    public bool success;              // Did it succeed?
    public string errorMessage;       // Error if failed
    public float qualityScore;        // 0-1 quality rating
    public Dictionary<string, float> metrics; // Custom metrics

    // LLM Training
    public string humanReadable;      // Natural language description
    public string[] capabilities;     // Required skills
    public int complexityLevel;       // 1-4 (simple to expert)

    // Hierarchy
    public string parentActionId;     // Parent task
    public string[] childActionIds;   // Subtasks
}
```

### ActionType Enum
```csharp
public enum ActionType {
    Task,           // High-level task
    Movement,       // Robot movement
    Manipulation,   // Gripper/object manipulation
    Coordination,   // Multi-robot coordination
    Observation     // Sensing/detection
}
```

### SceneSnapshot
```csharp
public class SceneSnapshot {
    public string snapshotId;
    public string timestamp;
    public float gameTime;
    public Object[] objects;           // Objects in scene
    public RobotState[] robots;        // Robot states
    public int totalObjects;
    public int graspableObjects;
    public string sceneDescription;    // Human-readable summary
}
```

### LogEntry
```csharp
public class LogEntry {
    public string logId;
    public string timestamp;
    public float gameTime;
    public string logType;             // "action", "scene", "session"

    public RobotAction action;         // Logged action
    public SceneSnapshot scene;        // Logged scene

    // LLM Training
    public string trainingPrompt;      // Input for LLM
    public string trainingResponse;    // Expected output
    public string[] learningPoints;    // Key takeaways
    public string difficultyLevel;     // "simple", "moderate", "complex", "expert"
}
```

---

## Common Usage Patterns

### Pattern 1: Basic Action Logging
```csharp
public class MyRobotController : MonoBehaviour {
    void MoveToTarget(Vector3 target) {
        // Start action
        string actionId = MainLogger.Instance.StartAction(
            "move_to_target",
            ActionType.Movement,
            new[] { robotId },
            startPos: transform.position,
            targetPos: target
        );

        // Perform movement...
        bool success = PerformMovement(target);

        // Complete action
        MainLogger.Instance.CompleteAction(
            actionId,
            success,
            qualityScore: CalculateAccuracy(),
            metrics: new Dictionary<string, float> {
                ["distance"] = Vector3.Distance(transform.position, target)
            }
        );
    }
}
```

### Pattern 2: Multi-Robot Coordination
```csharp
void HandoffObject(RobotController robot1, RobotController robot2, GameObject obj) {
    // Start coordination action
    string coordId = MainLogger.Instance.LogCoordination(
        "object_handoff",
        new[] { robot1.robotId, robot2.robotId },
        $"Passing {obj.name} between robots",
        new[] { obj.name }
    );

    // Perform handoff...
    bool success = ExecuteHandoff(robot1, robot2, obj);

    // Complete
    MainLogger.Instance.CompleteAction(coordId, success, 0.9f);
}
```

### Pattern 3: Automatic Logging with AutoLogger
```csharp
public class RobotSetup : MonoBehaviour {
    void Start() {
        // Just attach AutoLogger - it handles everything!
        var autoLogger = gameObject.AddComponent<AutoLogger>();
        autoLogger.enableAutoLogging = true;
        autoLogger.logMovement = true;
        autoLogger.logGripper = true;
    }
}
```

### Pattern 4: Environment Tracking
```csharp
void SetupScene() {
    var logger = MainLogger.Instance;

    // Register all objects
    foreach (var obj in FindObjectsOfType<Rigidbody>()) {
        logger.RegisterObject(obj.gameObject, isGraspable: obj.mass < 1f);
    }

    // Capture initial state
    logger.CaptureEnvironment("scene_start");

    // Capture state during task
    InvokeRepeating("CaptureState", 5f, 5f);
}

void CaptureState() {
    MainLogger.Instance.CaptureEnvironment();
}
```

### Pattern 5: Export for Training
```csharp
#if UNITY_EDITOR
[UnityEditor.MenuItem("My Tools/Export Training Data")]
static void ExportData() {
    // Export latest session to conversational format
    LLMExporter.QuickExport("conversational");

    // Generate statistics
    string logDir = Path.Combine(Application.persistentDataPath, "RobotLogs");
    var files = Directory.GetFiles(logDir, "*.json")
                         .OrderByDescending(f => File.GetLastWriteTime(f))
                         .ToArray();

    if (files.Length > 0) {
        var stats = LLMExporter.GenerateStatistics(files[0]);
        Debug.Log($"Success Rate: {stats["success_rate"]}");
    }
}
#endif
```

---

## Best Practices

### 1. **Use Descriptive Action Names**
```csharp
// Good
MainLogger.Instance.StartAction("move_to_pickup_position", ...);
MainLogger.Instance.StartAction("grasp_target_cube", ...);

// Bad
MainLogger.Instance.StartAction("action1", ...);
MainLogger.Instance.StartAction("move", ...);
```

### 2. **Always Complete Actions**
```csharp
string actionId = MainLogger.Instance.StartAction(...);
try {
    PerformAction();
    MainLogger.Instance.CompleteAction(actionId, true, quality);
} catch (Exception ex) {
    MainLogger.Instance.CompleteAction(actionId, false, 0f, ex.Message);
}
```

### 3. **Use Meaningful Metrics**
```csharp
var metrics = new Dictionary<string, float> {
    ["distance_error"] = Vector3.Distance(actual, target),
    ["time_taken"] = Time.time - startTime,
    ["smoothness"] = CalculateSmoothness(),
    ["energy_used"] = GetEnergyConsumption()
};
```

### 4. **Log Simulation Events**
```csharp
// Log important simulation milestones
MainLogger.Instance.LogSimulationEvent("training_started", "Episode 1 started");
MainLogger.Instance.LogSimulationEvent("collision_detected", $"Robot hit {obj.name}");
MainLogger.Instance.LogSimulationEvent("goal_reached", "Target achieved");
```

### 5. **Flush Logs Before Quit**
```csharp
void OnApplicationQuit() {
    MainLogger.Instance?.FlushLogs();
}
```

### 6. **Use AutoLogger for Simple Cases**
- Attach AutoLogger to robot GameObjects for automatic movement/gripper tracking
- Use manual logging for complex multi-step tasks

### 7. **Export Regularly**
- Use **Tools → Robot Logging → Export to JSONL** after training sessions
- Filter logs by success/complexity for focused training datasets

### 8. **Monitor Log Size**
- Check `maxFileSizeMB` setting (default 10MB)
- Adjust `maxRotatedFiles` based on storage needs
- Logs auto-rotate, so old data won't be lost

---

## Troubleshooting

### "MainLogger instance not found"
**Solution**: Add MainLogger component to scene as persistent GameObject

### No logs being written
**Solution**: Check `enableLogging = true` in Inspector

### Log files too large
**Solution**: Reduce `maxFileSizeMB` or increase rotation frequency

### Missing trajectory data
**Solution**: Ensure `trackTrajectories = true` and provide `startPos` when starting action

### Export fails
**Solution**: Ensure log files exist in `Application.persistentDataPath/RobotLogs/`

---

## Summary

| Task | Component | Method |
|------|-----------|--------|
| Log robot action | MainLogger | `StartAction()` → `CompleteAction()` |
| Quick simple log | MainLogger | `LogAction()` |
| Multi-robot coordination | MainLogger | `LogCoordination()` |
| Capture scene | MainLogger | `CaptureEnvironment()` |
| Simulation events | MainLogger | `LogSimulationEvent()` |
| Automatic tracking | AutoLogger | Attach to robot GameObject |
| Export for training | LLMExporter | `QuickExport()` or Unity menu |
| Generate stats | LLMExporter | `GenerateStatistics()` |
| Filter logs | LLMExporter | `FilterLogs()` |

---

**For more details**, see the source code documentation in each `.cs` file.
