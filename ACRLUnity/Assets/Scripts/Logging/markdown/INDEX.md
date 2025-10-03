# Simplified Robot Logging System - Quick Reference

## 📁 File Overview

### Core Files (Must Have)

#### 1. **SimplifiedDataModels.cs** (~150 lines)
**Purpose**: Core data structures for logging
**Contains**:
- `RobotAction` - Unified action/task/event class
- `SceneSnapshot` - Environment state
- `LogEntry` - Final output format
**When to edit**: Adding new data fields
**Dependencies**: None

#### 2. **SimplifiedRobotLogger.cs** (~350 lines)
**Purpose**: Main logging engine (replaces 5 old components)
**Contains**:
- Action tracking (start/complete)
- Environment monitoring
- Trajectory recording
- File writing
**When to edit**: Adding core logging features
**Dependencies**: SimplifiedDataModels

#### 3. **SimplifiedLLMExporter.cs** (~150 lines)
**Purpose**: Export logs for LLM training
**Contains**:
- JSONL export
- Conversational export
- Statistics generation
- Filter utilities
- Unity Editor menu integration
**When to edit**: Adding export formats or statistics
**Dependencies**: SimplifiedDataModels

### Optional Files (Nice to Have)

#### 4. **SimplifiedAutoLogger.cs** (~150 lines)
**Purpose**: Automatic logging for robots
**Contains**:
- Auto-detection of robot/gripper actions
- Automatic scene object registration
- Movement/gripper monitoring
**When to use**: Attach to RobotController or GripperController for automatic logging
**Dependencies**: SimplifiedRobotLogger, RobotController, GripperController

#### 5. **QuickStartExample.cs** (~100 lines)
**Purpose**: Example usage demonstration
**Contains**:
- Pick-and-place task example
- API usage patterns
- Export examples
**When to use**: Learning the API or as a template
**Dependencies**: SimplifiedRobotLogger, RobotController

### Documentation

#### 6. **README.md**
**Purpose**: Complete user guide
**Contains**:
- Feature overview
- Setup instructions
- API reference
- Output format examples
- Comparison with old system

#### 7. **MIGRATION_GUIDE.md**
**Purpose**: Migration from complex system
**Contains**:
- Step-by-step migration
- API mapping old → new
- Common issues and solutions
- Testing checklist

#### 8. **INDEX.md** (this file)
**Purpose**: Quick navigation and reference

## 🚀 Quick Start Cheat Sheet

### Minimal Setup (30 seconds)
```csharp
// Add to scene
var logger = new GameObject("RobotLogger")
    .AddComponent<SimplifiedRobotLogger>();
```

### With Auto-Logging (1 minute)
```csharp
// Add to scene
var logger = new GameObject("RobotLogger")
    .AddComponent<SimplifiedRobotLogger>();

// Add to each robot
robotController.gameObject.AddComponent<SimplifiedAutoLogger>();
```

### Manual Logging (Full Control)
```csharp
var logger = SimplifiedRobotLogger.Instance;

// Start action
string id = logger.StartAction(
    "pick_object",
    ActionType.Manipulation,
    new[] { "Robot1" },
    targetPos: target.position,
    objectIds: new[] { "Cube" }
);

// Complete action
logger.CompleteAction(id, success: true, qualityScore: 0.9f);
```

### Export Logs
```csharp
// From code
SimplifiedLLMExporter.QuickExport("jsonl");

// Or from menu
// Tools → Robot Logging → Export to JSONL
```

## 📊 API Quick Reference

### SimplifiedRobotLogger

#### Core Methods
```csharp
// Start an action
string StartAction(
    string actionName,
    ActionType type,
    string[] robotIds,
    Vector3? startPos = null,
    Vector3? targetPos = null,
    string[] objectIds = null,
    string description = null
)

// Complete an action
void CompleteAction(
    string actionId,
    bool success,
    float qualityScore = 0f,
    string errorMessage = null,
    Dictionary<string, float> metrics = null
)

// Log coordination
string LogCoordination(
    string coordinationName,
    string[] robotIds,
    string description = null,
    string[] objectIds = null
)

// Capture environment
void CaptureEnvironment(string snapshotId = null)

// Register object
void RegisterObject(
    GameObject obj,
    string objectType = null,
    bool isGraspable = true
)

// Backward compatibility
void LogAction(
    string type,
    string robotId,
    string objectName = null,
    Vector3? target = null,
    float[] jointAngles = null,
    float speed = 0f,
    bool success = true,
    string errorMessage = null
)
```

### SimplifiedLLMExporter

#### Export Methods
```csharp
// Quick export (latest log)
static void QuickExport(string format = "jsonl")

// Export specific file to JSONL
static void ExportToJSONL(string sourceFile, string outputFile)

// Export to conversational format
static void ExportToConversational(string sourceFile, string outputFile)

// Generate statistics
static Dictionary<string, object> GenerateStatistics(string logFile)

// Filter logs
static void FilterLogs(
    string sourceFile,
    string outputFile,
    ActionType? typeFilter = null,
    bool? successFilter = null,
    int? minComplexity = null
)
```

#### Unity Menu
- **Tools → Robot Logging → Export to JSONL**
- **Tools → Robot Logging → Export to Conversational**
- **Tools → Robot Logging → Generate Statistics**
- **Tools → Robot Logging → Open Log Directory**

### SimplifiedAutoLogger

#### Configuration
```csharp
public bool enableAutoLogging = true;
public string robotId;              // Auto-detected if empty
public bool logMovement = true;
public bool logGripper = true;
public bool autoRegisterObjects = true;
```

#### Manual Override Methods
```csharp
string LogCustomAction(
    string actionName,
    ActionType type,
    string description = null,
    Vector3? targetPos = null,
    string[] objectIds = null
)

void CompleteCustomAction(
    string actionId,
    bool success,
    float quality = 0.8f
)
```

## 🎯 Common Use Cases

### Use Case 1: Basic Movement Logging
```csharp
var logger = SimplifiedRobotLogger.Instance;

string moveId = logger.StartAction(
    "move_to_position",
    ActionType.Movement,
    new[] { "Robot1" },
    startPos: robot.position,
    targetPos: target.position
);

// ... robot moves ...

logger.CompleteAction(moveId, success: true, qualityScore: 0.95f);
```

### Use Case 2: Object Manipulation
```csharp
string pickId = logger.StartAction(
    "pick_object",
    ActionType.Manipulation,
    new[] { "Robot1" },
    objectIds: new[] { "Cube" },
    description: "Picking up red cube"
);

// ... gripper closes ...

logger.CompleteAction(pickId, success: true, qualityScore: 0.9f);
```

### Use Case 3: Multi-Robot Coordination
```csharp
string coordId = logger.LogCoordination(
    "synchronized_lift",
    new[] { "Robot1", "Robot2" },
    "Both robots lifting heavy object",
    new[] { "HeavyBox" }
);

// ... coordination happens ...

logger.CompleteAction(coordId, success: true, qualityScore: 0.85f);
```

### Use Case 4: Automatic Logging
```csharp
// Just attach SimplifiedAutoLogger to robots
// Actions are logged automatically when:
// - Robot target changes (movement)
// - Gripper position changes (manipulation)
// - Objects are detected in scene (registration)
```

## 🐛 Troubleshooting

### Logger Not Found
```csharp
// Check: Is SimplifiedRobotLogger in scene?
var logger = SimplifiedRobotLogger.Instance;
if (logger == null) {
    Debug.LogError("Add SimplifiedRobotLogger to scene!");
}
```

### No Logs Generated
```csharp
// Check: Is logging enabled?
logger.enableLogging = true;

// Check: Are actions being completed?
// Actions must be completed to be written to file
```

### Log File Not Found
```csharp
// Log location
string logDir = Path.Combine(
    Application.persistentDataPath,
    "RobotLogs"
);
Debug.Log($"Logs at: {logDir}");

// Or use menu: Tools → Robot Logging → Open Log Directory
```

### Export Fails
```csharp
// Check: Do log files exist?
string logPath = Path.Combine(
    Application.persistentDataPath,
    "RobotLogs",
    "robot_actions_*.jsonl"
);

// Run simulation first to generate logs
```

## 📈 Performance Guidelines

### Optimal Settings

**For Development** (High Detail):
```csharp
logger.enableLogging = true;
logger.captureEnvironment = true;
logger.environmentSampleRate = 1f;  // Every second
logger.trackTrajectories = true;
logger.trajectorySampleRate = 0.1f; // 10 points/second
```

**For Production** (Balanced):
```csharp
logger.enableLogging = true;
logger.captureEnvironment = true;
logger.environmentSampleRate = 5f;  // Every 5 seconds
logger.trackTrajectories = true;
logger.trajectorySampleRate = 0.5f; // 2 points/second
```

**For Performance** (Minimal):
```csharp
logger.enableLogging = true;
logger.captureEnvironment = false;  // No environment
logger.trackTrajectories = false;   // No trajectories
```

### Memory Management
- Each action: ~1KB
- Trajectory point: ~50 bytes
- Environment snapshot: ~5-10KB
- Typical usage: ~5MB for 1000 actions

## 🔄 Workflow

### 1. Setup Phase
1. Add SimplifiedRobotLogger to scene
2. (Optional) Add SimplifiedAutoLogger to robots
3. Configure settings in inspector
4. Run simulation

### 2. Collection Phase
1. Actions are logged automatically or manually
2. Logs written to JSONL file in real-time
3. Monitor console for any errors

### 3. Export Phase
1. Stop simulation (logs are flushed)
2. Use menu: Tools → Robot Logging → Export
3. Or call `SimplifiedLLMExporter.QuickExport()`
4. Logs are in: `PersistentDataPath/RobotLogs/`

### 4. Training Phase
1. Use exported JSONL with your LLM training pipeline
2. Each log entry has `trainingPrompt` and `trainingResponse`
3. Filter by complexity/type if needed
4. Train your model!

## 🎓 Learning Path

### Beginner (10 minutes)
1. Read README.md "Quick Setup" section
2. Run QuickStartExample.cs
3. Check log output
4. Try exporting with menu

### Intermediate (30 minutes)
1. Add SimplifiedRobotLogger to your scene
2. Manually log 2-3 actions
3. Attach SimplifiedAutoLogger to robots
4. Compare manual vs automatic logging

### Advanced (1 hour)
1. Customize RobotAction data fields
2. Add custom metrics to actions
3. Create filtered exports
4. Integrate with your LLM training pipeline

## 📚 Additional Resources

### Code Examples
- **QuickStartExample.cs** - Complete pick-and-place example
- **SimplifiedAutoLogger.cs** - Auto-logging patterns
- **SimplifiedRobotLogger.cs** - Full API implementation

### Documentation
- **README.md** - Complete feature guide
- **MIGRATION_GUIDE.md** - For upgrading from old system
- **SIMPLIFICATION_SUMMARY.md** - Before/after comparison

### Support
- Check Unity console for errors
- Review examples for patterns
- See MIGRATION_GUIDE.md for common issues

---

**Quick Links**:
- 📖 [README.md](README.md) - Start here
- 🔄 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Upgrade guide
- 📊 [SIMPLIFICATION_SUMMARY.md](../SIMPLIFICATION_SUMMARY.md) - Full comparison
- 💻 [QuickStartExample.cs](QuickStartExample.cs) - Code example
