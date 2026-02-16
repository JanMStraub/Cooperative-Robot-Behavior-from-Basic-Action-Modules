# Unity-Python Communication System

## Overview

The `ACRLUnity/Assets/Scripts/PythonCommunication` folder contains the high-level client implementations and coordination systems for bidirectional communication between Unity and the Python backend. Built on top of the robust **Core** architecture (see `Core/README.md`), this system enables:

- **Natural language robot control** via LLM-powered sequence parsing
- **Real-time world state streaming** for spatial reasoning operations
- **Bidirectional command/response flow** with Protocol V2 correlation
- **Multi-robot coordination verification** for collaborative workflows

The system connects to **4 active Python servers** running in the unified backend (`ACRLPython/orchestrators/RunRobotController.py`):
- **ImageServer** (ports 5005/5006) - Image streaming
- **CommandServer** (port 5010) - Bidirectional commands & results
- **SequenceServer** (port 5013) - Multi-command sequences with LLM parsing + **AutoRT integration**
- **WorldStateServer** (port 5014) - Robot/object state streaming

---

## Architecture Overview

```
PythonCommunication/
├── Core/                          # Base infrastructure (see Core/README.md)
│   ├── TCPClientBase.cs           # Abstract TCP connection management
│   ├── BidirectionalClientBase.cs # Request/response correlation
│   ├── UnityProtocol.cs           # Protocol V2 message encoding/decoding
│   └── JsonParser.cs              # Centralized JSON parsing
│
├── High-Level Clients (This Directory)
│   ├── SequenceClient.cs          # Multi-command sequence client (port 5013)
│   ├── AutoRTManager.cs           # Autonomous task generation client (port 5013)
│   ├── ResultsClient.cs           # Bidirectional results receiver (port 5010)
│   ├── UnifiedPythonReceiver.cs   # Result routing manager
│   ├── WorldStateClient.cs        # World state streaming client (port 5014)
│   └── WorldStatePublisher.cs     # Publish Unity state to Python
│
├── DataModels/
│   ├── SequenceDataModels.cs      # SequenceResult, Operation
│   ├── AutoRTDataModels.cs        # ProposedTask, TaskSelectionStrategy
│   ├── DetectionDataModels.cs     # DetectionResult, DetectedObject
│   └── RAGDataModels.cs           # OperationContext
│
├── Command Handling
│   └── PythonCommandHandler.cs    # Execute Python commands on Unity robots
│
└── Coordination Verification
    ├── ICoordinationVerifier.cs       # Interface for verification strategies
    ├── UnityCoordinationVerifier.cs   # Local Unity-side verification
    └── PythonCoordinationVerifier.cs  # Remote Python-side verification
```

---

## Component Details

### 1. SequenceClient.cs

**Purpose**: Sends multi-command natural language sequences to Python's SequenceServer (port 5013) and receives parsed operation results.

**Inheritance**: `BidirectionalClientBase<SequenceResult>` → `TCPClientBase` → `MonoBehaviour`

#### Key Features

**Natural Language Processing** (`SequenceClient.cs:168-226`):
- Sends natural language commands like `"move to the red cube and close the gripper"`
- Python parses using LLM/regex hybrid (CommandParser + RAG system)
- Returns structured `SequenceResult` with operation list

**Singleton Pattern** (`SequenceClient.cs:83-97`):
```csharp
public static SequenceClient Instance { get; private set; }

protected override void Awake()
{
    if (Instance == null)
    {
        Instance = this;
        DontDestroyOnLoad(gameObject);
        base.Awake();
        _serverPort = CommunicationConstants.SEQUENCE_SERVER_PORT; // 5013
    }
    else { Destroy(gameObject); }
}
```

**Command Sending** (`SequenceClient.cs:149-188`):
```csharp
public bool ExecuteSequence(string command, string robotId = null)
{
    if (!IsConnected || string.IsNullOrEmpty(command))
        return false;

    string robot = robotId ?? _defaultRobotId;
    uint requestId = GenerateRequestId();

    // Encode message using Protocol V2
    byte[] message = EncodeSequenceMessage(command, robot, requestId);

    // Send using base class method (handles locking internally)
    bool sent = SendRequest(message, requestId);

    if (sent && _logCommands)
    {
        Debug.Log(
            $"{LogPrefix} [req={requestId}] Sent sequence: '{command}' (robot={robot})"
        );
    }

    return sent;
}
```

**Convenience Methods** (`SequenceClient.cs:193-220`):
```csharp
// Move and control gripper
public bool MoveAndGrip(float x, float y, float z, bool closeGripper, string robotId = null)

// Pick operation (move, close gripper, lift)
public bool Pick(float x, float y, float z, float liftHeight = 0.1f, string robotId = null)

// Place operation (move, open gripper, lift)
public bool Place(float x, float y, float z, float liftHeight = 0.1f, string robotId = null)
```

**Auto-Execution** (`SequenceClient.cs:263-276`):
- If `_autoExecuteResult = true`, automatically forwards commands to `PythonCommandHandler`
- Enables seamless "say it and it happens" workflow

#### Usage Example

```csharp
// Simple command
SequenceClient.Instance.ExecuteSequence("move to (0.3, 0.2, 0.1)");

// With robot ID
SequenceClient.Instance.ExecuteSequence(
    "detect the blue cube and move to it",
    robotId: "Robot1"
);

// Using convenience methods
SequenceClient.Instance.MoveAndGrip(0.3f, 0.2f, 0.1f, closeGripper: true);
SequenceClient.Instance.Pick(0.3f, 0.2f, 0.05f, liftHeight: 0.1f);

// Subscribe to results via event
SequenceClient.Instance.OnSequenceResultReceived += result => {
    Debug.Log($"Sequence completed: {result.completed_commands}/{result.total_commands}");
};
```

#### Editor Integration

The component includes a **custom inspector** (`SequenceClientEditor.cs`) with:
- Text area for command input
- "Send Command" button for testing
- Real-time status display (commands completed, total)
- Connection status indicator

---

### 2. AutoRTManager.cs

**Purpose**: Autonomous Robot Task generation client that connects to Python's SequenceServer (port 5013) for LLM-powered task proposal generation with human-in-the-loop approval.

**Inheritance**: `BidirectionalClientBase<AutoRTResponse>` → `TCPClientBase` → `MonoBehaviour`

#### Key Features

**Autonomous Task Generation** (`AutoRTManager.cs:98-153`):
- Sends task generation requests to Python LLM backend
- Receives structured task proposals with operations lists
- Tasks validated against 30 registered operations in Registry
- Support for multi-robot collaborative tasks

**Human-in-the-Loop Workflow** (`AutoRTManager.cs:273-349`):
- Tasks appear in custom inspector UI for approval
- User can execute (approve) or reject each task
- Prevents accidental execution of unsafe/unwanted tasks
- Task queue management with configurable limits

**Continuous Loop Mode** (`AutoRTManager.cs:158-269`):
- Optional autonomous mode for continuous task generation
- Configurable delay between generations
- Can be started/stopped at runtime
- Loop status synchronized with Python backend

**Singleton Pattern** (`AutoRTManager.cs:76-93`):
```csharp
public static AutoRTManager Instance { get; private set; }

protected override void Awake()
{
    if (Instance == null)
    {
        Instance = this;
        DontDestroyOnLoad(gameObject);
        base.Awake();
        _serverPort = CommunicationConstants.SEQUENCE_SERVER_PORT; // 5013
    }
    else { Destroy(gameObject); }
}
```

**Task Generation** (`AutoRTManager.cs:104-153`):
```csharp
public bool GenerateTasks(int? numTasks = null)
{
    if (!IsConnected || _config == null)
        return false;

    int taskCount = numTasks ?? _config.maxTaskCandidates;
    uint requestId = GenerateRequestId();

    // Build params JSON
    string robotIdsJson = "[" + string.Join(",",
        Array.ConvertAll(_config.robotIds, r => $"\"{r}\"")) + "]";
    string paramsJson = $"{{\"num_tasks\":{taskCount}," +
                        $"\"robot_ids\":{robotIdsJson}," +
                        $"\"strategy\":\"{_config.strategy.ToString().ToLower()}\"}}";

    // Encode AUTORT_COMMAND message
    byte[] message = UnityProtocol.EncodeAutoRTCommand(
        "generate", paramsJson, requestId
    );

    return WriteToStream(message);
}
```

**Loop Control** (`AutoRTManager.cs:165-220`):
```csharp
public bool StartLoop(float? loopDelay = null)
{
    if (!IsConnected || _loopRunning)
        return false;

    float delay = loopDelay ?? _config.loopDelaySeconds;
    uint requestId = GenerateRequestId();

    string paramsJson = $"{{\"loop_delay\":{delay}," +
                        $"\"robot_ids\":{robotIdsJson}," +
                        $"\"strategy\":\"{_config.strategy}\"}}";

    byte[] message = UnityProtocol.EncodeAutoRTCommand(
        "start_loop", paramsJson, requestId
    );

    return WriteToStream(message);
}

public bool StopLoop()
{
    if (!IsConnected || !_loopRunning)
        return false;

    byte[] message = UnityProtocol.EncodeAutoRTCommand(
        "stop_loop", "{}", GenerateRequestId()
    );

    return WriteToStream(message);
}
```

**Task Execution** (`AutoRTManager.cs:280-326`):
```csharp
public bool ExecuteTask(ProposedTask task)
{
    if (!IsConnected || task == null)
        return false;

    string paramsJson = $"{{\"task_id\":\"{task.task_id}\"}}";
    byte[] message = UnityProtocol.EncodeAutoRTCommand(
        "execute_task", paramsJson, GenerateRequestId()
    );

    return WriteToStream(message);
}

public void RejectTask(ProposedTask task)
{
    if (task != null)
    {
        _pendingTasks.Remove(task);
        Debug.Log($"Rejected task: {task.task_id}");
    }
}
```

**Protocol V2 Compliance** (`AutoRTManager.cs:360-406`):
```csharp
protected override AutoRTResponse ReceiveResponse()
{
    // Read header (5 bytes: type + request_id)
    byte[] headerBuffer = new byte[UnityProtocol.HEADER_SIZE];
    ReadExactly(_stream, headerBuffer, UnityProtocol.HEADER_SIZE);

    // Decode header
    UnityProtocol.DecodeHeader(headerBuffer, 0,
        out MessageType type, out uint requestId);

    if (type != MessageType.AUTORT_RESPONSE)
    {
        throw new IOException($"Expected AUTORT_RESPONSE, got {type}");
    }

    // Read JSON payload
    byte[] lenBuffer = new byte[4];
    ReadExactly(_stream, lenBuffer, 4);
    int jsonLen = BitConverter.ToInt32(lenBuffer, 0);

    byte[] jsonBytes = new byte[jsonLen];
    ReadExactly(_stream, jsonBytes, jsonLen);
    string json = Encoding.UTF8.GetString(jsonBytes);

    // Parse to AutoRTResponse
    if (JsonParser.TryParseWithLogging<AutoRTResponse>(
        json, out AutoRTResponse response, LogPrefix))
    {
        response.request_id = requestId;
        return response;
    }

    return null;
}
```

**Response Handling** (`AutoRTManager.cs:420-475`):
```csharp
protected override void OnResponseReceived(AutoRTResponse response)
{
    if (response == null)
        return;

    // Update loop status
    if (response.loop_running != _loopRunning)
    {
        _loopRunning = response.loop_running;
        OnLoopStatusChanged?.Invoke(_loopRunning);
    }

    // Handle errors
    if (response.HasError)
    {
        Debug.LogWarning($"Response error: {response.error}");
        _statusMessage = $"Error: {response.error}";
        return;
    }

    // Handle received tasks
    if (response.tasks != null && response.tasks.Count > 0)
    {
        foreach (var task in response.tasks)
        {
            if (_pendingTasks.Count < _config.maxDisplayTasks)
                _pendingTasks.Add(task);
        }

        _statusMessage = $"Received {response.tasks.Count} tasks";
        OnTasksReceived?.Invoke(response.tasks);
    }
}
```

#### Configuration

AutoRTManager requires an `AutoRTConfig` ScriptableObject asset:

```csharp
[Header("Configuration")]
[SerializeField]
private AutoRTConfig _config;
```

**AutoRTConfig Settings** (see `ConfigScripts/AutoRTConfig.cs`):
- `maxTaskCandidates` (1-5) - Number of tasks to generate per request
- `strategy` - Task selection strategy (Balanced/Simple/Complex/Random)
- `enableContinuousLoop` - Toggle continuous loop mode
- `loopDelaySeconds` (1-60) - Delay between generations
- `robotIds` - Robot IDs for task generation
- `enableCollaborativeTasks` - Allow multi-robot tasks
- `maxDisplayTasks` (5-20) - Maximum tasks in inspector UI
- `autoRefresh` - Auto-refresh UI in play mode
- `uiRefreshRate` (0.1-2) - UI update frequency

#### Custom Inspector

**AutoRTManagerEditor.cs** provides a rich UI for task management:

**Features**:
- **Generate Tasks** button - Manual one-shot task generation
- **Start/Stop Loop** buttons - Control continuous mode
- **Task List Display** - Scrollable list of pending tasks
- **Per-Task Actions** - Execute/Reject buttons for each task
- **Status Display** - Current loop status and message
- **Real-Time Updates** - Auto-refreshes in play mode

**Inspector Layout**:
```
┌─ AutoRT Manager ──────────────────────────┐
│ Config: DefaultAutoRTConfig               │
│ Status: Idle                              │
│                                           │
│ [ Generate Tasks ]  [ Start Loop ]       │
│                                           │
│ Pending Tasks (3):                        │
│ ┌─────────────────────────────────────┐  │
│ │ task_001: Pick up red cube          │  │
│ │   3 operations, complexity: 3        │  │
│ │   [ Execute ] [ Reject ]             │  │
│ ├─────────────────────────────────────┤  │
│ │ task_002: Handoff blue cube         │  │
│ │   5 operations, complexity: 5        │  │
│ │   Requires: Robot1, Robot2           │  │
│ │   [ Execute ] [ Reject ]             │  │
│ └─────────────────────────────────────┘  │
│                                           │
│ [ Clear All Tasks ]                       │
└───────────────────────────────────────────┘
```

#### Events

```csharp
// Fired when tasks received from Python
public event Action<List<ProposedTask>> OnTasksReceived;

// Fired when loop status changes
public event Action<bool> OnLoopStatusChanged;
```

#### Usage Example

```csharp
using PythonCommunication;

public class AutoRTController : MonoBehaviour
{
    void Start()
    {
        // Subscribe to task events
        AutoRTManager.Instance.OnTasksReceived += tasks => {
            Debug.Log($"Received {tasks.Count} task proposals");
            foreach (var task in tasks)
            {
                Debug.Log($"  - {task.task_id}: {task.description}");
            }
        };

        AutoRTManager.Instance.OnLoopStatusChanged += isRunning => {
            Debug.Log($"Loop status: {(isRunning ? "RUNNING" : "STOPPED")}");
        };
    }

    // Manual task generation
    public void GenerateTasksManually()
    {
        AutoRTManager.Instance.GenerateTasks(numTasks: 3);
    }

    // Start continuous autonomous mode
    public void StartAutonomousMode()
    {
        AutoRTManager.Instance.StartLoop(loopDelay: 5f);
    }

    // Programmatic task approval
    public void ApproveTask(ProposedTask task)
    {
        if (IsTaskSafe(task))
        {
            AutoRTManager.Instance.ExecuteTask(task);
        }
        else
        {
            AutoRTManager.Instance.RejectTask(task);
        }
    }

    bool IsTaskSafe(ProposedTask task)
    {
        // Custom safety validation logic
        return task.estimated_complexity < 7;
    }
}
```

#### Python Backend Integration

AutoRT messages share port 5013 with SequenceServer but use distinct message types:

**Message Types**:
- `AUTORT_COMMAND` (Unity → Python) - Task generation/loop control
- `AUTORT_RESPONSE` (Python → Unity) - Task proposals/status updates

**Python Handler** (`ACRLPython/servers/SequenceServer.py`):
```python
def handle_autort_command(self, command: str, params: dict, request_id: int):
    if command == "generate":
        tasks = self.task_generator.generate_tasks(
            scene=self.get_current_scene(),
            robot_ids=params.get("robot_ids", ["Robot1"]),
            num_tasks=params.get("num_tasks", 3),
            strategy=params.get("strategy", "balanced")
        )
        return AutoRTResponse(tasks=tasks, request_id=request_id)

    elif command == "start_loop":
        self.start_generation_loop(params)
        return AutoRTResponse(loop_running=True, request_id=request_id)

    elif command == "stop_loop":
        self.stop_generation_loop()
        return AutoRTResponse(loop_running=False, request_id=request_id)

    elif command == "execute_task":
        task_id = params["task_id"]
        result = self.execute_approved_task(task_id)
        return AutoRTResponse(success=result, request_id=request_id)
```

**Task Generation Pipeline** (`ACRLPython/autort/TaskGenerator.py`):
1. Query world state for scene description
2. Build LLM prompt with detected objects + operations
3. Generate task proposals via LM Studio
4. Validate operations against Registry
5. Apply Pydantic schema validation
6. Return structured ProposedTask list

**Safety Validation** (Python-side):
- Workspace bounds checking
- Max velocity/force limits
- Minimum robot separation (0.2m)
- Operation type validation
- JSON schema enforcement

---

### 3. ResultsClient.cs

**Purpose**: Receives bidirectional command results from Python's CommandServer (port 5010).

**Inheritance**: `BidirectionalClientBase<string>` → `TCPClientBase` → `MonoBehaviour`

#### Key Features

**JSON Result Streaming** (`ResultsClient.cs:30-45`):
- Continuously receives JSON results from Python
- Fires `OnJsonReceived` event with raw JSON and request ID
- Routes to `UnifiedPythonReceiver` for parsing and dispatch

**Protocol V2 Compliance** (`ResultsClient.cs:78-115`):
```csharp
protected override string ReceiveResponse()
{
    // 1. Read header
    byte[] headerBuffer = new byte[UnityProtocol.HEADER_SIZE];
    ReadExactly(_stream, headerBuffer, UnityProtocol.HEADER_SIZE);

    // 2. Decode header
    UnityProtocol.DecodeHeader(headerBuffer, 0,
        out MessageType msgType, out uint requestId);

    // 3. Read JSON payload
    string json = ReadString();

    return json;
}
```

**Request ID Extraction** (`ResultsClient.cs:120-133`):
```csharp
protected override uint GetResponseRequestId(string json)
{
    // Parse request_id from JSON for correlation
    var match = System.Text.RegularExpressions.Regex.Match(
        json, @"""request_id"":\s*(\d+)"
    );
    if (match.Success)
        return uint.Parse(match.Groups[1].Value);
    return 0;
}
```

#### Usage Pattern

```csharp
// Typically managed by UnifiedPythonReceiver
var client = gameObject.AddComponent<ResultsClient>();
client.OnJsonReceived += (json, requestId) => {
    Debug.Log($"Result [{requestId}]: {json}");
};
```

---

### 3. UnifiedPythonReceiver.cs

**Purpose**: Unified manager that routes all Python results to appropriate handlers.

**Pattern**: Facade pattern wrapping `ResultsClient`

#### Key Features

**Result Routing** (`UnifiedPythonReceiver.cs:87-120`):
```csharp
private void HandleJsonResult(string json, uint requestId)
{
    // Route based on JSON structure
    if (json.Contains("\"command_type\""))
    {
        // Robot command from SequenceServer
        if (JsonParser.TryParse<RobotCommand>(json, out var command))
        {
            PythonCommandHandler.Instance?.HandleCommand(command);
        }
    }
    else if (json.Contains("\"result_type\""))
    {
        // LLM result from vision/analysis
        if (JsonParser.TryParse<LLMResult>(json, out var result))
        {
            OnLLMResultReceived?.Invoke(result);
        }
    }
}
```

**Event System** (`UnifiedPythonReceiver.cs:30`):
```csharp
public event Action<LLMResult> OnLLMResultReceived;
```

**Singleton Initialization** (`UnifiedPythonReceiver.cs:42-62`):
```csharp
private void Awake()
{
    if (Instance == null)
    {
        Instance = this;
        DontDestroyOnLoad(gameObject);

        // Create ResultsClient child object
        GameObject clientObj = new GameObject("ResultsClient");
        clientObj.transform.SetParent(transform);
        _client = clientObj.AddComponent<ResultsClient>();
        _client.OnJsonReceived += HandleJsonResult;
    }
    else { Destroy(gameObject); }
}
```

#### Usage

Typically auto-initialized in scene. Subscribe to events:

```csharp
UnifiedPythonReceiver.Instance.OnLLMResultReceived += result => {
    Debug.Log($"LLM Analysis: {result.response}");
};
```

---

### 4. WorldStatePublisher.cs

**Purpose**: Periodically publishes robot and object states to Python's WorldStateServer (port 5014) for spatial reasoning operations.

#### Key Features

**Periodic Updates** (`WorldStatePublisher.cs:100-102`):
```csharp
[Tooltip("Update rate in Hz (updates per second)")]
[SerializeField]
private float _updateRate = 2.0f; // 2 Hz = every 0.5 seconds
```

**World State Data** (`WorldStatePublisher.cs:13-74`):
```csharp
[System.Serializable]
public class WorldStateUpdate
{
    public string type = "world_state_update";
    public List<RobotStateData> robots;    // All robots in scene
    public List<ObjectStateData> objects;  // Detected objects
    public float timestamp;
}

[System.Serializable]
public class RobotStateData
{
    public string robot_id;
    public PositionData position;
    public RotationData rotation;
    public PositionData target_position;
    public string gripper_state;  // "open", "closed", "unknown"
    public bool is_moving;
    public bool is_initialized;
    public float[] joint_angles;
}
```

**State Collection** (`WorldStatePublisher.cs:180-240`):
```csharp
private void PublishWorldState()
{
    var update = new WorldStateUpdate
    {
        timestamp = Time.time,
        robots = new List<RobotStateData>(),
        objects = new List<ObjectStateData>()
    };

    // Collect robot states from RobotManager
    foreach (var kvp in RobotManager.Instance.Robots)
    {
        var robotData = new RobotStateData
        {
            robot_id = kvp.Key,
            position = new PositionData(robot.transform.position),
            rotation = new RotationData(robot.transform.rotation),
            is_moving = controller.IsMoving,
            gripper_state = gripper.IsOpen ? "open" : "closed",
            joint_angles = GetJointAngles(controller)
        };
        update.robots.Add(robotData);
    }

    // Send to WorldStateClient
    WorldStateClient.Instance?.SendWorldState(update);
}
```

**Usage**: Attach to a GameObject in the scene. Auto-discovers robots from `RobotManager`.

```csharp
// Optional: Track specific objects
WorldStatePublisher.Instance.TrackObject(cubeGameObject, "blue_cube");
```

---

### 5. WorldStateClient.cs

**Purpose**: TCP client for streaming world state updates to Python WorldStateServer (port 5014).

**Inheritance**: `TCPClientBase` → `MonoBehaviour`

**Note**: This is a **one-way** client (Unity → Python only), so it inherits from `TCPClientBase` instead of `BidirectionalClientBase`.

#### Key Features

**JSON Streaming** (`WorldStateClient.cs:68-98`):
```csharp
public void SendWorldState(WorldStateUpdate update)
{
    if (!IsConnected)
        return;

    // Serialize to JSON
    string json = JsonUtility.ToJson(update);

    // Encode with Protocol V2 header
    uint requestId = GenerateRequestId();
    byte[] header = UnityProtocol.EncodeHeader(
        MessageType.RESULT,
        requestId
    );

    byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
    byte[] message = new byte[header.Length + 4 + jsonBytes.Length];

    // Write: [header][length][json]
    Buffer.BlockCopy(header, 0, message, 0, header.Length);
    WriteUInt32BE(message, header.Length, (uint)jsonBytes.Length);
    Buffer.BlockCopy(jsonBytes, 0, message, header.Length + 4, jsonBytes.Length);

    WriteToStream(message);
}
```

**Singleton Pattern** (`WorldStateClient.cs:35-50`):
```csharp
public static WorldStateClient Instance { get; private set; }

private void Awake()
{
    if (Instance == null)
    {
        Instance = this;
        DontDestroyOnLoad(gameObject);
        base.Awake();
        _serverPort = 5014; // WorldStateServer port
    }
    else { Destroy(gameObject); }
}
```

---

### 6. PythonCommandHandler.cs

**Purpose**: Executes robot commands received from Python operations on Unity robots.

**Pattern**: Command pattern with async coroutine execution

#### Supported Commands

1. **move_to_coordinate** - Move robot end effector to target position
2. **control_gripper** - Open or close the gripper
3. **check_robot_status** - Get current robot state
4. **return_to_start_position** - Move robot back to initial position
5. **execute_grasp** - Execute grasp planning pipeline

#### Key Features

**Command Routing** (`PythonCommandHandler.cs:110-180`):
```csharp
public void HandleCommand(RobotCommand command)
{
    switch (command.command_type)
    {
        case "move_to_coordinate":
            StartCoroutine(ExecuteMoveToCoordinate(command));
            break;

        case "control_gripper":
            StartCoroutine(ExecuteGripperControl(command));
            break;

        case "check_robot_status":
            StartCoroutine(ExecuteStatusCheck(command));
            break;

        case "execute_grasp":
            StartCoroutine(ExecuteGraspPipeline(command));
            break;

        default:
            Debug.LogWarning($"Unknown command: {command.command_type}");
            SendCompletion(command, false);
            break;
    }
}
```

**Async Execution with Completion** (`PythonCommandHandler.cs:240-295`):
```csharp
private IEnumerator ExecuteMoveToCoordinate(RobotCommand command)
{
    var robot = RobotManager.Instance.GetRobot(command.robot_id);
    if (robot == null)
    {
        SendCompletion(command, false);
        yield break;
    }

    // Set target position
    Vector3 target = new Vector3(
        command.parameters.target_position.x,
        command.parameters.target_position.y,
        command.parameters.target_position.z
    );

    robot.controller.SetTarget(target, Quaternion.identity);

    // Wait for target reached
    yield return new WaitUntil(() => robot.controller.TargetReached);

    // Send completion to Python
    SendCompletion(command, success: true);
}
```

**Completion Notification** (`PythonCommandHandler.cs:450-475`):
```csharp
private void SendCompletion(RobotCommand command, bool success)
{
    var completion = new CommandCompletionData
    {
        type = "command_completion",
        robot_id = command.robot_id,
        command_type = command.command_type,
        success = success,
        request_id = command.request_id,
        timestamp = Time.time
    };

    string json = JsonUtility.ToJson(completion);

    // Send back via ResultsClient (port 5010)
    ResultsClient.Instance?.SendCompletion(json, command.request_id);
}
```

#### Grasp Pipeline Integration (`PythonCommandHandler.cs:380-430`)

```csharp
private IEnumerator ExecuteGraspPipeline(RobotCommand command)
{
    var pipeline = FindObjectOfType<GraspPlanningPipeline>();
    if (pipeline == null)
    {
        Debug.LogError("GraspPlanningPipeline not found");
        SendCompletion(command, false);
        yield break;
    }

    // Extract object to grasp
    string objectId = command.parameters.object_id;
    GameObject targetObject = GameObject.Find(objectId);

    // Execute grasp planning
    bool success = false;
    pipeline.PlanAndExecuteGrasp(
        targetObject,
        approach: command.parameters.preferred_approach ?? "auto",
        onComplete: (result) => { success = result; }
    );

    // Wait for completion
    yield return new WaitUntil(() => pipeline.IsComplete);

    SendCompletion(command, success);
}
```

---

## Coordination Verification System

The coordination verification system enables dual-robot workflows to verify spatial constraints and collision safety. It supports both **local Unity-side** and **remote Python-side** verification strategies.

### ICoordinationVerifier.cs

**Purpose**: Interface defining coordination verification contract.

```csharp
public interface ICoordinationVerifier
{
    /// <summary>
    /// Verify coordination between two robots
    /// </summary>
    /// <param name="robot1Id">First robot ID</param>
    /// <param name="robot2Id">Second robot ID</param>
    /// <param name="operation">Operation type ("handoff", "collaborative_grasp", etc.)</param>
    /// <param name="callback">Callback with verification result</param>
    void VerifyCoordination(
        string robot1Id,
        string robot2Id,
        string operation,
        Action<bool, string> callback
    );

    /// <summary>
    /// Check if verifier is ready
    /// </summary>
    bool IsReady { get; }
}
```

### UnityCoordinationVerifier.cs

**Purpose**: Local Unity-side collision detection and distance verification.

**Implementation**: `ICoordinationVerifier`

#### Key Features

**Distance Checks** (`UnityCoordinationVerifier.cs:80-120`):
```csharp
public void VerifyCoordination(
    string robot1Id,
    string robot2Id,
    string operation,
    Action<bool, string> callback
)
{
    var robot1 = RobotManager.Instance.GetRobot(robot1Id);
    var robot2 = RobotManager.Instance.GetRobot(robot2Id);

    if (robot1 == null || robot2 == null)
    {
        callback(false, "Robot not found");
        return;
    }

    // Check distance
    float distance = Vector3.Distance(
        robot1.transform.position,
        robot2.transform.position
    );

    if (operation == "handoff")
    {
        // Handoff requires close proximity
        if (distance > _maxHandoffDistance)
        {
            callback(false, $"Robots too far apart: {distance:F2}m");
            return;
        }
    }

    // Check collision potential
    if (CheckCollisionPotential(robot1, robot2))
    {
        callback(false, "Collision detected");
        return;
    }

    callback(true, "Coordination verified");
}
```

**Collision Detection** (`UnityCoordinationVerifier.cs:130-165`):
```csharp
private bool CheckCollisionPotential(RobotInstance r1, RobotInstance r2)
{
    // Get all colliders for both robots
    Collider[] r1Colliders = r1.robotGameObject.GetComponentsInChildren<Collider>();
    Collider[] r2Colliders = r2.robotGameObject.GetComponentsInChildren<Collider>();

    foreach (var c1 in r1Colliders)
    {
        foreach (var c2 in r2Colliders)
        {
            // Check if bounds overlap
            if (c1.bounds.Intersects(c2.bounds))
            {
                return true; // Collision potential
            }
        }
    }

    return false; // No collision
}
```

### PythonCoordinationVerifier.cs

**Purpose**: Remote verification via Python's coordination verification system.

**Implementation**: `ICoordinationVerifier`

**Note**: Sends verification request to Python WorldStateServer and waits for response. Useful for complex spatial reasoning that leverages Python's world state tracking.

```csharp
public void VerifyCoordination(
    string robot1Id,
    string robot2Id,
    string operation,
    Action<bool, string> callback
)
{
    // Build verification request
    var request = new CoordinationVerificationRequest
    {
        robot1_id = robot1Id,
        robot2_id = robot2Id,
        operation = operation,
        timestamp = Time.time
    };

    // Send to Python WorldStateServer
    WorldStateClient.Instance?.SendVerificationRequest(request, callback);
}
```

---

## Data Models

### SequenceDataModels.cs

**SequenceResult** - Result from SequenceServer:
```csharp
[System.Serializable]
public class SequenceResult
{
    public string status;            // "success", "partial", "error"
    public string message;           // Human-readable description
    public List<Operation> operations; // Parsed operations
    public uint request_id;          // Protocol V2 correlation ID
}

[System.Serializable]
public class Operation
{
    public string operation_name;    // "move_to_coordinate", "control_gripper", etc.
    public string robot_id;          // Target robot
    public Dictionary<string, object> parameters; // Operation-specific params
}
```

### DetectionDataModels.cs

**DetectionResult** - Object detection result:
```csharp
[System.Serializable]
public class DetectionResult
{
    public string status;            // "success", "no_objects", "error"
    public List<DetectedObject> objects;
    public string camera_id;
    public uint request_id;
}

[System.Serializable]
public class DetectedObject
{
    public string object_id;
    public string color;
    public Vector3 position;         // 3D world position (if stereo)
    public float confidence;         // 0.0 - 1.0
    public BoundingBox bbox;         // 2D image bounding box
}
```

### RAGDataModels.cs

**OperationContext** - RAG query result:
```csharp
[System.Serializable]
public class OperationContext
{
    public string operation_name;
    public string description;
    public List<string> parameters;
    public string example;
    public float similarity_score;
}
```

### AutoRTDataModels.cs

**ProposedTask** - Task proposal from AutoRT:
```csharp
[System.Serializable]
public class ProposedTask
{
    public string task_id;               // Unique identifier (e.g., "task_001")
    public string description;           // Human-readable task description
    public List<TaskOperation> operations; // Operations to execute
    public List<string> required_robots; // Robot IDs needed
    public int estimated_complexity;     // 1-10 complexity score
}

[System.Serializable]
public class TaskOperation
{
    public string type;                  // Operation name from Registry
    public string robot_id;              // Target robot
    public Dictionary<string, object> parameters; // Operation parameters
}
```

**AutoRTResponse** - Response from Python AutoRT:
```csharp
[System.Serializable]
public class AutoRTResponse
{
    public List<ProposedTask> tasks;     // Generated task proposals
    public bool loop_running;            // Current loop status
    public string error;                 // Error message if failed
    public uint request_id;              // Protocol V2 correlation

    public bool HasError => !string.IsNullOrEmpty(error);
}
```

**TaskSelectionStrategy** - Strategy enum:
```csharp
public enum TaskSelectionStrategy
{
    Balanced,   // Mix of simple and complex tasks
    Simple,     // Prioritize low-complexity tasks
    Complex,    // Prioritize challenging tasks
    Random      // Random sampling
}
```

---

## Communication Flow Examples

### Example 1: Natural Language Command Sequence

```
USER: "Detect the red cube and move to it"
  │
  ├─> SequenceClient.ExecuteSequence("Detect the red cube and move to it")
  │       │
  │       └─> Python SequenceServer (port 5013)
  │               ├─ CommandParser + RAG (semantic operation matching)
  │               └─ Returns: SequenceResult with 2 operations:
  │                   1. detect_object(color="red", shape="cube")
  │                   2. move_relative_to_object(target="$detected_object")
  │
  ├─> SequenceClient receives SequenceResult (via ResultsClient)
  │
  ├─> PythonCommandHandler.HandleSequence(result)
  │       ├─ Execute operation 1: detect_object
  │       │   └─> Returns: {"object_id": "Cube_01", "position": [0.3, 0.2, 0.1]}
  │       │
  │       └─ Execute operation 2: move_relative_to_object
  │           └─> RobotController.SetTarget([0.3, 0.2, 0.1])
  │
  └─> Robot moves to detected cube
```

### Example 2: World State Streaming for Spatial Reasoning

```
Unity Scene                     Python WorldStateServer
─────────────                   ───────────────────────

WorldStatePublisher
  │ (every 0.5s)
  │
  ├─> Collect robot states
  │   ├─ Robot1: pos=(0.2, 0.15, 0.3), gripper=open, moving=true
  │   └─ Robot2: pos=(0.5, 0.18, 0.4), gripper=closed, moving=false
  │
  ├─> Collect object states
  │   └─ BlueCube: pos=(0.35, 0.2, 0.25), confidence=0.95
  │
  ├─> WorldStateClient.SendWorldState(update)
  │       │
  │       └──────────────────────────────> WorldStateServer receives update
  │                                        ├─ Updates internal WorldState
  │                                        ├─ Enables spatial operations:
  │                                        │   • move_relative_to_object
  │                                        │   • check_collision
  │                                        │   • verify_handoff_distance
  │                                        └─ Fires OnWorldStateUpdated event
```

### Example 3: Coordination Verification

```
Unity Scene                     Python Backend
─────────────                   ──────────────

CollaborativeStrategy
  │
  ├─> Request handoff verification
  │   PythonCoordinationVerifier.VerifyCoordination(
  │       robot1Id: "Robot1",
  │       robot2Id: "Robot2",
  │       operation: "handoff"
  │   )
  │       │
  │       └──────────────────────> Python WorldState
  │                                ├─ Check robot distance
  │                                ├─ Check gripper states
  │                                ├─ Check collision potential
  │                                └─ Returns: {success: true}
  │       <──────────────────────┘
  │
  └─> Callback receives result
      └─> Execute handoff sequence
```

### Example 4: Autonomous Task Generation (AutoRT)

```
Unity Inspector                 Python AutoRT                    Unity Execution
───────────────                 ─────────────                    ───────────────

User clicks "Generate Tasks"
  │
  ├─> AutoRTManager.GenerateTasks(3)
  │       │
  │       └──────────────────────> SequenceServer (port 5013)
  │                                ├─ Query WorldStateServer for scene
  │                                │   └─ Returns: 2 red cubes, 1 blue cube
  │                                │
  │                                ├─ TaskGenerator builds LLM prompt:
  │                                │   • Scene: red cubes at [0.3,0.2,0.1], [0.5,0.2,0.1]
  │                                │   • Available: 30 operations from Registry
  │                                │   • Robots: Robot1, Robot2
  │                                │   • Strategy: Balanced
  │                                │
  │                                ├─ Query LM Studio LLM
  │                                │   └─ Generates 3 task proposals
  │                                │
  │                                ├─ Validate operations (Registry check)
  │                                ├─ Apply Pydantic schema validation
  │                                └─ Returns: AutoRTResponse with tasks
  │       <──────────────────────┘
  │
  ├─> AutoRTManager.OnResponseReceived()
  │   └─ Adds tasks to _pendingTasks list
  │
  ├─> Inspector UI updates (custom editor)
  │   Shows 3 pending tasks:
  │   ┌─────────────────────────────────┐
  │   │ task_001: Pick up first red cube │
  │   │   [ Execute ] [ Reject ]          │
  │   ├─────────────────────────────────┤
  │   │ task_002: Handoff to Robot2      │
  │   │   [ Execute ] [ Reject ]          │
  │   └─────────────────────────────────┘
  │
User clicks "Execute" on task_001
  │
  ├─> AutoRTManager.ExecuteTask(task_001)
  │       │
  │       └──────────────────────> SequenceServer
  │                                └─ Executes task operations:
  │                                    1. detect_object_stereo(color="red")
  │                                    2. move_to_coordinate([0.3,0.2,0.1])
  │                                    3. control_gripper(action="close")
  │                                <──────────────────────┘
  │                                                        │
  │                                                        └──────> PythonCommandHandler
  │                                                                ├─ Detect red cube
  │                                                                ├─ Move to position
  │                                                                └─ Close gripper
  │
Task executed successfully!
```

**Continuous Loop Mode**:

```
Unity Inspector                 Python AutoRT (Loop Thread)      Unity Execution
───────────────                 ───────────────────────────      ───────────────

User clicks "Start Loop"
  │
  ├─> AutoRTManager.StartLoop(delay=5.0)
  │       │
  │       └──────────────────────> SequenceServer
  │                                └─ Starts background loop thread
  │                                    while loop_running:
  │                                        ├─ Generate tasks (3 candidates)
  │                                        ├─ Send to Unity (AUTORT_RESPONSE)
  │       <──────────────────────────────┤
  │                                        ├─ Wait for approval/rejection
  │                                        └─ sleep(5.0 seconds)
  │
  ├─> Tasks appear in inspector UI
  │   User reviews and clicks "Execute" or "Reject"
  │
  └─> Loop continues every 5 seconds...

User clicks "Stop Loop"
  │
  └─> AutoRTManager.StopLoop()
          │
          └──────────────────────> SequenceServer
                                   └─ Sets loop_running = False
                                       (loop thread exits)
```

---

## Configuration and Constants

All network configuration is centralized in `Constants.cs` (`Core` namespace):

```csharp
public static class CommunicationConstants
{
    public const string SERVER_HOST = "127.0.0.1";

    // Active ports (February 2026 unified architecture)
    public const int IMAGE_SERVER_PORT = 5005;           // Single image streaming
    public const int STEREO_IMAGE_PORT = 5006;           // Stereo image pairs
    public const int COMMAND_SERVER_PORT = 5010;         // Bidirectional commands/results
    public const int SEQUENCE_SERVER_PORT = 5013;        // Multi-command sequences + AutoRT
    public const int WORLD_STATE_PORT = 5014;            // World state streaming

    // Legacy ports (deprecated, kept for reference)
    // public const int LLM_RESULTS_PORT = 5010;         // Now COMMAND_SERVER_PORT
    // public const int DETECTION_PORT = 5006;           // Now STEREO_IMAGE_PORT

    // Protocol settings
    public const int MAX_JSON_LENGTH = 10 * 1024 * 1024; // 10MB
    public const float RECONNECT_INTERVAL = 2f;          // 2 seconds
    public const int MAX_ITEMS_PER_FRAME = 50;           // Response processing limit
}
```

**AutoRT Configuration** (`ConfigScripts/AutoRTConfig.cs`):

```csharp
[CreateAssetMenu(fileName = "AutoRTConfig", menuName = "Robotics/AutoRT Config")]
public class AutoRTConfig : ScriptableObject
{
    [Header("Task Generation")]
    [Range(1, 5)]
    public int maxTaskCandidates = 3;
    public TaskSelectionStrategy strategy = TaskSelectionStrategy.Balanced;

    [Header("Continuous Loop")]
    public bool enableContinuousLoop = false;
    [Range(1f, 60f)]
    public float loopDelaySeconds = 5f;

    [Header("Robot Assignment")]
    public string[] robotIds = new[] { "Robot1", "Robot2" };
    public bool enableCollaborativeTasks = true;

    [Header("UI Settings")]
    [Range(5, 20)]
    public int maxDisplayTasks = 10;
    public bool autoRefresh = true;
    [Range(0.1f, 2f)]
    public float uiRefreshRate = 0.5f;
}
```

**Python AutoRT Configuration** (`ACRLPython/config/AutoRT.py`):

```python
# LLM configuration
LM_STUDIO_URL = "http://localhost:1234/v1"
TASK_GENERATION_MODEL = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
SAFETY_VALIDATION_MODEL = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"

# AutoRT loop settings
MAX_TASK_CANDIDATES = 3
LOOP_DELAY_SECONDS = 5.0
HUMAN_IN_LOOP_DEFAULT = True
USE_VLM_REASONING = False

# Safety settings
WORKSPACE_BOUNDS = {
    'min_corner': (-1.0, -1.0, 0.0),
    'max_corner': (1.0, 1.0, 1.5),
}
MAX_VELOCITY = 2.0  # m/s
MIN_ROBOT_SEPARATION = 0.2  # meters
MAX_GRIPPER_FORCE = 50.0  # Newtons

# Multi-robot settings
DEFAULT_ROBOTS = ["Robot1", "Robot2"]
ENABLE_COLLABORATIVE_TASKS = True
```

---

## Design Patterns and Best Practices

### 1. Singleton Pattern (All Managers)

All high-level clients use the singleton pattern for global access:

```csharp
public static SequenceClient Instance { get; private set; }

private void Awake()
{
    if (Instance == null)
    {
        Instance = this;
        DontDestroyOnLoad(gameObject);
        base.Awake();
    }
    else { Destroy(gameObject); }
}
```

**Why**: Ensures only one instance exists and provides global access point. `DontDestroyOnLoad` ensures persistence across scene transitions.

### 2. Event-Driven Architecture

Components communicate via C# events to reduce coupling:

```csharp
// Publisher
public event Action<SequenceResult> OnSequenceResultReceived;

// Subscriber
SequenceClient.Instance.OnSequenceResultReceived += result => {
    Debug.Log($"Sequence completed: {result.status}");
};
```

**Benefits**: Decouples components, enables multiple subscribers, simplifies testing.

### 3. Coroutine-Based Async Execution

Command execution uses Unity coroutines for async operation:

```csharp
private IEnumerator ExecuteMoveToCoordinate(RobotCommand command)
{
    // Set target
    robot.controller.SetTarget(targetPosition);

    // Wait for completion
    yield return new WaitUntil(() => robot.controller.TargetReached);

    // Send completion
    SendCompletion(command, success: true);
}
```

**Why**: Unity's main thread requires coroutines for async operations. This pattern enables non-blocking execution while maintaining Unity API access.

### 4. Strategy Pattern (Coordination Verification)

Coordination verification uses the strategy pattern for swappable verification methods:

```csharp
ICoordinationVerifier verifier;

// Choose strategy at runtime
if (useLocalVerification)
    verifier = new UnityCoordinationVerifier();
else
    verifier = new PythonCoordinationVerifier();

// Verify using selected strategy
verifier.VerifyCoordination(robot1, robot2, "handoff", callback);
```

**Benefits**: Easy to add new verification strategies, testable in isolation.

### 5. Command Pattern (Robot Commands)

Robot commands use the command pattern for encapsulation and queuing:

```csharp
[System.Serializable]
public class RobotCommand
{
    public string command_type;
    public string robot_id;
    public CommandParameters parameters;
    public uint request_id;
}
```

**Benefits**: Commands are first-class objects, can be queued/logged/replayed, supports undo/redo.

---

## Error Handling

### 1. Connection Failures

Clients automatically reconnect on connection loss:

```csharp
// Inherited from TCPClientBase
private void Update()
{
    if (_autoReconnect && !IsConnected && !_isReconnecting)
    {
        _reconnectTimer += Time.deltaTime;
        if (_reconnectTimer >= CommunicationConstants.RECONNECT_INTERVAL)
        {
            _reconnectTimer = 0f;
            ConnectAsync();
        }
    }
}
```

### 2. JSON Parsing Errors

All JSON parsing uses `JsonParser.TryParse` for safe error handling:

```csharp
if (JsonParser.TryParseWithLogging<RobotCommand>(json, out var command, "[HANDLER]"))
{
    // Use command
}
else
{
    // Error already logged
    SendCompletion(command, success: false);
}
```

### 3. Command Execution Failures

Commands include timeout protection:

```csharp
private IEnumerator ExecuteWithTimeout(RobotCommand command, float timeout)
{
    float elapsed = 0f;

    while (!robot.controller.TargetReached && elapsed < timeout)
    {
        elapsed += Time.deltaTime;
        yield return null;
    }

    if (elapsed >= timeout)
    {
        Debug.LogWarning($"Command timeout: {command.command_type}");
        SendCompletion(command, success: false);
    }
}
```

### 4. Missing Components

Defensive null checks before accessing components:

```csharp
var robot = RobotManager.Instance?.GetRobot(robotId);
if (robot == null)
{
    Debug.LogError($"Robot not found: {robotId}");
    SendCompletion(command, success: false);
    return;
}
```

---

## Performance Optimization

### 1. Frame Budget Management

Response processing is limited to 50 items per frame to prevent frame drops:

```csharp
// In BidirectionalClientBase.ProcessResponseQueue()
const int MAX_ITEMS_PER_FRAME = 50;
int processedCount = 0;

while (_responseQueue.Count > 0 && processedCount < MAX_ITEMS_PER_FRAME)
{
    TResponse response = _responseQueue.Dequeue();
    ProcessResponse(response);
    processedCount++;
}
```

### 2. Background Threading

All blocking I/O happens on background threads (inherited from `BidirectionalClientBase`):

```csharp
protected override void OnConnected()
{
    _receiveShouldRun = true;
    _receiveThread = new Thread(ReceiveLoop);
    _receiveThread.Start();
}
```

### 3. Update Rate Control

World state publishing uses configurable update rate:

```csharp
[SerializeField]
private float _updateRate = 2.0f; // 2 Hz

private void Update()
{
    _timer += Time.deltaTime;
    if (_timer >= 1f / _updateRate)
    {
        _timer = 0f;
        PublishWorldState();
    }
}
```

### 4. Object Pooling (Recommended)

For high-frequency commands, consider using object pooling:

```csharp
// TODO: Implement command pooling for reduced GC pressure
private Queue<RobotCommand> _commandPool = new Queue<RobotCommand>();

private RobotCommand GetPooledCommand()
{
    return _commandPool.Count > 0 ? _commandPool.Dequeue() : new RobotCommand();
}
```

---

## Testing

### Unit Tests

See `ACRLUnity/Assets/Tests/PlayMode/` for integration tests:

- **TCPClientTests.cs** - Connection lifecycle, threading, error handling
- **UnityProtocolTests.cs** - Message encoding/decoding validation
- **CoordinationIntegrationTests.cs** - Multi-robot coordination flows

### Manual Testing via Editor

**SequenceClient** includes custom inspector for testing:

1. Open Unity Editor
2. Select `SequenceClient` GameObject in scene
3. Enter command in text area (e.g., `"move to (0.3, 0.2, 0.1)"`)
4. Click "Send Command" button
5. Watch console for results and robot movement

**WorldStatePublisher** debug logging:

```csharp
[SerializeField]
private bool _enableDebugLogging = true;

private void PublishWorldState()
{
    var update = CollectWorldState();

    if (_enableDebugLogging)
    {
        Debug.Log($"Publishing state: {update.robots.Count} robots, {update.objects.Count} objects");
    }

    WorldStateClient.Instance?.SendWorldState(update);
}
```

### Python Backend Testing

Test Python servers independently:

```bash
cd ACRLPython

# Start unified backend
python -m orchestrators.RunRobotController

# In separate terminal, test SequenceServer
python -m tests.TestSequenceServer

# Test CommandServer
python -m tests.TestCommandServer
```

---

## Usage Examples

### Basic Setup

**Scene Hierarchy** (minimal setup):

```
Scene
├── SimulationManager
├── RobotManager
├── PythonCommunication (Empty GameObject)
│   ├── SequenceClient (Component)
│   ├── AutoRTManager (Component)           # ✅ NEW: Autonomous task generation
│   ├── UnifiedPythonReceiver (Component)
│   ├── WorldStatePublisher (Component)
│   ├── WorldStateClient (Component)
│   └── PythonCommandHandler (Component)
└── Robots (AR4 prefabs)
```

**Optional Setup** (with AutoRT):

```
Scene
├── SimulationManager
├── RobotManager
├── PythonCommunication (Empty GameObject)
│   ├── SequenceClient
│   ├── AutoRTManager                       # Autonomous task generation
│   │   └── Config: DefaultAutoRTConfig.asset
│   ├── UnifiedPythonReceiver
│   ├── WorldStatePublisher
│   ├── WorldStateClient
│   └── PythonCommandHandler
└── Robots (AR4 prefabs)
```

**Initialization Order**:

1. `RobotManager` initializes first (Awake)
2. `SequenceClient`, `UnifiedPythonReceiver`, `WorldStateClient` connect to Python (Start)
3. `WorldStatePublisher` begins publishing (Update loop)
4. `PythonCommandHandler` subscribes to command events

### Example 1: Send Natural Language Command

```csharp
using PythonCommunication;

public class RobotCommandTester : MonoBehaviour
{
    void Start()
    {
        // Simple command
        SequenceClient.Instance.ExecuteSequence(
            "move to the red cube and close the gripper"
        );

        // With robot ID
        SequenceClient.Instance.ExecuteSequence(
            "detect all cubes",
            robotId: "Robot1"
        );

        // Subscribe to results
        SequenceClient.Instance.OnSequenceResultReceived += result => {
            if (result.success)
            {
                Debug.Log($"Completed {result.completed_commands} commands");
            }
        };

        // Using convenience methods
        SequenceClient.Instance.Pick(0.3f, 0.2f, 0.05f);
    }
}
```

### Example 2: Subscribe to LLM Results

```csharp
using PythonCommunication;

public class VisionResultHandler : MonoBehaviour
{
    void Start()
    {
        UnifiedPythonReceiver.Instance.OnLLMResultReceived += HandleVisionResult;
    }

    void HandleVisionResult(LLMResult result)
    {
        Debug.Log($"Vision analysis: {result.response}");

        // Extract detected objects
        foreach (var obj in result.detected_objects)
        {
            Debug.Log($"  - {obj.color} {obj.object_type} at {obj.position}");
        }
    }

    void OnDestroy()
    {
        if (UnifiedPythonReceiver.Instance != null)
        {
            UnifiedPythonReceiver.Instance.OnLLMResultReceived -= HandleVisionResult;
        }
    }
}
```

### Example 3: Coordination Verification

```csharp
using PythonCommunication;

public class HandoffCoordinator : MonoBehaviour
{
    private ICoordinationVerifier _verifier;

    void Start()
    {
        // Choose verification strategy
        _verifier = new PythonCoordinationVerifier(); // Use Python world state
        // or
        // _verifier = new UnityCoordinationVerifier(); // Use local Unity checks
    }

    public void AttemptHandoff(string robot1, string robot2)
    {
        _verifier.VerifyCoordination(
            robot1,
            robot2,
            "handoff",
            callback: (success, message) => {
                if (success)
                {
                    Debug.Log("Coordination verified - executing handoff");
                    ExecuteHandoff(robot1, robot2);
                }
                else
                {
                    Debug.LogWarning($"Coordination failed: {message}");
                }
            }
        );
    }

    private void ExecuteHandoff(string robot1, string robot2)
    {
        // Send handoff command sequence
        SequenceClient.Instance.ExecuteSequence(
            $"robot {robot1}: open gripper; robot {robot2}: close gripper"
        );
    }
}
```

### Example 4: Track Custom Objects in World State

```csharp
using PythonCommunication;

public class ObjectTracker : MonoBehaviour
{
    void Start()
    {
        // Track specific objects for Python spatial reasoning
        GameObject redCube = GameObject.Find("RedCube");
        GameObject blueCube = GameObject.Find("BlueCube");

        WorldStatePublisher.Instance.TrackObject(redCube, "red_cube");
        WorldStatePublisher.Instance.TrackObject(blueCube, "blue_cube");

        Debug.Log("Objects registered for world state tracking");
    }
}
```

### Example 5: Autonomous Task Generation with AutoRT

```csharp
using PythonCommunication;
using PythonCommunication.DataModels;

public class AutoRTController : MonoBehaviour
{
    [SerializeField] private AutoRTConfig _config;
    private bool _autonomousModeActive = false;

    void Start()
    {
        // Subscribe to AutoRT events
        AutoRTManager.Instance.OnTasksReceived += HandleTasksReceived;
        AutoRTManager.Instance.OnLoopStatusChanged += HandleLoopStatusChanged;

        Debug.Log("AutoRT Controller initialized");
    }

    // Manual task generation (one-shot)
    public void GenerateTasksOnce()
    {
        if (AutoRTManager.Instance.GenerateTasks(numTasks: 3))
        {
            Debug.Log("Task generation request sent");
        }
    }

    // Start continuous autonomous mode
    public void StartAutonomousMode()
    {
        if (AutoRTManager.Instance.StartLoop(loopDelay: 5f))
        {
            _autonomousModeActive = true;
            Debug.Log("Autonomous mode started - tasks will generate every 5 seconds");
        }
    }

    // Stop continuous mode
    public void StopAutonomousMode()
    {
        if (AutoRTManager.Instance.StopLoop())
        {
            _autonomousModeActive = false;
            Debug.Log("Autonomous mode stopped");
        }
    }

    // Handle task proposals
    void HandleTasksReceived(List<ProposedTask> tasks)
    {
        Debug.Log($"Received {tasks.Count} task proposals:");

        foreach (var task in tasks)
        {
            Debug.Log($"\n[{task.task_id}] {task.description}");
            Debug.Log($"  Complexity: {task.estimated_complexity}/10");
            Debug.Log($"  Robots: {string.Join(", ", task.required_robots)}");
            Debug.Log($"  Operations: {task.operations.Count}");

            // Automatic approval based on complexity
            if (task.estimated_complexity <= 5)
            {
                Debug.Log($"  → Auto-approving (complexity <= 5)");
                AutoRTManager.Instance.ExecuteTask(task);
            }
            else
            {
                Debug.Log($"  → Requires manual approval (complexity > 5)");
                // Task will appear in inspector UI for manual review
            }
        }
    }

    // Handle loop status changes
    void HandleLoopStatusChanged(bool isRunning)
    {
        Debug.Log($"Loop status changed: {(isRunning ? "RUNNING" : "STOPPED")}");
        _autonomousModeActive = isRunning;
    }

    // Custom safety validation
    bool IsTaskSafe(ProposedTask task)
    {
        // Check complexity
        if (task.estimated_complexity > 8)
        {
            Debug.LogWarning($"Task too complex: {task.estimated_complexity}");
            return false;
        }

        // Check operation types
        foreach (var op in task.operations)
        {
            // Disallow certain high-risk operations
            if (op.type == "force_control" || op.type == "emergency_stop")
            {
                Debug.LogWarning($"Unsafe operation detected: {op.type}");
                return false;
            }
        }

        // Check robot availability
        foreach (var robotId in task.required_robots)
        {
            var robot = RobotManager.Instance?.GetRobot(robotId);
            if (robot == null || !robot.IsInitialized)
            {
                Debug.LogWarning($"Robot not available: {robotId}");
                return false;
            }
        }

        return true;
    }

    // Approve task with custom validation
    public void ApproveTaskWithValidation(ProposedTask task)
    {
        if (IsTaskSafe(task))
        {
            AutoRTManager.Instance.ExecuteTask(task);
            Debug.Log($"Approved and executing: {task.task_id}");
        }
        else
        {
            AutoRTManager.Instance.RejectTask(task);
            Debug.LogWarning($"Rejected unsafe task: {task.task_id}");
        }
    }

    void OnDestroy()
    {
        // Clean up event subscriptions
        if (AutoRTManager.Instance != null)
        {
            AutoRTManager.Instance.OnTasksReceived -= HandleTasksReceived;
            AutoRTManager.Instance.OnLoopStatusChanged -= HandleLoopStatusChanged;
        }

        // Stop autonomous mode if active
        if (_autonomousModeActive)
        {
            StopAutonomousMode();
        }
    }
}
```

**Using the Custom Inspector**:

1. Add `AutoRTManager` component to a GameObject
2. Assign `AutoRTConfig` asset to the Config field
3. Enter play mode
4. Use inspector buttons:
   - **Generate Tasks** - Request task proposals from Python
   - **Start Loop** - Begin continuous autonomous generation
   - **Execute** - Approve and run a specific task
   - **Reject** - Discard a task without running it
   - **Clear All Tasks** - Remove all pending tasks

**Programmatic Control**:

```csharp
// From code
AutoRTManager.Instance.GenerateTasks(3);

// Wait for tasks to arrive via OnTasksReceived event
// Then approve/reject programmatically
AutoRTManager.Instance.ExecuteTask(selectedTask);
AutoRTManager.Instance.RejectTask(unwantedTask);
```

---

## Troubleshooting

### Common Issues

**1. Commands not executing**

```
Symptom: SequenceClient shows "Connected" but commands have no effect
Solution:
  - Check PythonCommandHandler is attached to scene
  - Verify RobotManager has robots registered
  - Check Python backend logs for parsing errors
  - Enable _logCommands in SequenceClient inspector
```

**2. Connection failures**

```
Symptom: Client shows "Disconnected" repeatedly
Solution:
  - Verify Python backend is running: python -m orchestrators.RunRobotController
  - Check ports in Constants.cs match Python LLMConfig.py
  - Ensure no firewall blocking localhost connections
  - Check Unity console for connection error messages
```

**3. World state not updating in Python**

```
Symptom: Python operations fail with "world state unavailable"
Solution:
  - Verify WorldStatePublisher is enabled in inspector
  - Check WorldStateClient shows "Connected" status
  - Increase update rate (_updateRate) for faster updates
  - Enable debug logging to verify state publishing
```

**4. Request ID mismatches**

```
Symptom: Responses seem random or out of order
Solution:
  - This should not happen with Protocol V2
  - Verify both Unity and Python use same protocol version
  - Check UnityProtocol.VERSION == 2
  - Check Python core/UnityProtocol.py VERSION == 2
```

**5. High latency**

```
Symptom: Commands execute slowly
Solution:
  - Reduce world state update rate (lower _updateRate)
  - Check network performance (even localhost has overhead)
  - Profile Python backend for slow operations
  - Consider using UnityCoordinationVerifier for local checks
```

---

## Migration Guide

### From Legacy Architecture (Pre-December 2025)

**Old (6+ servers)**:
```csharp
// Separate clients for each server
StreamingClient.Instance.SendImage(...);
DetectionClient.Instance.SendStereoImage(...);
ResultsReceiver.Instance.ReceiveResults();
StatusClient.Instance.QueryStatus(...);
RAGClient.Instance.QueryRAG(...);
```

**New (4 servers)**:
```csharp
// Unified clients with clear responsibilities
SequenceClient.Instance.ExecuteSequence("detect and move to cube");
// Result automatically routed to PythonCommandHandler
// World state automatically published by WorldStatePublisher
```

**Changes**:
1. **ImageServer** (ports 5005/5006) replaces `StreamingServer` + `StereoDetectionServer`
2. **CommandServer** (port 5010) replaces `ResultsServer` + `StatusServer`
3. **SequenceServer** (port 5013) remains, now with integrated RAG
4. **WorldStateServer** (port 5014) new - separate from command channel

**Migration Steps**:
1. Remove old client components from scene
2. Add new unified components (see Basic Setup)
3. Update port constants in `Constants.cs`
4. Update Python backend to use `RunRobotController.py`
5. Test with simple command sequence

---

## Summary

The **PythonCommunication** system provides a production-ready, high-level interface for Unity-Python robot control:

- **SequenceClient**: Natural language command parsing and execution
- **AutoRTManager**: Autonomous task generation with LLM planning and human-in-the-loop approval (February 2026)
- **ResultsClient**: Bidirectional command results
- **UnifiedPythonReceiver**: Unified result routing and event dispatch
- **WorldStatePublisher/Client**: Real-time state streaming for spatial reasoning
- **PythonCommandHandler**: Async command execution on Unity robots
- **Coordination Verification**: Dual-robot safety verification (local or remote)

### Key Features (February 2026 Update)

**AutoRT System**:
- LLM-powered autonomous task generation via LM Studio
- Human-in-the-loop approval workflow with custom inspector UI
- Continuous loop mode for fully autonomous operation
- Multi-robot collaborative task support
- Task validation against 30 registered operations
- Configurable task selection strategies (Balanced/Simple/Complex/Random)
- Safety constraints (workspace bounds, velocity limits, robot separation)
- Pydantic schema validation for type safety

**Protocol V2**:
- Request ID correlation for reliable multi-robot communication
- Dedicated message types (AUTORT_COMMAND, AUTORT_RESPONSE)
- Thread-safe request/response matching
- Shared port architecture (SequenceServer + AutoRT on port 5013)

**30 Operations System**:
- Organized by complexity levels (1-5)
- Validated operation types in task generation
- Registry integration for semantic matching
- Variable passing between operations
