# Auto-Cooperative Robot Learning

A Unity-based simulation environment for dual AR4 robotic arms that collaboratively solve tasks through LLM-driven multi-agent coordination. This project is part of a master's thesis exploring autonomous cooperative behavior in robotic systems.

## Description

The goal of this project is to have two AR4 robot arms positioned side by side that collaboratively solve tasks which would be impossible for a single robot to accomplish. The system uses inverse kinematics control, LLM-driven task planning, multi-robot coordination patterns, and vision-based object detection.

**Key Features**:
- Unity 6000.3.0f1 simulation environment with physics-based ArticulationBody robots
- Damped least-squares inverse kinematics (6-DOF control)
- Multiple coordination modes: Independent (✅), Sequential (✅), Collaborative (⚠️ partial), Master-Slave (❌), Distributed (❌)
- **Unified Python Backend**: Single entry point (RunRobotController) orchestrates all servers
- **Operations System**: 29 registered operations including atomic actions, perception, and sync primitives
- **AutoRT System**: Autonomous task generation with LLM-based planning and human-in-the-loop approval
- **Self-Improvement Loop**: Dynamic runtime code generation, structure/syntax validation, sandbox execution, and success/failure outcome tracking
- **Knowledge Graph**: Dynamic relation tracking for tracking complex topological environment states
- **ROS 2 & Docker Integration**: Physical robot control capabilities via `ROSMotionClient` and containerized ROS deployments
- **Advanced Python Grasp Planning**: Approach-aware motion (Top/Front/Side) generation and scoring via Python backend
- LLM vision integration (Ollama) for scene understanding and natural language commands
- Object detection with YOLO streaming support
- **Stereo Vision & VGN**: 3D object localization, stereo depth map reconstruction, and VGN-based local grasp network
- **camera/ & hardware/ Abstraction**: Sim↔real switching via `--env sim|real` flag; no code changes required
- **Web UI (Mission Control)**: Optional dashboard served via `--web PORT`; REST/WebSocket endpoints
- **Protocol V2**: Request ID correlation for reliable multi-robot communication
- **RAG System**: Integrated semantic search for operation matching in natural language commands
- JSONL logging system for execution data collection
- Python-Unity TCP communication with persistent connections and health checks

## Getting Started

### Prerequisites

- **Unity Hub** with Unity Editor **6000.3.0f1** (exact version required)
- **Python 3.8+** with virtual environment support
- **Git** with submodule support
- **Ollama** (optional, for LLM vision features)

### Dependencies

**Unity Packages** (managed via Package Manager):
- NuGetForUnity (for MathNet.Numerics)
- Unity Input System (1.14.2)
- Universal Render Pipeline (17.2.0)
- Unity Test Framework (1.5.1)

**Python Dependencies**:
- numpy, matplotlib (data processing)
- opencv-python (computer vision, object detection)
- ollama (LLM vision integration)

### Installing

1. **Clone the repository with submodules**:
   ```bash
   git clone --recursive https://github.com/JanMStraub/Auto-Cooperative-Robot-Learning.git
   cd Auto-Cooperative-Robot-Learning
   ```

2. **Setup Python environment**:
   ```bash
   cd ACRLPython
   python -m venv acrl
   source acrl/bin/activate  # On Windows: acrl\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Open Unity project**:
   - Open Unity Hub
   - Add project from `ACRLUnity/` folder
   - Ensure Unity version **6000.3.0f1** is installed
   - Open the project (dependencies will auto-install)

4. **Install NuGet packages** (if not auto-installed):
   - In Unity: NuGet > Manage NuGet Packages
   - Install `MathNet.Numerics` (required for IK computation)

### Executing Program

#### Quick Start with Python Backend

1. **Start the unified Python backend** (single command):

   ```bash
   cd ACRLPython
   source acrl/bin/activate  # On Windows: acrl\Scripts\activate
   python -m orchestrators.RunRobotController
   ```

   This starts all servers: ImageServer (5005/5006), CommandServer (5010), SequenceServer (5013), WorldStateServer (5014), AutoRTServer (5015). WebUIServer (8000) is optional via `--web 8000`.

2. **Run Unity simulation**:
   - Open `ACRLUnity/Assets/Scenes/1xAR4Scene.unity` for single robot testing
   - Press Play in Unity Editor
   - Use natural language commands via SequenceClient:

     ```csharp
     SequenceClient.Instance.SendCommand("Detect the blue cube, move to it, close the gripper");
     ```

#### Running Simulations in Unity

1. **Single Robot Testing**:
   ```
   Open: ACRLUnity/Assets/Scenes/1xAR4Scene.unity
   ```

2. **Multi-Robot Simulation**:
   ```
   Open: ACRLUnity/Assets/Scenes/16xAR4Scene.unity
   ```

3. **Using SimulationManager**:
   - Select SimulationManager GameObject in hierarchy
   - Use Inspector controls: Start, Pause, Resume, Reset
   - Configure coordination mode and settings via SimulationConfig asset

#### Testing and Development

**Run Unity Tests**:
- Window > General > Test Runner
- Select PlayMode or EditMode tests
- Click "Run All" or run individual tests

**Build Standalone**:
- File > Build Settings
- Select platform (PC, Mac & Linux Standalone recommended)
- Click "Build" or "Build and Run"

## Recent Major Updates

### Self-Improvement Loop & Dynamic Operations (March 2026)

**Autonomous Code Generation & Validation** - System that dynamically generates, validates, and incorporates new operations:

**Core Features**:
- **Dynamic Operation Generation**: LLM generates new operations on-the-fly when existing ones lack required capabilities.
- **Validation Pipeline**: `SyntaxValidator`, `StructureValidator`, and `SandboxExecutor` ensure generated code is safe and structurally sound.
- **Review System**: CLI tool (`ReviewOperations.py`) for human review and approval of generated operations.
- **Execution Feedback**: `OutcomeTracker` and `FeedbackCollector` monitor success/failure rates of operations.
- **RAG Indexing**: Failed operations preserve metadata to inform future LLM generation and avoid repeating mistakes.

### VGN & Stereo Vision Integration (March 2026)

Advanced 3D perception pipeline for physical robot integration:

- **Stereo Reconstruction**: Robust stereo matching system for accurate 3D point cloud generation (`generate_point_cloud` operation).
- **VGN-based Grasping**: Local grasp network replaces GraspNet Docker service; runs entirely within Python backend.
- **camera/ Abstraction**: `Provider.py` interface switches between Unity (`UnityProvider`) and real cameras (`LocalProvider`) via `--env sim|real`.
- **YOLO Pipeline Updates**: Real-time object detection stream integration.
- **Conflict Resolution**: `ConflictResolver.py` handles ambiguous detections in crowded scenes.

### AutoRT System (February 2026)

**Autonomous Robot Task generation** - LLM-powered task planning with human oversight:

**Core Features**:
- **Autonomous Task Generation**: LLM generates diverse task proposals based on detected scene objects
- **Human-in-the-Loop**: Unity custom inspector UI for task approval/rejection before execution
- **Continuous Loop Mode**: Optional autonomous mode with configurable delay between generations
- **Multi-Robot Coordination**: Supports collaborative tasks using signal/wait primitives
- **Registry Integration**: Tasks validated against 29 registered operations
- **Pydantic Validation**: Type-safe task structures with automatic JSON schema enforcement

**Architecture**:
- **Unity Side**: `AutoRTManager` (singleton, shares port 5013 with SequenceServer)
- **Python Side**: `TaskGenerator` (LLM querying) + integration in `SequenceServer`
- **Configuration**: `AutoRTConfig.asset` (Unity) + `config/AutoRT.py` (Python)
- **Custom Editor**: Inspector UI with task list, approve/reject buttons, loop controls

**Task Selection Strategies** (configurable in `AutoRTConfig`):
- **Balanced**: Mix of simple and complex tasks
- **Simple**: Prioritize low-complexity tasks (good for testing)
- **Complex**: Prioritize challenging multi-robot coordination
- **Random**: Diverse task sampling

**Usage**:
```csharp
// In Unity Inspector (AutoRTManager component):
// 1. Click "Generate Tasks" - tasks appear in inspector UI
// 2. Review task descriptions and operations
// 3. Click "Execute" to approve and run, or "Reject" to discard
// 4. Optional: Enable "Continuous Loop" for autonomous generation

// Or programmatically:
AutoRTManager.Instance.GenerateTasks(numTasks: 3);
AutoRTManager.Instance.StartLoop(loopDelay: 5f);
AutoRTManager.Instance.ExecuteTask(selectedTask);
```

**Safety Features**:
- Workspace bounds validation
- Max velocity/force limits
- Minimum robot separation (0.2m)
- Operation type validation against Registry

### Unified Python Backend (December 2025)

### Unified Python Backend

The Python backend has been consolidated from 6+ separate servers into a single unified architecture:

**Before**: Multiple orchestrators (RunDetector, RunStereoDetector, RunAnalyzer, RunSequenceServer, RunRAGServer, RunStatusServer)

**After**: Single entry point `RunRobotController` managing consolidated servers (now 6 active + 1 optional WebUIServer)

**Benefits**:

- Single command to start all backend services
- Reduced complexity and improved maintainability
- Centralized configuration via `LLMConfig.py`
- Thread-safe image storage with `UnifiedImageStorage`
- Integrated RAG system for natural language command parsing

### Protocol V2

Major protocol upgrade adding request ID correlation:

- All messages include `request_id` (uint32) for matching queries with responses
- Prevents race conditions in multi-robot scenarios
- Persistent TCP connections with keepalive
- Health checks and automatic recovery
- Thread-safe request/response matching with dedicated queues

### Grasp Planning System

New approach-aware grasping:

- Three grasp approaches: Top, Front, Side
- Automatic approach calculation based on object geometry
- Pre-grasp positioning with configurable offset
- Automatic gripper control during grasp execution
- Integration with vision system for object detection

### Operations System

17 registered operations providing structured robot control:

- Type-safe parameter validation
- Rich metadata (descriptions, examples, failure modes)
- Variable passing between operations (`detect -> $target`)
- Precondition checking and verification
- Integrated with RAG for semantic search

## Architecture Overview

### Core Systems

**Four Singleton Managers**:
- **SimulationManager**: Top-level orchestrator controlling simulation state and coordination modes
- **RobotManager**: Robot lifecycle management, configuration loading, target assignment
- **MainLogger**: Unified logging system for execution data with action tracking and trajectories
- **AutoRTManager**: Autonomous task generation client with human-in-the-loop approval UI (port 5013)

**Robot Control Layers**:
1. **RobotController**: Inverse kinematics computation using damped least-squares method
2. **GripperController**: End-effector control with open/close commands

**Vision & Perception Systems**:

- **LLM Vision** (Ollama): Scene understanding and natural language descriptions
- **Object Detection**: Color-based HSV segmentation + YOLO streaming support
- **Stereo Depth**: 3D localization using stereo disparity estimation
- **UnifiedImageStorage**: Thread-safe singleton for centralized image access

**Python Backend Architecture (February 2026, updated March 2026)**:

- **Unified Entry Point**: `RunRobotController` orchestrates all servers
- **6 Active Servers**:
  - **ImageServer** (5005/5006): Unified single and stereo image receiver
  - **CommandServer** (5010): Bidirectional commands and completions
  - **SequenceServer** (5013): Multi-command sequence orchestration + AutoRT integration
  - **WorldStateServer** (5014): Robot/object state streaming
  - **AutoRTServer** (5015): Autonomous task generation
  - **WebUIServer** (8000, optional): Mission Control dashboard (`--web PORT`)
- **camera/ & hardware/ Abstraction**: Sim↔real switching without code changes (`--env sim|real`)
- **AutoRT Module**: LLM-based autonomous task generation with Pydantic validation
- **Protocol V2**: Request ID correlation prevents race conditions in multi-robot scenarios
- **Persistent Connections**: TCP keepalive with health checks and automatic recovery

**LLM-Driven Control Systems**:

- **Operations System**: 29 registered operations organized by complexity (Levels 1-5)
  - **Level 1-2 Basic** (19 ops): Navigation, gripper control, perception (incl. `generate_point_cloud`), field detection, sync primitives
  - **Level 3 Intermediate** (6 ops): `grasp_object`, `align_object`, `move_to_region`, `follow_path`, `move_relative_to_object`, `move_between_objects`
  - **Level 4 Multi-Robot** (3 ops): `detect_other_robot`, `mirror_movement`, `grasp_object_for_handoff`
  - **Level 5 Collaborative** (1 op): `stabilize_object`
  - Variable passing: `detect -> $target`, then `move to $target`
- **AutoRT System**: Autonomous task generation with LLM planning and human approval workflow
- **Integrated RAG System**: Semantic search using LM Studio embeddings for natural language command parsing
- **CommandParser**: LLM/regex hybrid parser with operation registry matching
- **SequenceExecutor**: Sequential operation execution with state tracking

**Data Logging**:
- JSONL logging per robot or per session
- Export format with action types, trajectories, and metrics
- Thread-safe concurrent writes for multi-robot scenarios

### Key Directories

```
Auto-Cooperative-Robot-Learning/
├── ACRLUnity/                           # Unity project root
│   ├── Assets/
│   │   ├── Configuration/               # Robot, simulation, and grasp config assets
│   │   ├── Data/                        # Runtime data assets
│   │   ├── Scenes/                      # 1xAR4Scene, 16xAR4Scene
│   │   ├── Scripts/                     # C# source code
│   │   │   ├── ConfigScripts/           # ScriptableObject configs
│   │   │   ├── Logging/                 # Data logging system
│   │   │   ├── PythonCommunication/     # TCP clients and Protocol V2
│   │   │   ├── RobotScripts/            # Robot control and IK
│   │   │   ├── SimulationScripts/       # Coordination strategies
│   │   │   └── *.cs                     # Core controllers and managers
│   │   └── Prefabs/                     # Robot and environment prefabs
│   ├── Packages/                        # Unity package dependencies
│   └── ProjectSettings/                 # Unity project settings
├── ACRLPython/                          # Python backend (February 2026, updated March 2026)
│   ├── core/                            # TCPServerBase, UnityProtocol V2, Imports, LoggingSetup
│   ├── camera/                          # ✅ Sim↔real camera abstraction (--env flag)
│   │   ├── Provider.py                  # Abstract CameraProvider interface
│   │   ├── UnityProvider.py             # Adapter for Unity ImageStorage
│   │   └── LocalProvider.py             # Adapter for real cameras (USB/RealSense)
│   ├── hardware/                        # ✅ Sim↔real robot hardware abstraction
│   │   ├── Interface.py                 # Abstract RobotHardwareInterface
│   │   ├── UnityInterface.py            # Adapter for Unity robot control
│   │   └── ROSInterface.py              # Adapter for ROS/MoveIt control
│   ├── servers/                         # 6 active servers (+ 1 optional)
│   │   ├── ImageServer.py               # ✅ Unified image receiver (5005/5006)
│   │   ├── CommandServer.py             # ✅ Bidirectional commands (5010)
│   │   ├── SequenceServer.py            # ✅ Multi-command sequences (5013)
│   │   ├── WorldStateServer.py          # ✅ Robot/object state streaming (5014)
│   │   ├── AutoRTServer.py              # ✅ Autonomous task generation (5015)
│   │   ├── NegotiationHub.py            # Multi-robot negotiation (NOT a TCP server)
│   │   ├── AutoRTIntegration.py         # AutoRTHandler singleton
│   │   └── WebUIServer.py               # ✅ Mission Control dashboard (8000, optional)
│   ├── autort/                          # ✅ Autonomous task generation
│   │   ├── TaskGenerator.py             # LLM-based task proposals
│   │   └── DataModels.py                # Pydantic models (ProposedTask, SceneDescription)
│   ├── agents/                          # LLM agents
│   │   ├── RobotLLMAgent.py             # Per-robot LLM agents
│   │   └── FeedbackCollector.py         # Injects anti-pattern warnings into CommandParser
│   ├── knowledge_graph/                 # Optional spatial reasoning (disabled by default)
│   ├── ros2/                            # ROSMotionClient, ROSBridge
│   ├── vision/                          # Object detection, depth estimation
│   ├── orchestrators/                   # Unified backend orchestrator
│   │   ├── RunRobotController.py        # ✅ PRIMARY entry point
│   │   ├── CommandParser.py             # LLM/regex command parser
│   │   ├── SequenceExecutor.py          # Sequential operation executor
│   │   └── OutcomeTracker.py            # Self-improvement outcome recording
│   ├── operations/                      # 29 registered operations (Levels 1-5)
│   │   ├── Base.py                      # Core operation classes
│   │   ├── Registry.py                  # Operation registry (29 ops)
│   │   ├── MoveOperations.py            # Navigation primitives
│   │   ├── GripperOperations.py         # Gripper control
│   │   ├── DetectionOperations.py       # Object detection + point cloud
│   │   ├── VisionOperations.py          # Scene analysis
│   │   ├── GraspOperations.py           # Grasp planning
│   │   ├── IntermediateOperations.py    # Complex single-robot tasks
│   │   ├── CoordinationOperations.py    # Multi-robot primitives
│   │   ├── CollaborativeOperations.py   # Collaborative tasks
│   │   └── WorldState.py                # Shared world state tracking
│   ├── rag/                             # Integrated RAG system
│   │   ├── Embeddings.py                # LM Studio embeddings
│   │   ├── VectorStore.py               # Numpy vector storage
│   │   └── QueryEngine.py               # Semantic search
│   ├── config/                          # Configuration modules
│   │   ├── AutoRT.py                    # ✅ AutoRT settings (LLM, safety, multi-robot)
│   │   └── Memory.py                    # LLM memory system config (MEMORY_ENABLED flag)
│   ├── tests/                           # Comprehensive test suite (80+ files)
│   ├── ACRLDashboard/                   # Web UI source (served by WebUIServer)
│   ├── LLMConfig.py                     # Backward-compatible config aggregator
│   └── acrl/                            # Python virtual environment
└── README.md
```

## Configuration

### Robot Configuration
Edit robot parameters via ScriptableObject assets:
```
ACRLUnity/Assets/Configuration/RobotConfig_*.asset
```

Key parameters:
- Joint stiffness, damping, force limits
- IK convergence threshold and max joint step
- Performance limits (max reach, velocity, acceleration)

### Simulation Configuration
Configure simulation via:
```
ACRLUnity/Assets/Configuration/SimulationConfig.asset
```

Options:
- Time scale, auto-start, reset on error
- Coordination mode (Independent/Collaborative/Master-Slave/etc.)
- Performance settings (target FPS, vSync)

### AutoRT Configuration
Configure autonomous task generation:
```
ACRLUnity/Assets/Configuration/DefaultAutoRTConfig.asset  (Unity)
ACRLPython/config/AutoRT.py                               (Python)
```

Unity Options:
- Max task candidates (1-5)
- Task selection strategy (Balanced/Simple/Complex/Random)
- Continuous loop settings (enable, delay)
- Robot assignment and collaborative tasks
- UI settings (max display tasks, refresh rate)

Python Options:
- LLM settings (LM Studio URL, models for generation/safety)
- Loop settings (max tasks, delay, human-in-the-loop default)
- Safety constraints (workspace bounds, velocity limits, separation)
- Multi-robot configuration (default robots, collaborative tasks)

## Development Branches

- `main` - Stable release branch
- `feature_self_improvement` - **CURRENT**: Dynamic operations, outcome tracking, and Sandbox execution
- `feature_autort` - AutoRT autonomous task generation system
- `feature_streaming` - YOLO streaming, unified backend, Protocol V2
- `feature_robot_cooperation` - Multi-robot coordination strategies
- `feature_rag` - RAG system integration (merged into feature_streaming)
- `feature_detect_object` - Object detection and stereo vision systems
- `navigate_to_object` - Navigation to detected objects
- `feature_gripper` - Gripper control implementation

## Quick Start

**5-Minute Start**:

1. Clone repository:

   ```bash
   git clone --recursive https://github.com/JanMStraub/Auto-Cooperative-Robot-Learning.git
   cd Auto-Cooperative-Robot-Learning
   ```

2. Setup Python backend:

   ```bash
   cd ACRLPython
   python -m venv acrl
   source acrl/bin/activate  # On Windows: acrl\Scripts\activate
   pip install -r requirements.txt
   python -m orchestrators.RunRobotController
   ```

3. Run Unity simulation:
   - Open `ACRLUnity/` in Unity Hub (version 6000.3.0f1 required)
   - Open scene: `Assets/Scenes/1xAR4Scene.unity`
   - Press Play

**For Natural Language Control**:

1. Ensure Python backend is running (see step 2 above)
2. In Unity, send commands via SequenceClient:

   ```csharp
   // Example: Detect, move, and grasp
   SequenceClient.Instance.SendCommand(
       "Detect the blue cube, move to it, close the gripper"
   );

   // Example: Multi-robot coordination
   SequenceClient.Instance.SendCommand(
       "Robot1: detect red cube and signal ready; " +
       "Robot2: wait for ready then move to blue cube"
   );
   ```

**For Autonomous Task Generation (AutoRT)**:

1. Ensure Python backend is running (see step 2 above)
2. In Unity scene, add AutoRTManager GameObject:
   - Create empty GameObject named "AutoRTManager"
   - Add `AutoRTManager` component
   - Assign `AutoRTConfig` asset from Configuration folder
3. Use custom inspector UI:
   - Click "Generate Tasks" button
   - Review proposed tasks in inspector
   - Click "Execute" to approve or "Reject" to discard
   - Optional: Enable "Start Loop" for continuous autonomous operation

**Available Operations** (29 total, organized by complexity):

**Level 1-2 Basic Operations** (19):

- Navigation: `move_to_coordinate`, `move_from_a_to_b`, `adjust_end_effector_orientation`, `return_to_start`
- Gripper: `control_gripper`, `release_object`
- Perception: `detect_objects`, `detect_object_stereo`, `analyze_scene`, `estimate_distance_to_object`, `estimate_distance_between_objects`, `generate_point_cloud`
- Field Detection: `detect_field`, `get_field_center`, `detect_all_fields`
- Status: `check_robot_status`
- Sync: `signal`, `wait_for_signal`, `wait`

**Level 3 Intermediate** (6):

- `grasp_object`, `align_object`, `move_relative_to_object`, `move_between_objects`, `move_to_region`, `follow_path`

**Level 4 Multi-Robot** (3):

- `detect_other_robot`, `mirror_movement`, `grasp_object_for_handoff`

**Level 5 Collaborative** (1):

- `stabilize_object`

## License

This project is licensed under the MIT License.

## Acknowledgments

- [AR4 Robot](https://github.com/zebleck/AR4) - Robot model and gripper controller inspiration
- [MathNet.Numerics](https://numerics.mathdotnet.com/) - Linear algebra for IK computation
- Unity Technologies - ArticulationBody physics system

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{straub2025acrl,
  author = {Jan M. Straub},
  title = {Auto-Cooperative Robot Learning},
  school = {Heidelberg University},
  year = {2025}
}
```

## Contact

For questions or collaboration:

- GitHub: [@JanMStraub](https://github.com/JanMStraub)
- Repository: [Auto-Cooperative-Robot-Learning](https://github.com/JanMStraub/Auto-Cooperative-Robot-Learning)
