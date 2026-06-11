# Auto-Cooperative Robot Learning

A Unity-based simulation environment for dual AR4 robotic arms that collaboratively solve tasks through LLM-driven multi-agent coordination. This project is part of a master's thesis exploring autonomous cooperative behavior in robotic systems.

## Description

The goal of this project is to have two AR4 robot arms positioned facing each other that collaboratively solve tasks. The system uses inverse kinematics control, LLM-driven task planning, multi-robot coordination patterns, and vision-based object detection.

**Key Features**:

- Unity 6000.3.11f1 simulation environment with physics-based ArticulationBody robots
- Damped least-squares inverse kinematics (6-DOF control)
- Multi-robot coordination via signal/wait primitives and collaborative operations
- Unified Python Backend: Single entry point (RunRobotController) orchestrates all servers
- Operations System: 29 registered operations including atomic actions, perception, sync primitives, and bimanual cooperative ops
- AutoRT System: Autonomous task generation with LLM-based planning and human-in-the-loop approval
- Knowledge Graph: Dynamic relation tracking for tracking complex topological environment states
- ROS 2 & Docker Integration: Physical robot control capabilities via `ROSMotionClient` and containerized ROS deployments
- Advanced Python Grasp Planning: Approach-aware motion (Top/Front/Side) generation and scoring via Python backend
- LLM vision integration for scene understanding and natural language commands
- Object detection with YOLO streaming support
- Stereo Vision & VGN: 3D object localization, stereo depth map reconstruction, and VGN-based local grasp network
- Camera/ & Hardware/ Abstraction: Sim2real switching via `--env sim|real` flag; no code changes required
- Web UI (Mission Control): Optional dashboard served via `--web PORT`; REST/WebSocket endpoints
- Protocol: Request ID correlation for reliable multi-robot communication
- RAG System: Integrated semantic search for operation matching in natural language commands
- Python-Unity TCP communication with persistent connections and health checks

## Getting Started

### Prerequisites

- **Unity Hub** with Unity Editor **6000.3.11f1** (exact version required)
- **Python 3.8+** with virtual environment support
- **Git** with submodule support
- **LM Studio** (or other tool like it, for RAG embeddings and LLM-based task generation)

### Dependencies

**Unity Packages** (managed via Package Manager):

- NuGetForUnity (for MathNet.Numerics)
- Unity Input System (1.14.2)
- Universal Render Pipeline (17.2.0)
- Unity Test Framework (1.5.1)

**Python Dependencies**:

- numpy, matplotlib (data processing)
- opencv-python (computer vision, object detection)
- openai (LLM integration)

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
   - Ensure Unity version **6000.3.11f1** is installed
   - Open the project (dependencies will auto-install)
   - Alternativly use standalone version: `ACRLUnity_build.app` (MacOS)

4. **Install NuGet packages** (if not auto-installed):
   - In Unity: NuGet > Manage NuGet Packages
   - Install `MathNet.Numerics` (required for IK computation)

### Executing Program

#### Quick Start with Python Backend

1. **Start the unified Python backend** (single command):

   ```bash
   cd ACRLPython
   source acrl/bin/activate  # On Windows: acrl\Scripts\activate
   ./start_servers.sh          # convenience script (or: python -m orchestrators.RunRobotController)
   ```

   This starts all servers: ImageServer (5006), CommandServer (5007), SequenceServer (5008), WorldStateServer (5009), AutoRTServer (5010), WebUIServer (8000).

2. **Run Unity simulation**:
   - If you use the standalone version, skip this as it starts automatically
   - Open `ACRLUnity/Assets/Scenes/1xAR4Scene.unity` for single robot testing
   - Press Play in Unity Editor
   - Use natural language commands via SequenceClient:

     ```csharp
     SequenceClient.Instance.SendCommand("Detect the blue cube, move to it, close the gripper");
     ```

     **OR** use the WebUI to send commands

#### Testing and Development

**Run Unity Tests**:

- Window > General > Test Runner
- Select PlayMode or EditMode tests
- Click "Run All" or run individual tests

**Build Standalone**:

- File > Build Settings
- Select platform (PC, Mac & Linux Standalone recommended)
- Click "Build" or "Build and Run"

## Architecture Overview

### Core Systems

**Core Singleton Managers**:

- **SimulationManager**: Top-level orchestrator controlling simulation state
- **RobotManager**: Robot lifecycle management, configuration loading, target assignment
- **AutoRTManager**: Autonomous task generation client with human-in-the-loop approval UI (port 5010)

**Robot Control Layers**:

- **RobotController**: Inverse kinematics computation using damped least-squares method
- **GripperController**: End-effector control with open/close commands

**Vision & Perception Systems**:

- **LLM Vision**: Scene understanding and natural language descriptions
- **Object Detection**: Color-based HSV segmentation + YOLO streaming support
- **Stereo Depth**: 3D localization using stereo disparity estimation
- **UnifiedImageStorage**: Thread-safe singleton for centralized image access

**Python Backend Architecture**:

- **Unified Entry Point**: `RunRobotController` orchestrates all servers
- **6 Active Servers**:
  - **ImageServer** (5006): Stereo image receiver
  - **CommandServer** (5007): Bidirectional commands and completions
  - **SequenceServer** (5008): Multi-command sequence orchestration + AutoRT integration
  - **WorldStateServer** (5009): Robot/object state streaming
  - **AutoRTServer** (5010): Autonomous task generation
  - **WebUIServer** (8000, optional): Mission Control dashboard (`--web PORT`)
- **Camera/ & Hardware/ Abstraction**: Sim2real switching without code changes (`--env sim|real`)
- **AutoRT Module**: LLM-based autonomous task generation with Pydantic validation
- **Protocol**: Request ID correlation prevents race conditions in multi-robot scenarios
- **Persistent Connections**: TCP keepalive with health checks and automatic recovery

**LLM-Driven Control Systems**:

- **Operations System**: 29 registered operations organized by complexity
  - **Atomic** (8): `control_gripper`, `release_object`, `check_robot_status`, `signal`, `wait_for_signal`, `wait`, `reset_simulation`, `yield_workspace`
  - **Basic** (6): `move_to_coordinate`, `adjust_end_effector_orientation`, `return_to_start_position`, `generate_point_cloud`, `detect_field`, `detect_all_fields`
  - **Intermediate** (9): `pick_object_at_coordinate`, `place_object`, `place_between_objects`, `detect_object_stereo`, `analyze_scene`, `move_relative_to_object`, `detect_other_robot`, `check_partner_status`, `place_for_partner`
  - **Complex** (6): `grasp_object`, `mirror_movement_of_other_robot`, `receive_handoff`, `stabilize_object`, `synchronized_grasp`, `joint_transport`
  - Variable passing: `detect -> $target`, then `move to $target`
- **AutoRT System**: Autonomous task generation with LLM planning and human approval workflow
- **Integrated RAG System**: Semantic search using LM Studio embeddings for natural language command parsing
- **CommandParser**: LLM/regex hybrid parser with operation registry matching
- **SequenceExecutor**: Sequential operation executor with state tracking

**ROS 2 / MoveIt Integration** :

- MoveIt used for **planning only** — Unity executes all trajectories via `ROSTrajectorySubscriber`
- Three control modes via `ROSControlModeManager`: Unity (default IK), ROS (MoveIt plans), Hybrid (ROS priority with Unity fallback)
- `ROSMotionClient` (Python) handles coordinate transforms (Unity Y-up left-handed → ROS Z-up right-handed) and sends plans over TCP to `ROSBridge` running in Docker
- Containerized stack in `rosUnityIntegration/` — ROS 2 + MoveIt + `ros_tcp_endpoint` (port 10000)

**Multi-Robot Negotiation** (disabled by default, enable via `NEGOTIATION_ENABLED=true`):

- `NegotiationHub` (singleton in `servers/NegotiationHub.py`) — NOT a TCP server; called directly by `SequenceExecutor` before command parsing when collaboration keywords or 2+ robot IDs detected
- Per-robot `RobotLLMAgent` instances run parallel Analysis → round-robin Proposal → parallel Evaluation, up to `MAX_NEGOTIATION_ROUNDS=3`
- Falls back to normal `CommandParser` flow on timeout or failure

### Key Directories

```
Auto-Cooperative-Robot-Learning/
├── ACRLDashboard/                       # Web UI source (served by WebUIServer on port 8000)
├── rosUnityIntegration/                 # Docker-based ROS 2 + MoveIt + ros_tcp_endpoint
├── ACRLUnity/                           # Unity project root
│   ├── Assets/
│   │   ├── Configuration/               # Robot, simulation, and grasp config assets
│   │   ├── Data/                        # Runtime data assets
│   │   ├── Scenes/                      # 1xAR4Scene
│   │   ├── Scripts/                     # C# source code
│   │   │   ├── ConfigScripts/           # ScriptableObject configs
│   │   │   ├── PythonCommunication/     # TCP clients and Protocol
│   │   │   ├── RobotScripts/            # Robot control and IK
│   │   │   ├── SimulationScripts/       # Coordination strategies
│   │   │   └── *.cs                     # Core controllers and managers
│   │   └── Prefabs/                     # Robot and environment prefabs
│   ├── Packages/                        # Unity package dependencies
│   └── ProjectSettings/                 # Unity project settings
├── ACRLPython/                          # Python backend
│   ├── core/                            # TCPServerBase, UnityProtocol, Imports, LoggingSetup
│   ├── camera/                          # Sim2real camera abstraction (--env flag)
│   │   ├── Provider.py                  # Abstract CameraProvider interface
│   │   ├── UnityProvider.py             # Adapter for Unity ImageStorage
│   │   └── LocalProvider.py             # Adapter for real cameras (USB/RealSense)
│   ├── hardware/                        # Sim2real robot hardware abstraction
│   │   ├── Interface.py                 # Abstract RobotHardwareInterface
│   │   ├── UnityInterface.py            # Adapter for Unity robot control
│   │   └── ROSInterface.py              # Adapter for ROS/MoveIt control
│   ├── servers/                         # 6 active servers (+ 1 optional)
│   │   ├── ImageServer.py               # Stereo image receiver (5006)
│   │   ├── CommandServer.py             # Bidirectional commands (5007)
│   │   ├── SequenceServer.py            # Multi-command sequences (5008)
│   │   ├── WorldStateServer.py          # Robot/object state streaming (5009)
│   │   ├── AutoRTServer.py              # Autonomous task generation (5010)
│   │   ├── NegotiationHub.py            # Multi-robot negotiation (NOT a TCP server)
│   │   ├── AutoRTIntegration.py         # AutoRTHandler singleton
│   │   └── WebUIServer.py               # Mission Control dashboard (8000, optional)
│   ├── autort/                          # Autonomous task generation
│   │   ├── AutoRTLoop.py                # AutoRTOrchestrator main loop
│   │   ├── TaskGenerator.py             # LLM-based task proposals
│   │   ├── TaskSelector.py              # Task selection strategies (balanced/explore/exploit/random)
│   │   ├── RobotConstitution.py         # Two-layer safety (semantic LLM + kinematic code)
│   │   └── DataModels.py                # Pydantic models (ProposedTask, SceneDescription, TaskVerdict)
│   ├── agents/                          # LLM agents
│   │   └── RobotLLMAgent.py             # Per-robot LLM agents
│   ├── knowledge_graph/                 # Optional spatial reasoning (disabled by default; KNOWLEDGE_GRAPH_ENABLED=false)
│   │   ├── Core.py                      # Thread-safe KnowledgeGraph (NetworkX MultiDiGraph)
│   │   ├── Schema.py                    # RobotNode, ObjectNode, RegionNode dataclasses
│   │   ├── GraphBuilder.py              # WorldState → graph sync
│   │   ├── QueryEngine.py               # Reachability, proximity, handoff planning
│   │   └── _singleton.py                # Module-level singleton accessor
│   ├── ros2/                            # ROSMotionClient, ROSBridge
│   ├── vision/                          # Object detection, depth estimation
│   ├── orchestrators/                   # Unified backend orchestrator
│   │   ├── RunRobotController.py        # PRIMARY entry point
│   │   ├── RunAutoRT.py                 # Standalone AutoRT entry point
│   │   ├── CommandParser.py             # LLM/regex command parser
│   │   └── SequenceExecutor.py          # Sequential operation executor
│   ├── operations/                      # 29 registered operations (Atomic/Basic/Intermediate/Complex)
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
│   │   ├── AutoRT.py                    # AutoRT settings (LLM, safety, multi-robot)
│   │   ├── Servers.py                   # Port assignments and server settings
│   │   ├── Robot.py                     # Robot base positions and transforms
│   │   ├── ROS.py                       # ROS integration settings
│   │   └── KnowledgeGraph.py            # Knowledge graph settings
│   ├── tests/                           # Comprehensive test suite (80+ files)
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

## Benchmarks

**Run Benchmarks**:

```bash
cd ACRLPython
source acrl/bin/activate
# Ensure servers are running first (./start_servers.sh)

# Run all benchmarks with the live Unity simulation
python -m benchmarks.run --all --live

# Run single benchmark (1-16)
python -m benchmarks.run --benchmark 3

# Dry-run (no hardware required)
python -m benchmarks.run --all --dry-run
```

Benchmark cases:

- **B1**: Navigate to object (detect + grasp)
- **B2**: Sequential multi-target navigation
- **B3**: Navigate and lift (approach-aware grasp)
- **B4**: Pick and place
- **B5**: Pose-aware grasp (oriented top-down)
- **B6**: Robot handoff (one robot hands object to the other)
- **B7**: Dual-robot reorient with sync barriers
- **B8**: Heterogeneous chain (cycles B1/B3/B4 sub-tasks)
- **B9**: Impossible task (parse-only; validates graceful failure)
- **B10**: Parallel independent (dual-robot concurrent tasks)
- **B11**: RAG ablation
- **B12**: Reflexion ablation
- **B13**: Negotiation ablation
- **B14**: Knowledge Graph ablation (parse-only)
- **B15**: VGN ablation
- **B16**: ROS ablation

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
   - Optional: Enable "Start Loop" for continuous autonomous operatio

**Available Operations** (29 total, organized by complexity):

**Atomic** (8):

- `control_gripper`, `release_object`, `check_robot_status`, `signal`, `wait_for_signal`, `wait`, `reset_simulation`, `yield_workspace`

**Basic** (6):

- `move_to_coordinate`, `adjust_end_effector_orientation`, `return_to_start_position`, `generate_point_cloud`, `detect_field`, `detect_all_fields`

**Intermediate** (9):

- `pick_object_at_coordinate`, `place_object`, `place_between_objects`, `detect_object_stereo`, `analyze_scene`, `move_relative_to_object`, `detect_other_robot`, `check_partner_status`, `place_for_partner`

**Complex** (6):

- `grasp_object`, `mirror_movement_of_other_robot`, `receive_handoff`, `stabilize_object`, `synchronized_grasp`, `joint_transport`

## License

This project is licensed under the MIT License.

## Acknowledgments

- [AR4 Robot](https://github.com/zebleck/AR4) - Robot model and gripper controller inspiration
- [MathNet.Numerics](https://numerics.mathdotnet.com/) - Linear algebra for IK computation
- Unity Technologies - ArticulationBody physics system

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{straub2026acrl,
  author = {Jan M. Straub},
  title = {Auto-Cooperative Robot Learning},
  school = {Heidelberg University},
  year = {2026}
}
```

## Contact

For questions or collaboration:

- GitHub: [@JanMStraub](https://github.com/JanMStraub)
- Repository: [Auto-Cooperative-Robot-Learning](https://github.com/JanMStraub/Auto-Cooperative-Robot-Learning)
