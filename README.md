# Auto-Cooperative Robot Learning

A Unity-based reinforcement learning environment for training dual AR4 robotic arms to collaboratively solve tasks through multi-agent coordination. This project is part of a master's thesis exploring autonomous cooperative behavior in robotic systems.

## Description

The goal of this project is to have two AR4 robot arms positioned side by side that learn to collaboratively solve tasks which would be impossible for a single robot to accomplish. The system uses Unity ML-Agents for reinforcement learning, implementing inverse kinematics control, multi-robot coordination patterns, vision-based object detection, and comprehensive data logging for LLM training.

**Key Features**:
- Unity 6000.2.5f1 simulation environment with physics-based ArticulationBody robots
- Damped least-squares inverse kinematics (6-DOF control)
- PPO-based reinforcement learning with LSTM memory
- Multiple coordination modes: Independent, Collaborative, Master-Slave, Distributed, Sequential
- LLM vision integration (Ollama) for scene understanding
- Object detection system (color-based HSV segmentation)
- Stereo vision depth estimation for 3D object localization
- **Operations System**: Structured robot command framework with parameter validation and rich metadata
- **RAG System**: Semantic search over robot operations using LM Studio embeddings for LLM-driven control
- JSONL logging system for LLM training data generation
- Python-Unity TCP communication for real-time vision processing

## Getting Started

### Prerequisites

- **Unity Hub** with Unity Editor **6000.2.5f1** (exact version required)
- **Python 3.8+** for ML-Agents training
- **Git** with submodule support

### Dependencies

**Unity Packages** (managed via Package Manager):
- Unity ML-Agents (via git submodule)
- NuGetForUnity (for MathNet.Numerics)
- Unity Input System (1.14.2)
- Universal Render Pipeline (17.2.0)
- Unity Test Framework (1.5.1)

**Python Dependencies**:
- mlagents (Unity ML-Agents Toolkit) - installed in ml-agents submodule
- torch (PyTorch for neural networks)
- numpy, matplotlib (data processing)
- opencv-python (computer vision, object detection)
- ollama (LLM vision integration)

### Installing

1. **Clone the repository with submodules**:
   ```bash
   git clone --recursive https://github.com/JanMStraub/Auto-Cooperative-Robot-Learning.git
   cd Auto-Cooperative-Robot-Learning
   ```

2. **Setup ML-Agents Python environment**:
   ```bash
   cd ml-agents
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ./ml-agents-envs
   pip install -e ./ml-agents
   cd ..
   ```

3. **Open Unity project**:
   - Open Unity Hub
   - Add project from `ACRLUnity/` folder
   - Ensure Unity version **6000.2.5f1** is installed
   - Open the project (dependencies will auto-install)

4. **Install NuGet packages** (if not auto-installed):
   - In Unity: NuGet > Manage NuGet Packages
   - Install `MathNet.Numerics` (required for IK computation)

### Executing Program

#### Running Simulations in Unity

1. **Single Robot Testing**:
   ```
   Open: ACRLUnity/Assets/Scenes/1xAR4Scene.unity
   ```

2. **Multi-Robot Training**:
   ```
   Open: ACRLUnity/Assets/Scenes/16xAR4Scene.unity
   ```

3. **Using SimulationManager**:
   - Select SimulationManager GameObject in hierarchy
   - Use Inspector controls: Start, Pause, Resume, Reset
   - Configure coordination mode and settings via SimulationConfig asset

#### Training with ML-Agents

1. **Configure training parameters**:
   ```
   Edit: ACRLUnity/Assets/Configuration/RobotNavigation.yaml
   ```

2. **Run training** (from ml-agents directory):
   ```bash
   cd ml-agents
   source venv/bin/activate
   mlagents-learn ../ACRLUnity/Assets/Configuration/RobotNavigation.yaml --run-id=ar4_training
   ```

3. **Monitor training** (in separate terminal):
   ```bash
   tensorboard --logdir results/
   ```

4. **Trained models** are saved to:
   ```
   ml-agents/results/ar4_training/*.onnx
   ```

5. **Deploy model to Unity**:
   - Copy `.onnx` file to `ACRLUnity/Assets/Data/`
   - Assign to RobotAgent component's "Model" field

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

**Three Singleton Managers**:
- **SimulationManager**: Top-level orchestrator controlling simulation state and coordination modes
- **RobotManager**: Robot lifecycle management, configuration loading, target assignment
- **MainLogger**: Unified logging system for LLM training data with action tracking and trajectories

**Robot Control Layers**:
1. **RobotController**: Inverse kinematics computation using damped least-squares method
2. **RobotAgent**: ML-Agents integration with PPO training and episode management
3. **GripperController**: End-effector control with open/close commands

**Vision & Perception Systems**:
- **LLM Vision** (Ollama): Scene understanding and natural language descriptions
- **Object Detection**: HSV color-based cube detection with bounding boxes
- **Stereo Depth**: 3D localization using stereo disparity estimation
- **TCP Communication**: Real-time image streaming between Unity and Python (ports 5005-5009)

**LLM-Driven Control Systems**:
- **Operations System**: Structured framework for defining robot operations with parameters, preconditions, and failure modes
- **RAG System**: Semantic search using LM Studio embeddings to find relevant operations from natural language queries
- **Operation Registry**: Central catalog of all available robot commands with validation and execution
- **Parameter Validation**: Automatic validation with detailed error messages and recovery suggestions

**Data Logging**:
- JSONL logging per robot or per session
- LLM-ready export format with action types, trajectories, and metrics
- Thread-safe concurrent writes for multi-robot scenarios
- Export tools for LLM training and statistics generation

### Key Directories

```
Auto-Cooperative-Robot-Learning/
├── ACRLUnity/                    # Unity project root
│   ├── Assets/
│   │   ├── Configuration/        # Robot configs and ML training params
│   │   ├── Data/                 # Trained ML models (.onnx)
│   │   ├── Scenes/               # 1xAR4Scene, 16xAR4Scene
│   │   ├── Scripts/              # C# source code
│   │   │   ├── ConfigScripts/    # ScriptableObject configs
│   │   │   ├── Logging/          # Data logging system
│   │   │   └── *.cs              # Core controllers and managers
│   │   └── Prefabs/              # Robot and environment prefabs
│   ├── Packages/                 # Unity package dependencies
│   └── ProjectSettings/          # Unity project settings
├── ml-agents/                    # ML-Agents submodule (Python)
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

### ML Training Configuration
PPO hyperparameters in:
```
ACRLUnity/Assets/Configuration/RobotNavigation.yaml
```

Default settings:
- Batch size: 256
- Learning rate: 3e-4 (linear decay)
- Hidden units: 256 (3 layers)
- LSTM memory: 256 sequence length
- Max steps: 1M

## Development Branches

- `main` - Stable release branch
- `feature_detect_object` - Current development (object detection and stereo vision)
- `navigate_to_object` - Navigation to detected objects
- `feature_gripper` - Gripper control implementation
- `feature_ml` - ML-Agents integration features
- `work_package_2` - Research milestone tracking

## Quick Start

**5-Minute Start**:
1. Clone repository: `git clone --recursive https://github.com/JanMStraub/Auto-Cooperative-Robot-Learning.git`
2. Open `ACRLUnity/` in Unity Hub (version 6000.2.5f1 required)
3. Open scene: `Assets/Scenes/1xAR4Scene.unity`
4. Press Play to run simulation

**For ML Training**:
1. Setup ML-Agents: `cd ml-agents && python -m venv venv && source venv/bin/activate && pip install -e ./ml-agents`
2. Run training: `mlagents-learn ../ACRLUnity/Assets/Configuration/RobotNavigation.yaml --run-id=test`
3. Monitor: `tensorboard --logdir results/`

**For Vision/Detection**:
1. Setup Python environment: `cd ACRLPython && source acrl/bin/activate`
2. Run object detector: `python -m LLMCommunication.orchestrators.RunDetector`
3. In Unity: Use CameraController to send images and receive detection results

**For LLM-Driven Control** (Operations + RAG):
1. Setup Python environment: `cd ACRLPython && source acrl/bin/activate`
2. Start LM Studio with embedding model (e.g., nomic-embed-text)
3. Test operations: `python -m LLMCommunication.operations.example_usage`
4. Use RAG for semantic search: See `ACRLPython/LLMCommunication/rag/README.md`
5. Documentation: See `ACRLPython/LLMCommunication/operations/README.md` and `RAG_OPERATIONS_GUIDE.md`

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) - Reinforcement learning framework
- [AR4 Robot](https://github.com/zebleck/AR4) - Robot model and gripper controller inspiration
- [MathNet.Numerics](https://numerics.mathdotnet.com/) - Linear algebra for IK computation
- Unity Technologies - ArticulationBody physics system

## Citation

If you use this work in your research, please cite:

```
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
