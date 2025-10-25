# Auto-Cooperative Robot Learning - User Guide

**Version**: 1.0
**Last Updated**: October 2025
**Project**: Master's Thesis - Heidelberg University

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Unity Simulation](#3-unity-simulation)
4. [ML-Agents Training](#4-ml-agents-training)
5. [LLM Communication System](#5-llm-communication-system)
6. [Object Detection](#6-object-detection)
7. [Stereo Vision & Depth Estimation](#7-stereo-vision--depth-estimation)
8. [Data Logging](#8-data-logging)
9. [Development Guide](#9-development-guide)
10. [Troubleshooting](#10-troubleshooting)
11. [Reference](#11-reference)

---

## 1. Introduction

### What is ACRL?

Auto-Cooperative Robot Learning (ACRL) is a Unity-based simulation environment for training dual AR4 robotic arms to perform collaborative tasks using reinforcement learning. The system integrates:

- **Physics-based simulation** (Unity 6000.2.5f1 with ArticulationBody)
- **Reinforcement learning** (Unity ML-Agents with PPO)
- **Computer vision** (LLM integration, object detection, stereo depth)
- **Data logging** (JSONL format for LLM training)

### Key Capabilities

✅ **Inverse Kinematics**: 6-DOF damped least-squares IK for precise control
✅ **Multi-Robot Coordination**: 5 coordination modes (Independent, Collaborative, Master-Slave, etc.)
✅ **Vision Integration**: Real-time LLM vision analysis via Ollama
✅ **Object Detection**: Color-based HSV segmentation for cube detection
✅ **3D Localization**: Stereo vision depth estimation
✅ **ML Training**: PPO with LSTM for complex task learning
✅ **Data Generation**: Comprehensive logging for LLM fine-tuning

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Unity Simulation                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Simulation   │  │   Robot      │  │  Main        │     │
│  │  Manager     │  │  Manager     │  │  Logger      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         └──────────┬───────┴──────────────────┘             │
│                    │                                         │
│         ┌──────────▼────────┐                              │
│         │  Robot Controllers │                              │
│         │  (IK + ML-Agents)  │                              │
│         └──────────┬─────────┘                              │
│                    │                                         │
│         ┌──────────▼────────┐                              │
│         │   Camera System   │                              │
│         │  (Image Capture)  │                              │
│         └──────────┬─────────┘                              │
└────────────────────┼──────────────────────────────────────┘
                     │ TCP (ports 5005-5009)
┌────────────────────▼──────────────────────────────────────┐
│                   Python Backend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Streaming   │  │   Object     │  │   Stereo     │    │
│  │   Server     │  │  Detector    │  │  Detector    │    │
│  │  (port 5005) │  │ (port 5007)  │  │ (port 5009)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                  │                  │             │
│         └──────────┬───────┴──────────────────┘             │
│                    │                                         │
│         ┌──────────▼────────┐                              │
│         │   LLM / Vision    │                              │
│         │  (Ollama, OpenCV) │                              │
│         └──────────┬─────────┘                              │
│                    │                                         │
│         ┌──────────▼─────────┐                             │
│         │   Results Server   │                             │
│         │    (port 5006)     │                             │
│         └────────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Getting Started

### 2.1 Prerequisites

**Required Software**:
- **Unity Hub** (latest version)
- **Unity Editor** version **6000.2.5f1** (exact version required)
- **Python 3.8+**
- **Git** with submodule support
- **Ollama** (optional, for LLM vision features)

**Operating Systems**:
- macOS (primary development platform)
- Linux (supported)
- Windows (supported with minor path adjustments)

### 2.2 Installation

#### Step 1: Clone Repository

```bash
# Clone with submodules (includes ml-agents)
git clone --recursive https://github.com/JanMStraub/Auto-Cooperative-Robot-Learning.git
cd Auto-Cooperative-Robot-Learning

# If already cloned without --recursive:
git submodule update --init --recursive
```

#### Step 2: Unity Setup

1. **Install Unity 6000.2.5f1**:
   - Open Unity Hub
   - Go to "Installs" → "Install Editor"
   - Select version **6000.2.5f1**
   - Add modules: Linux Build Support (optional), documentation

2. **Open Project**:
   - In Unity Hub: "Add" → Select `ACRLUnity/` folder
   - Open project
   - Wait for package resolution (may take 5-10 minutes first time)

3. **Verify Installation**:
   - Open `Assets/Scenes/1xAR4Scene.unity`
   - Press Play
   - Robot should appear and respond to SimulationManager controls

#### Step 3: Python Environments

You'll need **two separate Python environments**:

**A) ML-Agents Environment** (for training):
```bash
cd ml-agents
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
pip install torch  # May take several minutes
cd ..
```

**B) LLM Communication Environment** (for vision/detection):
```bash
cd ACRLPython
python -m venv acrl
source acrl/bin/activate  # Windows: acrl\Scripts\activate
pip install numpy opencv-python ollama
cd ..
```

#### Step 4: Verify Installation

**Unity Test**:
```bash
# In Unity Editor: Window > General > Test Runner
# Run all tests to verify functionality
```

**Python Test**:
```bash
cd ACRLPython
source acrl/bin/activate
python -m pytest Tests/  # Run Python test suite
```

### 2.3 Quick Start (5 Minutes)

**Run Your First Simulation**:

1. Open Unity → `ACRLUnity/Assets/Scenes/1xAR4Scene.unity`
2. Select `SimulationManager` in hierarchy
3. Click "Start Simulation" in Inspector
4. Watch robot move to target position using IK

**Try Object Detection**:

1. Start Python server:
   ```bash
   cd ACRLPython
   source acrl/bin/activate
   python -m LLMCommunication.orchestrators.RunDetector
   ```

2. In Unity, select robot's `CameraController`
3. Click "Send to LLM" in Inspector
4. Check Unity Console for detection results

---

## 3. Unity Simulation

### 3.1 Scene Structure

Each simulation scene contains these core GameObjects:

```
Scene Hierarchy:
├── SimulationManager      # Top-level orchestrator
├── RobotManager           # Robot lifecycle management
├── MainLogger             # Data logging system
├── AR4_Robot_1           # First robot instance
│   ├── Base
│   ├── Joint1-6          # Articulation chain
│   ├── Gripper
│   └── Camera            # Attached camera
├── AR4_Robot_2           # Second robot (multi-robot scenes)
├── Target                # IK target marker
└── Environment           # Scene objects, table, obstacles
```

### 3.2 SimulationManager

**Purpose**: Controls overall simulation lifecycle and coordination.

**Inspector Fields**:
- **Simulation Config**: Reference to ScriptableObject configuration
- **Auto Start**: Start simulation on scene load
- **Coordination Mode**: How robots work together

**Runtime Controls** (Custom Inspector):
- **Start Simulation**: Initialize and begin
- **Pause**: Freeze simulation
- **Resume**: Continue from pause
- **Reset**: Return to initial state

**Coordination Modes**:
1. **Independent**: Each robot acts autonomously
2. **Collaborative**: Robots share goals and coordinate actions
3. **Master-Slave**: One robot leads, others follow
4. **Distributed**: Decentralized decision-making
5. **Sequential**: Robots take turns executing tasks

**States**:
- `Initializing` → `Running` → `Paused` → `Running` → `Resetting`
- Error states trigger automatic reset (if configured)

### 3.3 RobotManager

**Purpose**: Manages robot instances, configurations, and targets.

**Key Features**:
- Maintains `Dictionary<string, RobotInstance>` for all robots
- Loads RobotConfig profiles (joint settings, IK parameters)
- Assigns targets dynamically
- Fires `OnTargetChanged` event for coordination

**Robot Registration**:
```csharp
// Automatically registers robots with RobotController components
// Manual registration:
RobotManager.Instance.RegisterRobot("Robot1", robotGameObject, targetObject, config);
```

**Target Assignment**:
```csharp
// Set target for a specific robot
RobotManager.Instance.SetTarget("Robot1", targetPosition, targetRotation);
```

### 3.4 Robot Control

#### RobotController (IK Computation)

**Location**: `Assets/Scripts/RobotScripts/RobotController.cs`

**Algorithm**: Damped Least-Squares Pseudo-Inverse IK
- 6-DOF control (XYZ position + RPY orientation)
- Damping factor λ = 0.1
- Pre-allocated matrices for garbage-collection-free operation
- Convergence threshold: 0.1m (configurable)

**Usage**:
```csharp
var controller = GetComponent<RobotController>();
controller.SetTarget(targetPosition, targetRotation);

// Check if target reached
if (controller.IsTargetReached()) {
    Debug.Log("Robot reached target!");
}
```

**Configuration** (via RobotConfig ScriptableObject):
- `convergenceThreshold`: Distance threshold for "reached" (default: 0.1m)
- `maxJointStepRad`: Max joint angle change per update (default: 0.1 rad)
- `adjustmentSpeed`: IK solver speed multiplier

#### RobotAgent (ML-Agents)

**Location**: `Assets/Scripts/RobotScripts/RobotAgent.cs`

**Observations** (Vector size: 18):
- Base to gripper position (XYZ): 3 values
- Target to gripper position (XYZ): 3 values
- Joint angles (6 joints): 6 values
- Joint velocities (6 joints): 6 values

**Actions** (Continuous, size: 6):
- Target joint angles for each of 6 joints
- Normalized to [-1, 1], scaled to joint limits

**Rewards**:
- **Distance Progress**: +reward for moving closer to target
- **Proximity**: Log-scaled reward for being near target
- **Goal Reached**: +10.0 bonus on success
- **Time Penalty**: -0.001 per step (encourages efficiency)
- **Smoothness Penalty**: -reward for jerky movements

**Episode Management**:
- Max steps: 2000
- Reset on success (target reached) or timeout
- Random starting position within reachable workspace

#### GripperController

**Location**: `Assets/Scripts/RobotScripts/GripperController.cs`

**Commands**:
```csharp
var gripper = GetComponent<GripperController>();
gripper.Open();   // Open gripper
gripper.Close();  // Close gripper
```

**Custom Inspector**: Runtime testing buttons for open/close

### 3.5 Configuration System

**ScriptableObject-based configs** stored in `Assets/Configuration/`:

**RobotConfig** (per robot):
```
- Joint Settings (per joint):
  - Stiffness: 500-800
  - Damping: 100-250
  - Force Limit: 1000
  - Angle Limits: min/max degrees

- IK Settings:
  - Convergence Threshold: 0.1m
  - Max Joint Step: 0.1 rad
  - Adjustment Speed: 1.0

- Performance Limits:
  - Max Reach Distance: 0.8m
  - Max Velocity: 2.0 m/s
  - Max Acceleration: 5.0 m/s²
```

**SimulationConfig** (global):
```
- Time Scale: 1.0 (realtime)
- Auto Start: true/false
- Reset On Error: true/false
- Coordination Mode: enum
- Target Frame Rate: 60
- VSync: enabled/disabled
```

**Editing Configs**:
1. Select `.asset` file in Project window
2. Edit values in Inspector
3. Changes apply immediately to active instances

### 3.6 Camera System

**CameraController** (`Assets/Scripts/CameraScripts/CameraController.cs`):

**Features**:
- Capture camera view as PNG/JPG
- Send to Python servers (LLM, detection)
- Receive and display results
- Optional stereo mode for depth estimation

**Inspector Fields**:
- **Camera**: Camera component to capture
- **Camera ID**: Identifier for multi-camera setups
- **Send to LLM**: Capture and analyze with LLM
- **Use Stereo Depth**: Enable stereo vision mode
- **Right Camera**: Second camera for stereo pair

**Usage**:
```csharp
var camController = GetComponent<CameraController>();

// Capture and send for LLM analysis
camController.CaptureAndSend();

// Manual capture
byte[] imageData = camController.CaptureImage();
```

---

## 4. ML-Agents Training

### 4.1 Training Overview

**Training Pipeline**:
1. Unity builds standalone player with sensors/actuators
2. Python mlagents-learn runs PPO training algorithm
3. Neural network learns policy through trial and error
4. Trained model (.onnx) deployed back to Unity

### 4.2 Training Configuration

**File**: `ACRLUnity/Assets/Configuration/RobotNavigation.yaml`

**Key Parameters**:
```yaml
behaviors:
  RobotBehavior:
    trainer_type: ppo

    hyperparameters:
      batch_size: 256                # Samples per gradient update
      buffer_size: 2048              # Experience replay buffer
      learning_rate: 0.0003          # Initial learning rate
      learning_rate_schedule: linear # Decay to zero

    network_settings:
      normalize: false
      hidden_units: 256             # Neurons per layer
      num_layers: 3                 # Hidden layers
      vis_encode_type: simple
      memory:
        sequence_length: 256        # LSTM memory length
        memory_size: 128            # LSTM hidden state size

    reward_signals:
      extrinsic:
        gamma: 0.99                 # Discount factor
        strength: 1.0

    max_steps: 1000000             # Total training steps
    time_horizon: 1000             # Steps before experience batch
    summary_freq: 10000            # TensorBoard update frequency
    checkpoint_interval: 10000     # Model save frequency
```

**Tuning Guide**:
- **Increase `batch_size`** if training is unstable (more stable gradients)
- **Decrease `learning_rate`** if policy oscillates (smoother learning)
- **Increase `hidden_units`** for more complex tasks (more capacity)
- **Increase `max_steps`** if not converging (longer training)

### 4.3 Running Training

**Step 1: Activate ML Environment**:
```bash
cd ml-agents
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**Step 2: Start Training**:
```bash
# Basic training
mlagents-learn ../ACRLUnity/Assets/Configuration/RobotNavigation.yaml --run-id=my_training

# Resume from checkpoint
mlagents-learn ../ACRLUnity/Assets/Configuration/RobotNavigation.yaml --run-id=my_training --resume

# Inference mode (test trained model)
mlagents-learn ../ACRLUnity/Assets/Configuration/RobotNavigation.yaml --run-id=my_training --inference
```

**Step 3: Press Play in Unity** when prompted

**Step 4: Monitor Training**:
```bash
# In separate terminal
cd ml-agents
tensorboard --logdir results/

# Open browser to http://localhost:6006
```

### 4.4 Deploying Trained Models

**Step 1: Locate Model**:
```bash
# Trained models saved to:
ml-agents/results/my_training/RobotBehavior/RobotBehavior-XXXXXX.onnx
```

**Step 2: Copy to Unity**:
```bash
cp results/my_training/RobotBehavior/RobotBehavior-final.onnx ../ACRLUnity/Assets/Data/
```

**Step 3: Assign in Unity**:
1. Select robot with `RobotAgent` component
2. Find "Model" field in Inspector
3. Drag `.onnx` file from `Assets/Data/`
4. Set "Behavior Type" to "Inference Only"
5. Press Play to test

### 4.5 Training Tips

**Best Practices**:
- Start with simple tasks (single target reaching)
- Gradually increase complexity (moving targets, obstacles)
- Use curriculum learning (easy → hard scenarios)
- Monitor reward curves in TensorBoard
- Save checkpoints frequently (`checkpoint_interval: 10000`)

**Common Issues**:
- **Reward not increasing**: Check reward function, ensure task is possible
- **Policy collapses**: Reduce learning rate, increase batch size
- **NaN values**: Reduce learning rate, check observation normalization
- **Slow training**: Use standalone build (faster than editor), reduce `time_horizon`

---

## 5. LLM Communication System

### 5.1 System Overview

The LLM Communication system enables real-time vision analysis using Ollama vision models. Images captured in Unity are sent to Python, processed by an LLM, and results returned to Unity.

**Architecture**:
```
Unity Camera → ImageSender → StreamingServer (port 5005) → ImageStorage
                                    ↓
                           RunAnalyzer detects image
                                    ↓
                           Ollama LLM processes
                                    ↓
                    ResultsBroadcaster → ResultsServer (port 5006)
                                    ↓
                          Unity LLMResultsReceiver
```

### 5.2 Python Side Setup

**Step 1: Install Ollama**:
```bash
# macOS
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

**Step 2: Download Vision Model**:
```bash
# Download LLaVA (7B parameters, ~4GB)
ollama pull llava

# Or Gemma3 (smaller, faster)
ollama pull gemma3

# Test model
ollama run llava "Describe this image" --image test.jpg
```

**Step 3: Start LLM Analyzer**:
```bash
cd ACRLPython
source acrl/bin/activate
python -m LLMCommunication.orchestrators.RunAnalyzer --model llava

# Output:
# Starting StreamingServer on 127.0.0.1:5005...
# Starting ResultsServer on 127.0.0.1:5006...
# Monitoring for new images...
```

### 5.3 Unity Side Setup

**Step 1: Add ImageSender** (if not present):
```
GameObject → Create Empty → "ImageSender"
Add Component → ImageSender.cs
Inspector:
  - Server Host: 127.0.0.1
  - Server Port: 5005
  - Auto Connect: ✓
```

**Step 2: Add LLMResultsReceiver** (if not present):
```
GameObject → Create Empty → "LLMResultsReceiver"
Add Component → LLMResultsReceiver.cs
Inspector:
  - Server Host: 127.0.0.1
  - Server Port: 5006
  - Auto Connect: ✓
```

**Step 3: Send Image for Analysis**:

Option A - Use CameraController:
```
Select robot's CameraController → Inspector → Click "Send to LLM"
```

Option B - Via Script:
```csharp
using LLMCommunication;

public class MyScript : MonoBehaviour {
    void Update() {
        if (Input.GetKeyDown(KeyCode.Space)) {
            Camera cam = GetComponent<Camera>();
            ImageSender.Instance.CaptureAndSendCamera(cam, "MyCamera", "Describe the scene");
        }
    }
}
```

### 5.4 Receiving LLM Results

**Subscribe to Event**:
```csharp
using LLMCommunication;

public class LLMHandler : MonoBehaviour {
    void Start() {
        LLMResultsReceiver.Instance.OnResultReceived += HandleLLMResult;
    }

    void HandleLLMResult(LLMResult result) {
        if (result.success) {
            Debug.Log($"LLM Response for {result.camera_id}:");
            Debug.Log(result.response);

            // Use result.metadata for model info, processing time, etc.
        } else {
            Debug.LogError($"LLM analysis failed: {result.error}");
        }
    }

    void OnDestroy() {
        if (LLMResultsReceiver.Instance != null) {
            LLMResultsReceiver.Instance.OnResultReceived -= HandleLLMResult;
        }
    }
}
```

**LLMResult Structure**:
```csharp
public class LLMResult {
    public bool success;           // True if analysis succeeded
    public string response;        // LLM's text response
    public string camera_id;       // Which camera sent the image
    public string timestamp;       // ISO 8601 timestamp
    public Dictionary<string, object> metadata;  // Model info, timing, etc.
    public string error;           // Error message if failed
}
```

### 5.5 LLM Prompting Tips

**Effective Prompts**:
```csharp
// ✅ GOOD: Specific, structured
"List all objects visible in the scene with their colors and approximate positions"

// ✅ GOOD: Task-oriented
"Describe the spatial relationship between the red cube and the blue cube"

// ❌ BAD: Too vague
"What do you see?"

// ❌ BAD: Assumes capabilities
"Calculate the exact 3D coordinates of all objects"  // LLM can't do precise measurements
```

**Use Cases**:
- Scene understanding for task planning
- Object identification when detection fails
- Natural language descriptions for logging
- Verification of robot actions ("Did the robot reach the target?")

---

## 6. Object Detection

### 6.1 Detection System Overview

The object detection system uses color-based HSV segmentation to detect red and blue cubes in camera images. It provides both pixel coordinates and Unity world coordinates.

**Features**:
- Real-time detection (~10-50ms per frame)
- Bounding box and center point localization
- Confidence scores for each detection
- Pixel coordinates → Unity world coordinates (via raycasting)
- Configurable color ranges and filters

### 6.2 Python Setup

**Start Detection Server**:
```bash
cd ACRLPython
source acrl/bin/activate

# Basic start
python -m LLMCommunication.orchestrators.RunDetector

# With options
python -m LLMCommunication.orchestrators.RunDetector \
    --camera AR4Left \      # Monitor specific camera only
    --interval 0.5 \        # Check every 0.5s (default: 1.0s)
    --debug                 # Save annotated debug images

# Output:
# Starting DetectionServer on 127.0.0.1:5007...
# Starting StreamingServer on 127.0.0.1:5005...
# Monitoring for new images every 1.0s...
```

### 6.3 Unity Setup

**Add DetectionResultsReceiver**:
```
GameObject → Create Empty → "DetectionResultsReceiver"
Add Component → DetectionResultsReceiver.cs
Inspector:
  - Server Host: 127.0.0.1
  - Server Port: 5007
  - Camera Mappings:
    [0]: Camera Id: "AR4Left"
         Camera: [Drag Main Camera here]
```

**Add Visualizer** (Optional):
```
GameObject → ACRL → Cube Detection Visualizer
```

### 6.4 Sending Images for Detection

**Option A - CameraController**:
```
Select CameraController → Inspector → "Capture and Send"
```

**Option B - Script**:
```csharp
using LLMCommunication;

public class DetectionTester : MonoBehaviour {
    public Camera targetCamera;

    void Update() {
        if (Input.GetKeyDown(KeyCode.D)) {
            // Capture and send for detection
            CameraController camController = targetCamera.GetComponent<CameraController>();
            if (camController != null) {
                camController.CaptureAndSend();
            }
        }
    }
}
```

### 6.5 Receiving Detection Results

**Subscribe to Events**:
```csharp
using LLMCommunication;

public class CubeTracker : MonoBehaviour {
    void Start() {
        DetectionResultsReceiver.Instance.OnDetectionWithWorldReceived += HandleDetection;
    }

    void HandleDetection(DetectionResultWithWorld result) {
        Debug.Log($"Detected {result.CubesWithWorldCoords.Length} cubes from {result.OriginalResult.camera_id}");

        foreach (var cube in result.CubesWithWorldCoords) {
            // Pixel coordinates
            Vector2Int pixelPos = cube.OriginalDetection.center_px;
            BoundingBoxPx bbox = cube.OriginalDetection.bbox_px;

            // World coordinates (if raycast succeeded)
            if (cube.HasWorldPosition) {
                Vector3 worldPos = cube.WorldPosition;
                GameObject hitObject = cube.HitObject;

                Debug.Log($"{cube.OriginalDetection.color} cube:");
                Debug.Log($"  Pixel: ({pixelPos.x}, {pixelPos.y})");
                Debug.Log($"  World: {worldPos}");
                Debug.Log($"  Confidence: {cube.OriginalDetection.confidence:F2}");

                // Navigate robot to cube
                RobotManager.Instance.SetTarget("Robot1", worldPos);
            }
        }
    }

    void OnDestroy() {
        if (DetectionResultsReceiver.Instance != null) {
            DetectionResultsReceiver.Instance.OnDetectionWithWorldReceived -= HandleDetection;
        }
    }
}
```

**Filter by Color**:
```csharp
void HandleDetection(DetectionResultWithWorld result) {
    // Get only red cubes with valid world positions
    var redCubes = System.Array.FindAll(
        result.CubesWithWorldCoords,
        c => c.OriginalDetection.color == "red" && c.HasWorldPosition
    );

    if (redCubes.Length > 0) {
        // Navigate to nearest red cube
        float minDist = float.MaxValue;
        Vector3 nearestPos = Vector3.zero;

        foreach (var cube in redCubes) {
            float dist = Vector3.Distance(transform.position, cube.WorldPosition);
            if (dist < minDist) {
                minDist = dist;
                nearestPos = cube.WorldPosition;
            }
        }

        RobotManager.Instance.SetTarget("Robot1", nearestPos);
    }
}
```

### 6.6 Configuration & Tuning

**Edit** `ACRLPython/LLMCommunication/config.py`:

```python
# Color Ranges (HSV)
RED_HSV_LOWER_1 = (0, 100, 100)      # Adjust H (hue) for red tones
RED_HSV_UPPER_1 = (10, 255, 255)
RED_HSV_LOWER_2 = (170, 100, 100)    # Red wraps around in HSV
RED_HSV_UPPER_2 = (180, 255, 255)

BLUE_HSV_LOWER = (100, 100, 100)     # Adjust for blue objects
BLUE_HSV_UPPER = (130, 255, 255)

# Detection Filters
MIN_CUBE_AREA_PX = 100              # Minimum size (pixels²)
MAX_CUBE_AREA_PX = 100000           # Maximum size
MIN_ASPECT_RATIO = 0.5              # Min width/height ratio
MAX_ASPECT_RATIO = 2.0              # Max width/height ratio
MIN_CONFIDENCE = 0.6                # Minimum detection confidence
```

**Tuning Process**:

1. **Enable debug mode**:
   ```bash
   python -m LLMCommunication.orchestrators.RunDetector --debug
   ```

2. **Capture test image** from Unity

3. **Check debug images** in `ACRLPython/LLMCommunication/debug_detections/`

4. **Adjust HSV ranges** based on what's being detected/missed

5. **Test again** until detection is reliable

**HSV Color Space Guide**:
- **H (Hue)**: 0-180 in OpenCV (0=red, 60=yellow, 120=green)
- **S (Saturation)**: 0-255 (0=gray, 255=pure color)
- **V (Value)**: 0-255 (0=black, 255=bright)

For more details, see `ACRLPython/LLMCommunication/OBJECT_DETECTION_README.md`.

---

## 7. Stereo Vision & Depth Estimation

### 7.1 Overview

Stereo vision provides accurate 3D localization without relying on Unity's physics raycasting. This is more realistic for transfer to real robots.

**Benefits**:
✅ Accurate depth without perfect scene geometry
✅ Real-world metric measurements
✅ Transferable to physical robots
✅ Better training data for robot learning

### 7.2 Camera Setup

**Requirements**:
- Two cameras with identical settings (FOV, resolution, rotation)
- Known baseline distance between cameras
- Parallel camera orientations (same rotation)

**Example Setup**:
```
Left Camera:
  Position: (0, 0.2, 0)
  Rotation: (0, 0, 0)
  FOV: 60°
  Resolution: 640x480

Right Camera:
  Position: (0.1, 0.2, 0)  # 0.1m baseline
  Rotation: (0, 0, 0)       # Same as left
  FOV: 60°
  Resolution: 640x480
```

### 7.3 Python Setup

**Start Stereo Detector**:
```bash
cd ACRLPython
source acrl/bin/activate

python -m LLMCommunication.orchestrators.RunStereoDetector \
    --baseline 0.1 \    # Camera separation in meters
    --fov 60           # Camera field of view in degrees

# Output:
# Stereo Object Detector with 3D Position Estimation
# Camera baseline: 0.1m
# Camera FOV: 60°
# Stereo detection server: 127.0.0.1:5009
# Results server: 127.0.0.1:5006
# Ready for stereo image pairs...
```

### 7.4 Unity Setup

**Configure CameraController for Stereo**:
```
Select CameraController:
  - Use Stereo Depth: ✓
  - Right Camera: [Drag right camera]
  - Stereo Baseline: 0.1  (match Python --baseline)
  - Camera FOV: 60        (match Python --fov)
```

**Verify StereoImageSender** exists:
```
GameObject → Create Empty → "StereoImageSender"
Add Component → StereoImageSender.cs
Inspector:
  - Server Host: 127.0.0.1
  - Server Port: 5009
  - Auto Connect: ✓
```

### 7.5 Sending Stereo Pairs

**Using CameraController**:
```
With "Use Stereo Depth" enabled:
Click "Send to LLM" → Automatically sends stereo pair
```

**Programmatically**:
```csharp
using LLMCommunication;

public class StereoCapture : MonoBehaviour {
    public Camera leftCamera;
    public Camera rightCamera;

    void Update() {
        if (Input.GetKeyDown(KeyCode.S)) {
            StereoImageSender.Instance.CaptureAndSendStereoPair(
                leftCamera,
                rightCamera,
                "detect cubes with depth"
            );
        }
    }
}
```

### 7.6 Receiving Stereo Results

**Detection results with stereo include `world_position`**:
```csharp
void HandleDetection(DetectionResultWithWorld result) {
    // Check if stereo depth was used
    if (result.OriginalResult.metadata?.detection_mode == "stereo_3d") {
        Debug.Log("Using stereo depth estimation");

        foreach (var cube in result.CubesWithWorldCoords) {
            if (cube.HasWorldPosition) {
                // Position calculated from stereo disparity, not raycast
                Vector3 worldPos = cube.WorldPosition;
                Debug.Log($"Stereo 3D position: {worldPos}");

                // Higher confidence than raycast-based positions
                float conf = cube.OriginalDetection.confidence;
            }
        }
    }
}
```

### 7.7 Calibration & Tuning

**Measure Baseline Accurately**:
```
Use ruler or calipers to measure camera center-to-center distance
Example: 10cm = 0.1m baseline
```

**Verify FOV**:
```csharp
// In Unity, check camera FOV:
Camera cam = GetComponent<Camera>();
Debug.Log($"Camera FOV: {cam.fieldOfView}");  // Should match Python --fov
```

**Test with Known Objects**:
```
1. Place object at known distance (e.g., 0.5m from cameras)
2. Run stereo detection
3. Compare estimated depth to ground truth
4. Adjust baseline/FOV if needed
```

For complete details, see `ACRLPython/LLMCommunication/STEREO_DEPTH_README.md`.

---

## 8. Data Logging

### 8.1 MainLogger System

**Purpose**: Unified logging for robot actions, environment state, and trajectories suitable for LLM training.

**Log Format**: JSONL (JSON Lines) - one JSON object per line, easy to stream and parse.

### 8.2 Configuration

**Inspector Settings** (MainLogger GameObject):
```
Enable Logging: ✓
Log Directory: [default: persistentDataPath/RobotLogs/]
Operation Type: "training"  (creates subfolder)
Per Robot Files: ✓  (separate file per robot)

Capture Environment: ✓  (periodic snapshots)
Environment Sample Rate: 2.0s

Track Trajectories: ✓  (record robot paths)
Trajectory Sample Rate: 0.2s
```

**File Organization**:
```
{persistentDataPath}/RobotLogs/
└── training/
    └── session_2025-10-25_14-30-00/
        ├── Robot1_actions.jsonl
        ├── Robot2_actions.jsonl
        └── environment_snapshots.jsonl
```

### 8.3 Logging Actions

**Start and Complete Actions**:
```csharp
using UnityEngine;
using System.Collections.Generic;

public class RobotTaskExecutor : MonoBehaviour {
    void ExecutePickupTask(Vector3 targetPos) {
        // Start tracking the action
        string actionId = MainLogger.Instance.StartAction(
            actionName: "pickup_cube",
            type: ActionType.Manipulation,
            robotIds: new[] { "Robot1" },
            startPos: transform.position,
            targetPos: targetPos,
            description: "Picking up red cube from table"
        );

        // Perform the action...
        MoveToTarget(targetPos);
        GripCube();

        // Complete the action
        var metrics = new Dictionary<string, float> {
            ["distance_traveled"] = Vector3.Distance(transform.position, targetPos),
            ["time_elapsed"] = 5.2f,
            ["grasp_force"] = 12.5f
        };

        MainLogger.Instance.CompleteAction(
            actionId: actionId,
            success: true,
            qualityScore: 0.95f,  // 0.0 to 1.0
            metrics: metrics
        );
    }
}
```

**ActionType Enum**:
- `Task`: High-level task (e.g., "assemble widget")
- `Movement`: Navigation/positioning
- `Manipulation`: Grasping, placing objects
- `Coordination`: Multi-robot cooperation
- `Observation`: Sensing, vision capture

### 8.4 Logging Events

**For instantaneous events**:
```csharp
// Collision detected
MainLogger.Instance.LogSimulationEvent(
    eventName: "collision_detected",
    description: "Robot arm collided with table edge",
    robotIds: new[] { "Robot1" },
    objectIds: new[] { "Table" }
);

// Vision processing complete
MainLogger.Instance.LogSimulationEvent(
    eventName: "object_detected",
    description: "Red cube detected at (0.5, 0.1, 0.3)",
    robotIds: new[] { "Robot1" },
    objectIds: new[] { "RedCube_01" }
);
```

### 8.5 Multi-Robot Coordination Logging

```csharp
// Log coordination between robots
string coordId = MainLogger.Instance.LogCoordination(
    coordinationName: "dual_arm_handoff",
    robotIds: new[] { "Robot1", "Robot2" },
    description: "Passing cube from Robot1 to Robot2",
    objectIds: new[] { "BlueCube" }
);

// Later, complete the coordination action
MainLogger.Instance.CompleteAction(
    actionId: coordId,
    success: true,
    qualityScore: 0.88f
);
```

### 8.6 Object Registration

**Register scene objects for tracking**:
```csharp
void Start() {
    // Register all cubes in scene
    GameObject[] cubes = GameObject.FindGameObjectsWithTag("Cube");
    foreach (var cube in cubes) {
        MainLogger.Instance.RegisterObject(
            gameObject: cube,
            objectType: "cube",
            isGraspable: true
        );
    }

    // Register static objects
    MainLogger.Instance.RegisterObject(
        gameObject: GameObject.Find("Table"),
        objectType: "furniture",
        isGraspable: false
    );
}
```

### 8.7 AutoLogger Component

**Automatic logging without manual code**:

1. Attach `AutoLogger.cs` to robot GameObject
2. Configure in Inspector:
   ```
   Enable Auto Logging: ✓
   Log Movement: ✓
   Log Gripper: ✓
   Auto Register Objects: ✓
   ```

AutoLogger automatically:
- Logs movement actions when target changes
- Logs gripper open/close events
- Registers scene objects with colliders
- No additional code needed

### 8.8 Exporting for LLM Training

**Unity Editor Menu**: `Tools > Robot Logging > ...`

**Export to JSONL**:
```
Tools > Robot Logging > Export to JSONL
- Select source log file
- Choose output location
- Creates clean JSONL for training
```

**Export to Conversational Format**:
```
Tools > Robot Logging > Export to Conversational
- Converts to chat format for LLM fine-tuning
- Format: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
```

**Generate Statistics**:
```
Tools > Robot Logging > Generate Statistics
- Shows success rate by action type
- Average quality scores
- Time distributions
- Object interaction counts
```

**Programmatic Export**:
```csharp
using UnityEngine;

public class LogExporter : MonoBehaviour {
    void ExportLogs() {
        string sourceLog = Application.persistentDataPath + "/RobotLogs/training/session.jsonl";
        string outputFile = Application.dataPath + "/../TrainingData/cleaned.jsonl";

        LLMExporter.ExportToJSONL(sourceLog, outputFile);

        // Or conversational format
        LLMExporter.ExportToConversational(sourceLog, outputFile);

        // Or statistics
        var stats = LLMExporter.GenerateStatistics(sourceLog);
        Debug.Log($"Total actions: {stats[\"total_actions\"]}");
        Debug.Log($"Success rate: {stats[\"success_rate\"]}%");
    }
}
```

---

## 9. Development Guide

### 9.1 Project Structure

```
Auto-Cooperative-Robot-Learning/
├── ACRLUnity/                    # Unity project
│   ├── Assets/
│   │   ├── Configuration/        # ScriptableObject configs
│   │   ├── Data/                 # Trained models (.onnx)
│   │   ├── Prefabs/              # Robot, environment prefabs
│   │   ├── Scenes/               # Simulation scenes
│   │   └── Scripts/
│   │       ├── ConfigScripts/    # Config ScriptableObjects
│   │       ├── RobotScripts/     # Robot control
│   │       ├── CameraScripts/    # Vision capture
│   │       ├── Logging/          # Data logging
│   │       ├── LLMCommunication/ # TCP clients
│   │       └── SimulationScripts/# Managers
│   ├── Packages/                 # Unity packages
│   └── ProjectSettings/          # Unity settings
│
├── ACRLPython/                   # Python backend
│   ├── LLMCommunication/
│   │   ├── core/                 # Base classes
│   │   ├── servers/              # TCP servers
│   │   ├── vision/               # Detection, LLM
│   │   ├── orchestrators/        # Main entry points
│   │   ├── config.py             # Configuration
│   │   ├── OBJECT_DETECTION_README.md
│   │   └── STEREO_DEPTH_README.md
│   ├── StereoImageReconstruction/# Stereo vision library
│   ├── Tests/                    # Unit tests
│   └── acrl/                     # Python venv
│
├── ml-agents/                    # Git submodule
│   ├── ml-agents/                # Python package
│   ├── ml-agents-envs/           # Environment API
│   └── venv/                     # ML Python venv
│
├── CLAUDE.md                     # Claude Code guidance
├── README.md                     # Project README
└── USER_GUIDE.md                 # This file
```

### 9.2 Coding Conventions

**C# (Unity)**:

1. **Singleton Pattern** (for managers):
   ```csharp
   public class MyManager : MonoBehaviour {
       public static MyManager Instance { get; private set; }

       void Awake() {
           if (Instance == null) {
               Instance = this;
               DontDestroyOnLoad(gameObject);
               InitializeManager();
           } else {
               Destroy(gameObject);
           }
       }
   }
   ```

2. **Event-Driven Communication**:
   ```csharp
   public event Action<StateType> OnStateChanged;

   void ChangeState(StateType newState) {
       OnStateChanged?.Invoke(newState);
   }
   ```

3. **Docstrings on All Functions**:
   ```csharp
   /// <summary>
   /// Sets the target position for the robot to reach.
   /// </summary>
   /// <param name="position">Target position in world space</param>
   /// <param name="rotation">Target rotation</param>
   public void SetTarget(Vector3 position, Quaternion rotation) { ... }
   ```

4. **Performance Optimization**:
   - Use `TryGetComponent<T>()` instead of `GetComponent<T>()`
   - Pre-allocate arrays/matrices for frequently called code
   - Avoid string concatenation in hot paths

**Python**:

1. **Type Hints**:
   ```python
   def detect_cubes(image: np.ndarray) -> List[DetectionObject]:
       ...
   ```

2. **Docstrings**:
   ```python
   def estimate_depth(imgL: np.ndarray, imgR: np.ndarray) -> float:
       """
       Estimate depth using stereo disparity.

       Args:
           imgL: Left camera image
           imgR: Right camera image

       Returns:
           Estimated depth in meters
       """
       ...
   ```

3. **Relative Imports**:
   ```python
   from ..core.TCPServerBase import TCPServerBase
   from .. import config as cfg
   ```

### 9.3 Testing

**Unity Tests** (`Window > General > Test Runner`):
- EditMode tests: Validate logic without Play mode
- PlayMode tests: Integration tests with scene

**Python Tests**:
```bash
cd ACRLPython
source acrl/bin/activate
pytest Tests/ -v

# With coverage
pytest Tests/ --cov=LLMCommunication --cov-report=html
```

### 9.4 Git Workflow

```bash
# Feature development
git checkout -b feature_my_feature
# ... make changes ...
git add .
git commit -m "Add my feature"
git push origin feature_my_feature

# Create PR to main branch

# Merge conflicts
git pull origin main
# ... resolve conflicts ...
git commit -m "Merge main into feature_my_feature"
```

### 9.5 Adding New Features

**Example: Add New Coordination Mode**

1. **Update SimulationConfig.cs**:
   ```csharp
   public enum CoordinationMode {
       Independent,
       Collaborative,
       MyNewMode  // Add new mode
   }
   ```

2. **Implement Logic in SimulationManager.cs**:
   ```csharp
   void UpdateRobots() {
       switch (config.coordinationMode) {
           case CoordinationMode.MyNewMode:
               UpdateMyNewMode();
               break;
           // ... other cases ...
       }
   }

   void UpdateMyNewMode() {
       // Implement coordination logic
   }
   ```

3. **Test**:
   - Create test scene
   - Set coordination mode to MyNewMode
   - Verify behavior

4. **Document** in CLAUDE.md and this guide

---

## 10. Troubleshooting

### 10.1 Unity Issues

**Problem**: Robot jerks or oscillates

**Solution**:
- Reduce `adjustmentSpeed` in RobotConfig
- Increase joint damping values
- Lower `maxJointStepRad`

---

**Problem**: IK not converging

**Solution**:
- Check target is within reach (< `maxReachDistance`)
- Verify joint angle limits aren't too restrictive
- Increase `convergenceThreshold` if accuracy not critical
- Check for gimbal lock (avoid extreme rotations)

---

**Problem**: ML training not improving

**Solution**:
- Verify observations are normalized
- Check reward function returns non-zero values
- Reduce `learning_rate` (try 1e-4)
- Increase `batch_size` for stability
- Ensure task is solvable (test manually)

---

**Problem**: NullReferenceException in manager singletons

**Solution**:
- Ensure manager GameObject exists in scene
- Check Awake() runs before accessing Instance
- Use `?` operator: `Manager.Instance?.Method()`

---

### 10.2 Python Communication Issues

**Problem**: "Connection refused" when Unity tries to connect

**Solution**:
```bash
# Check if Python server is running
lsof -nP -iTCP:5005  # Or 5006, 5007, 5009

# If not running, start it:
cd ACRLPython
source acrl/bin/activate
python -m LLMCommunication.orchestrators.RunDetector

# Check firewall isn't blocking ports
```

---

**Problem**: Images not reaching Python

**Solution**:
- Verify `ImageSender` is connected (check Unity Inspector)
- Confirm `StreamingServer` started (check Python terminal)
- Check image size < 10MB (UnityProtocol limit)
- Look for errors in Unity Console and Python terminal

---

**Problem**: Detection results empty (no cubes detected)

**Solution**:
```bash
# Enable debug mode to see what's being detected
python -m LLMCommunication.orchestrators.RunDetector --debug

# Check saved images in:
# ACRLPython/LLMCommunication/debug_detections/

# Adjust color ranges in config.py
# Test with known good image
```

---

**Problem**: Stereo depth estimation failing

**Solution**:
- Verify cameras have identical resolution and FOV
- Check baseline distance matches Python `--baseline` parameter
- Ensure cameras are parallel (same rotation)
- Increase texture/contrast in scene (stereo needs features)
- Check image size mismatch error in Python logs

---

### 10.3 ML-Agents Training Issues

**Problem**: "Couldn't connect to Unity environment"

**Solution**:
- Press Play in Unity Editor when prompted
- Check Unity isn't paused or in background
- Verify mlagents-learn version matches package version
- Try standalone build instead of Editor

---

**Problem**: Training crashes with NaN values

**Solution**:
- Reduce `learning_rate` (try 3e-5)
- Check observation values aren't extreme (normalize inputs)
- Verify reward function doesn't return inf/NaN
- Increase `batch_size` for more stable gradients

---

**Problem**: Trained model performs poorly

**Solution**:
- Train longer (`max_steps` > 1M)
- Simplify task first, then add complexity
- Check reward signal aligns with desired behavior
- Use curriculum learning (easy→hard scenarios)
- Monitor TensorBoard for reward plateaus

---

### 10.4 General Debugging Tips

**Enable Verbose Logging**:
```csharp
// Unity
Debug.Log($"[MyScript] Variable value: {myVar}");

// Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Port Usage**:
```bash
# macOS/Linux
lsof -nP -iTCP | grep LISTEN

# Kill process on port
kill -9 $(lsof -t -i:5005)
```

**Verify Python Environment**:
```bash
# Check active environment
which python  # Should show venv path

# List installed packages
pip list

# Verify imports
python -c "import cv2; print(cv2.__version__)"
```

**Unity Console Filters**:
- Use Console filters to show only Errors, Warnings, or specific tags
- Collapse identical messages
- Enable "Show timestamp" for debugging timing issues

---

## 11. Reference

### 11.1 Port Reference

| Port | Service | Direction | Purpose |
|------|---------|-----------|---------|
| 5005 | StreamingServer | Unity → Python | Send camera images |
| 5006 | ResultsServer | Python → Unity | Send LLM/detection results |
| 5007 | DetectionServer | Python → Unity | Send object detection results |
| 5009 | StereoDetectionServer | Unity → Python | Send stereo image pairs |
| 6006 | TensorBoard | N/A | ML training monitoring (HTTP) |

### 11.2 Key File Locations

**Unity**:
- Robot configs: `ACRLUnity/Assets/Configuration/*.asset`
- Trained models: `ACRLUnity/Assets/Data/*.onnx`
- Training config: `ACRLUnity/Assets/Configuration/RobotNavigation.yaml`
- Log files: `{persistentDataPath}/RobotLogs/`
- Persistent data: `~/Library/Application Support/JanMStraub/ACRL/` (macOS)

**Python**:
- Config: `ACRLPython/LLMCommunication/config.py`
- Detection guide: `ACRLPython/LLMCommunication/OBJECT_DETECTION_README.md`
- Stereo guide: `ACRLPython/LLMCommunication/STEREO_DEPTH_README.md`
- Test suite: `ACRLPython/Tests/`

**ML-Agents**:
- Training results: `ml-agents/results/{run_id}/`
- Trained models: `ml-agents/results/{run_id}/{behavior_name}/*.onnx`
- TensorBoard logs: `ml-agents/results/`

### 11.3 Command Cheat Sheet

**Unity**:
```bash
# Open Unity project
open -a Unity ACRLUnity/

# Build standalone
# File > Build Settings > Build
```

**ML Training**:
```bash
# Setup environment
cd ml-agents && source venv/bin/activate

# Start training
mlagents-learn ../ACRLUnity/Assets/Configuration/RobotNavigation.yaml --run-id=test

# Resume training
mlagents-learn ../ACRLUnity/Assets/Configuration/RobotNavigation.yaml --run-id=test --resume

# Monitor with TensorBoard
tensorboard --logdir results/
```

**LLM/Detection**:
```bash
# Setup environment
cd ACRLPython && source acrl/bin/activate

# Start LLM analyzer
python -m LLMCommunication.orchestrators.RunAnalyzer --model llava

# Start object detector
python -m LLMCommunication.orchestrators.RunDetector

# Start stereo detector
python -m LLMCommunication.orchestrators.RunStereoDetector --baseline 0.1 --fov 60

# Run tests
pytest Tests/ -v
```

**Git**:
```bash
# Clone with submodules
git clone --recursive https://github.com/JanMStraub/Auto-Cooperative-Robot-Learning.git

# Update submodules
git submodule update --init --recursive

# Create feature branch
git checkout -b feature_name

# Merge latest main
git pull origin main
```

### 11.4 Common Paths

**macOS**:
- Unity persistent data: `~/Library/Application Support/JanMStraub/ACRL/`
- Unity logs: `~/Library/Logs/JanMStraub/ACRL/`

**Linux**:
- Unity persistent data: `~/.config/unity3d/JanMStraub/ACRL/`
- Unity logs: `~/.config/unity3d/JanMStraub/ACRL/`

**Windows**:
- Unity persistent data: `%USERPROFILE%\AppData\LocalLow\JanMStraub\ACRL\`
- Unity logs: `%USERPROFILE%\AppData\Local\Unity\Editor\`

### 11.5 Useful Links

**Project**:
- GitHub: https://github.com/JanMStraub/Auto-Cooperative-Robot-Learning
- Issues: https://github.com/JanMStraub/Auto-Cooperative-Robot-Learning/issues

**Documentation**:
- Unity ML-Agents: https://github.com/Unity-Technologies/ml-agents
- Unity ArticulationBody: https://docs.unity3d.com/Manual/class-ArticulationBody.html
- Ollama: https://ollama.ai/
- OpenCV: https://docs.opencv.org/

**External Tools**:
- MathNet.Numerics: https://numerics.mathdotnet.com/
- AR4 Robot: https://github.com/zebleck/AR4

### 11.6 FAQ

**Q: Can I run multiple detection systems simultaneously?**
A: Yes! You can run RunAnalyzer (LLM), RunDetector (mono detection), and RunStereoDetector simultaneously. They use different ports and don't interfere.

**Q: How do I transfer a trained model to a real robot?**
A: Export the .onnx model and use ONNX Runtime on your robot's computer. You'll need to replicate the observation space (joint angles, positions) and action space (joint targets).

**Q: What's the difference between MainLogger and Unity's Debug.Log?**
A: MainLogger creates structured JSONL files for LLM training. Debug.Log is for developer debugging only. Use MainLogger for production logging.

**Q: Can I use a different LLM besides Ollama?**
A: Yes, edit `ACRLPython/LLMCommunication/vision/AnalyzeImage.py` to call your preferred API (OpenAI, Anthropic, etc.). Keep the same result format.

**Q: Why are there two Python environments?**
A: ML-Agents requires PyTorch and specific versions. LLM communication has different dependencies. Keeping them separate prevents conflicts.

**Q: How do I add more robots to a scene?**
A: Duplicate AR4_Robot prefab, assign unique names, register with RobotManager, and update SimulationManager coordination logic.

**Q: What's the maximum number of robots supported?**
A: Technically unlimited, but performance degrades beyond 16 robots. Use standalone build for better performance than Editor.

**Q: Can I use this for non-AR4 robots?**
A: Yes, but you'll need to create new robot model, update joint chain, reconfigure IK solver, and adjust observation/action spaces.

---

## Conclusion

This guide covers the essential aspects of the ACRL system. For additional details:

- **CLAUDE.md**: Technical implementation details for AI assistant
- **README.md**: Project overview and quick start
- **OBJECT_DETECTION_README.md**: Detailed object detection guide
- **STEREO_DEPTH_README.md**: Stereo vision depth estimation guide

For questions, issues, or contributions:
- GitHub Issues: https://github.com/JanMStraub/Auto-Cooperative-Robot-Learning/issues
- Contact: @JanMStraub

Happy robot learning! 🤖
