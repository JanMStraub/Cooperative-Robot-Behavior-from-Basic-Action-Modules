# Restructure to SequenceClient-Only Architecture

This plan outlines changes to simplify the system so that **SequenceClient** is the main Unity entry point for LLM-based robot control, while maintaining object detection capabilities.

---

## Overview

**Goal:** Unity sends natural language prompts → Python uses LLM to generate operations → Python executes operations (including detection) → Results returned to Unity via SequenceClient

**Required Ports (4 total):**
- **5005**: StreamingServer - Receives camera images from Unity
- **5010**: ResultsServer - Sends commands TO Unity
- **5012**: StatusServer - Receives completion signals FROM Unity
- **5013**: SequenceServer - Receives NL commands FROM Unity

---

## Phase 1: Create Detection Operations

### New File: `operations/DetectionOperations.py`

Add detection as operations that CommandParser can invoke:

```python
# operations/DetectionOperations.py

from operations.Base import BasicOperation, OperationParameter, OperationCategory, OperationComplexity
from vision.ObjectDetector import ObjectDetector
from vision.DepthEstimator import DepthEstimator
from servers.StreamingServer import ImageStorage

def detect_objects_execute(robot_id: str, camera_id: str = "main") -> dict:
    """Execute object detection on camera image."""
    image_storage = ImageStorage.get_instance()
    image_bytes = image_storage.get_camera_image(camera_id)

    if not image_bytes:
        return {"success": False, "error": "No image available from camera"}

    detector = ObjectDetector()
    detections = detector.detect_objects(image_bytes)

    return {
        "success": True,
        "detections": detections,
        "camera_id": camera_id
    }

def detect_with_depth_execute(robot_id: str, left_camera: str = "left", right_camera: str = "right", baseline: float = 0.1, fov: float = 60.0) -> dict:
    """Execute stereo detection with 3D coordinates."""
    image_storage = ImageStorage.get_instance()
    left_image = image_storage.get_camera_image(left_camera)
    right_image = image_storage.get_camera_image(right_camera)

    if not left_image or not right_image:
        return {"success": False, "error": "Stereo images not available"}

    detector = ObjectDetector()
    depth_estimator = DepthEstimator()

    left_detections = detector.detect_objects(left_image)
    right_detections = detector.detect_objects(right_image)

    positions_3d = depth_estimator.estimate_depth(
        left_image, right_image,
        left_detections,
        baseline, fov
    )

    return {
        "success": True,
        "detections_3d": positions_3d
    }

# Register operations
detect_objects = BasicOperation(
    name="detect_objects",
    description="Detect objects in camera image using color-based detection",
    category=OperationCategory.PERCEPTION,
    complexity=OperationComplexity.BASIC,
    parameters=[
        OperationParameter("robot_id", str, "Robot identifier", required=True),
        OperationParameter("camera_id", str, "Camera to use", required=False, default="main")
    ],
    execute_func=detect_objects_execute,
    examples=["detect objects", "find cubes", "scan for objects"]
)

detect_with_depth = BasicOperation(
    name="detect_with_depth",
    description="Detect objects and calculate 3D positions using stereo vision",
    category=OperationCategory.PERCEPTION,
    complexity=OperationComplexity.INTERMEDIATE,
    parameters=[
        OperationParameter("robot_id", str, "Robot identifier", required=True),
        OperationParameter("left_camera", str, "Left camera ID", required=False, default="left"),
        OperationParameter("right_camera", str, "Right camera ID", required=False, default="right"),
        OperationParameter("baseline", float, "Camera baseline in meters", required=False, default=0.1),
        OperationParameter("fov", float, "Camera field of view in degrees", required=False, default=60.0)
    ],
    execute_func=detect_with_depth_execute,
    examples=["detect objects with depth", "find 3D positions of cubes", "locate objects in 3D"]
)
```

### Update `operations/Registry.py`

Register the new detection operations:

```python
from operations.DetectionOperations import detect_objects, detect_with_depth

# In registry initialization
registry.register(detect_objects)
registry.register(detect_with_depth)
```

---

## Phase 2: Update RunSequenceServer

### Modify `orchestrators/RunSequenceServer.py`

Add StreamingServer startup for image reception:

```python
def main():
    # ... existing code ...

    # Start StreamingServer for image reception (needed for detection)
    from servers.StreamingServer import run_streaming_server_background
    streaming_thread = run_streaming_server_background(config)
    logger.info("StreamingServer started on port 5005")

    # Start ResultsServer (existing)
    # Start StatusServer (existing)
    # Start SequenceServer (existing)
```

---

## Phase 3: Update CommandParser

### Modify `orchestrators/CommandParser.py`

Ensure regex patterns include detection:

```python
# In _parse_with_regex method, add pattern:
elif re.search(r'detect.*(?:depth|3d|position)|find.*(?:3d|position)', part, re.I):
    commands.append({
        "operation": "detect_with_depth",
        "params": {"robot_id": robot_id}
    })
elif re.search(r'detect|find|scan|locate', part, re.I):
    commands.append({
        "operation": "detect_objects",
        "params": {"robot_id": robot_id}
    })
```

---

## Phase 4: Files to Remove

### Unity Scripts (DELETE):

```
ACRLUnity/Assets/Scripts/PythonCommunication/
├── RAGClient.cs                    # DELETE - RAG queries no longer needed
├── LLMResultsReceiver.cs           # DELETE - Use SequenceClient for all results
└── (keep everything else)
```

### Python Files (DELETE):

```
ACRLPython/
├── orchestrators/
│   ├── RunAnalyzer.py              # DELETE - Vision analysis via SequenceServer now
│   ├── RunDetector.py              # DELETE - Detection via operations now
│   ├── RunStereoDetector.py        # DELETE - Stereo via operations now
│   └── RunRAGServer.py             # DELETE - Not using RAG
├── servers/
│   └── RAGServer.py                # DELETE - Not using RAG
└── rag/                            # DELETE entire folder - Not using RAG
    ├── __init__.py
    ├── Config.py
    ├── Embeddings.py
    ├── VectorStore.py
    ├── Indexer.py
    ├── QueryEngine.py
    └── .rag_index.pkl
```

---

## Phase 5: Files to Keep

### Unity Scripts (KEEP):

```
ACRLUnity/Assets/Scripts/PythonCommunication/
├── Core/
│   ├── TCPClientBase.cs            # KEEP - Base infrastructure
│   ├── UnityProtocol.cs            # KEEP - Wire protocol
│   ├── ErrorHandling.cs            # KEEP - Error codes
│   └── JsonParser.cs               # KEEP - JSON parsing
├── SequenceClient.cs               # KEEP - Main entry point
├── PythonCommandHandler.cs         # KEEP - Executes commands from Python
├── UnifiedPythonSender.cs          # KEEP - Sends camera images
├── DepthResultsReceiver.cs         # KEEP - OR merge into SequenceClient
├── StatusResponseSender.cs         # KEEP - Sends completion signals
├── DetectionDataModels.cs          # KEEP - Detection models
└── DataModels/
    └── SequenceDataModels.cs       # KEEP - Sequence result models
```

### Python Files (KEEP):

```
ACRLPython/
├── core/
│   ├── __init__.py                 # KEEP
│   ├── TCPServerBase.py            # KEEP - Base class
│   └── UnityProtocol.py            # KEEP - Wire protocol
├── servers/
│   ├── __init__.py                 # KEEP
│   ├── SequenceServer.py           # KEEP - Main entry (port 5013)
│   ├── ResultsServer.py            # KEEP - Commands to Unity (port 5010)
│   ├── StatusServer.py             # KEEP - Completion signals (port 5012)
│   └── StreamingServer.py          # KEEP - Image reception (port 5005)
├── orchestrators/
│   ├── __init__.py                 # KEEP
│   ├── RunSequenceServer.py        # KEEP - Main orchestrator
│   ├── CommandParser.py            # KEEP - LLM parsing
│   └── SequenceExecutor.py         # KEEP - Operation execution
├── operations/
│   ├── __init__.py                 # KEEP
│   ├── Base.py                     # KEEP - Operation classes
│   ├── MoveOperations.py           # KEEP - Move operations
│   ├── GripperOperations.py        # KEEP - Gripper operations
│   ├── StatusOperations.py         # KEEP - Status operations
│   ├── DetectionOperations.py      # NEW - Detection operations
│   └── Registry.py                 # KEEP - Operation registry
├── vision/
│   ├── __init__.py                 # KEEP
│   ├── ObjectDetector.py           # KEEP - Color detection
│   └── DepthEstimator.py           # KEEP - Stereo depth
├── LLMConfig.py                    # KEEP - Configuration
└── tests/                          # KEEP - Update tests
```

---

## Phase 6: Update Imports and __init__.py

### Update `servers/__init__.py`:

```python
from .SequenceServer import SequenceServer, SequenceQueryHandler
from .ResultsServer import ResultsServer, ResultsBroadcaster
from .StatusServer import StatusServer, StatusResponseHandler
from .StreamingServer import StreamingServer, ImageStorage
# Remove: RAGServer
```

### Update `orchestrators/__init__.py`:

```python
from .RunSequenceServer import main
from .CommandParser import CommandParser
from .SequenceExecutor import SequenceExecutor
# Remove: RunAnalyzer, RunDetector, RunStereoDetector, RunRAGServer
```

---

## Example Flows

### Flow 1: Simple Movement
```
Unity: SequenceClient.ExecuteSequence("move to (0.3, 0.2, 0.1) and close gripper", "Robot1")
  ↓
Python: CommandParser → [move_to_coordinate, control_gripper]
  ↓
Python: SequenceExecutor executes each operation
  ↓
Unity: SequenceClient.OnSequenceResultReceived
```

### Flow 2: Detection + Movement
```
Unity: SequenceClient.ExecuteSequence("find the red cube and move to it", "Robot1")
  ↓
Python: CommandParser → [detect_with_depth, move_to_coordinate(use detected position)]
  ↓
Python: SequenceExecutor:
  1. detect_with_depth → finds red cube at (0.35, 0.18, 0.12)
  2. move_to_coordinate(0.35, 0.18, 0.12)
  ↓
Unity: SequenceClient.OnSequenceResultReceived with detection + movement results
```

### Flow 3: Full Pick Operation
```
Unity: SequenceClient.ExecuteSequence("pick up the blue cube", "Robot1")
  ↓
Python: CommandParser → [detect_with_depth, move_to_coordinate, control_gripper(close), move_up]
  ↓
Python: SequenceExecutor executes sequence
  ↓
Unity: SequenceClient.OnSequenceResultReceived
```

---

## Server Startup Command

After restructuring, start the system with:

```bash
cd ACRLPython
python -m orchestrators.RunSequenceServer --model gemma-3-12b
```

This starts:
- StreamingServer (5005)
- ResultsServer (5010)
- StatusServer (5012)
- SequenceServer (5013)

---

## Testing

### Test detection operations:

```python
# Test script
from operations import get_global_registry

registry = get_global_registry()

# Test detect_objects
result = registry.execute_operation_by_name(
    "detect_objects",
    robot_id="Robot1",
    camera_id="main"
)
print(result)

# Test detect_with_depth
result = registry.execute_operation_by_name(
    "detect_with_depth",
    robot_id="Robot1"
)
print(result)
```

### Test full sequence:

```bash
python -m orchestrators.RunSequenceServer --test
```

---

## Migration Notes

1. **Backup first**: Create a branch or backup before removing files
2. **Update tests**: Remove tests for deleted components, add tests for DetectionOperations
3. **Update CLAUDE.md**: Remove documentation for deleted components
4. **LM Studio required**: Ensure LM Studio is running with an appropriate model for parsing

---

## Questions to Consider

1. **Should detection results go directly back through SequenceClient?** (Yes, recommended)
2. **Should DepthResultsReceiver be merged into SequenceClient?** (Optional simplification)
3. **Do you need the image classification feature from AnalyzeImage.py?** (Can add as operation if needed)
