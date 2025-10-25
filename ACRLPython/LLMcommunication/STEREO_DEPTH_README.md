# Stereo Depth Estimation for Object Localization

## Overview

This system enables precise 3D object localization using stereo vision depth estimation. Instead of relying on raycasting in Unity, the system uses stereo disparity between two cameras to calculate accurate real-world 3D coordinates of detected objects.

## Architecture

### Python Components

#### 1. **DepthEstimator.py**
Core stereo depth estimation module.

**Functions:**
- `estimate_depth_at_point(imgL, imgR, pixel_x, pixel_y, camera_config)` - Estimates depth at a specific pixel using stereo disparity
- `pixel_to_world_coords(pixel_x, pixel_y, depth, camera_config)` - Converts 2D pixel + depth → 3D world coordinates
- `estimate_object_world_position(imgL, imgR, bbox_center_x, bbox_center_y, camera_config)` - Complete pipeline for object localization

**Dependencies:**
- Uses `StereoImageReconstruction` module for disparity calculation
- OpenCV for SGBM stereo matching
- NumPy for numerical operations

#### 2. **ObjectDetector.py** (Updated)
Color-based cube detector with stereo support.

**New Method:**
- `detect_cubes_stereo(imgL, imgR, camera_config)` - Detects objects and estimates 3D positions

**Output Format:**
```python
{
    "id": 0,
    "color": "red",
    "bbox_px": {...},
    "center_px": {...},
    "confidence": 0.95,
    "world_position": {"x": 0.5, "y": 0.2, "z": 1.0}  # NEW: 3D position in meters
}
```

#### 3. **StereoDetectionServer.py**
TCP server receiving stereo image pairs from Unity.

**Port:** 5009
**Protocol:**
```
[cam_pair_id_len:4][cam_pair_id:N]
[camera_L_id_len:4][camera_L_id:N]
[camera_R_id_len:4][camera_R_id:N]
[prompt_len:4][prompt:N]
[image_L_len:4][image_L_data:N]
[image_R_len:4][image_R_data:N]
```

**Features:**
- Thread-safe stereo image storage
- Supports multiple camera pairs
- Automatic image decoding and validation

#### 4. **RunStereoDetector.py**
Main orchestrator for the stereo detection pipeline.

**Workflow:**
1. Starts `StereoDetectionServer` (port 5009) - receives stereo pairs
2. Starts `ResultsServer` (port 5006) - sends results to Unity
3. Monitors for new stereo pairs
4. Runs detection with depth estimation
5. Broadcasts results with 3D world positions

**Usage:**
```bash
python RunStereoDetector.py --baseline 0.1 --fov 60
```

**Arguments:**
- `--baseline` - Camera baseline distance in meters (default: 0.1)
- `--fov` - Camera field of view in degrees (default: 60)
- `--detection-port` - Stereo server port (default: 5009)
- `--results-port` - Results server port (default: 5006)

### Unity Components

#### 1. **StereoImageSender.cs**
Sends stereo image pairs to Python.

**Namespace:** `LLMCommunication`
**Singleton:** `StereoImageSender.Instance`
**Port:** 5009

**Inspector Fields:**
- `Left Camera` - Left camera for stereo pair
- `Right Camera` - Right camera for stereo pair
- `Camera Pair ID` - Identifier for the camera pair
- `Enable Streaming` - Continuous stereo streaming mode
- `Send Interval` - Time between frames (default: 0.5s)

**API:**
```csharp
// Send stereo pair
StereoImageSender.Instance.CaptureAndSendStereoPair(leftCamera, rightCamera, "detect cubes");

// Or send pre-encoded images
StereoImageSender.Instance.SendStereoPair(leftBytes, rightBytes, "stereo", "L", "R", "prompt");
```

#### 2. **DetectionDataModels.cs** (Updated)
Enhanced detection models with 3D position support.

**New Class:**
```csharp
[Serializable]
public class WorldPosition
{
    public float x, y, z;  // Meters
    public Vector3 ToVector3();
    public bool IsValid();
}
```

**Updated DetectionObject:**
```csharp
public class DetectionObject
{
    // Existing fields...
    public WorldPosition world_position;  // NEW: Optional 3D position
}
```

**Updated DetectionMetadata:**
```csharp
public class DetectionMetadata
{
    public float camera_baseline_m;
    public float camera_fov_deg;
    public string detection_mode;  // "mono_2d" or "stereo_3d"
}
```

#### 3. **DetectionResultsReceiver.cs** (Updated)
Automatically uses stereo depth data when available.

**Logic:**
```csharp
if (detection.world_position != null && detection.world_position.IsValid())
{
    // Use stereo depth estimation
    worldPos = detection.world_position.ToVector3();
}
else
{
    // Fall back to raycasting
    detection.TryGetWorldPosition(camera, ...);
}
```

#### 4. **CameraController.cs** (Updated)
Supports stereo mode for object detection.

**New Inspector Fields:**
- `Use Stereo Depth` - Enable stereo depth estimation
- `Right Camera` - Right camera for stereo pair
- `Stereo Baseline` - Distance between cameras (meters)
- `Camera FOV` - Field of view in degrees

**Behavior:**
- When `Use Stereo Depth` is enabled and `Send to LLM` is clicked:
  - Captures both left and right camera images
  - Sends stereo pair to `StereoImageSender` (port 5009)
  - Results include 3D world positions
- When disabled:
  - Sends single image to `ImageSender` (port 5005)
  - Results use raycasting for world positions

#### 5. **Logging/DataModels.cs** (Updated)
RobotAction now includes stereo depth data.

**New Fields:**
```csharp
public class RobotAction
{
    // Existing fields...
    public Vector3? detectedTargetWorldPosition;  // 3D position from stereo
    public float? depthEstimationConfidence;
    public string depthEstimationMethod;  // "stereo_disparity" or "raycast"
}
```

## Setup Guide

### Python Side

1. **Install dependencies:**
```bash
cd ACRLPython
pip install opencv-python numpy
```

2. **Start the stereo detector:**
```bash
python LLMcommunication/RunStereoDetector.py --baseline 0.1 --fov 60
```

You should see:
```
Stereo Object Detector with 3D Position Estimation
Camera baseline: 0.1m
Camera FOV: 60°
Stereo detection server: 127.0.0.1:5009
Results server: 127.0.0.1:5006
Stereo detector ready - waiting for stereo image pairs from Unity
```

### Unity Side

1. **Add StereoImageSender to scene:**
   - Create empty GameObject: `StereoImageSender`
   - Add component: `StereoImageSender.cs`
   - Configure in inspector:
     - Server Host: `127.0.0.1`
     - Server Port: `5009`
     - Auto Connect: ✓

2. **Configure CameraController for stereo:**
   - Select your robot's CameraController GameObject
   - In Inspector:
     - `Use Stereo Depth`: ✓
     - `Right Camera`: Assign your right camera
     - `Stereo Baseline`: `0.1` (match Python baseline)
     - `Camera FOV`: `60` (match Python FOV)

3. **Set up stereo cameras:**
   - Position two cameras with known baseline distance
   - Example for 0.1m baseline:
     - Left camera: Position (0, 0, 0)
     - Right camera: Position (0.1, 0, 0)
   - Both cameras should have identical:
     - Field of view
     - Resolution
     - Rotation

4. **Verify DetectionResultsReceiver is configured:**
   - Should be receiving on port `5007` (for mono)
   - Will also receive stereo results on same port
   - Automatically detects and uses `world_position` field

## Usage Example

### Python
```bash
# Terminal 1: Start stereo detector
cd ACRLPython
python LLMcommunication/RunStereoDetector.py --baseline 0.1 --fov 60
```

### Unity
```csharp
// Example: Robot navigation with stereo depth
using LLMCommunication;

public class RobotNavigationController : MonoBehaviour
{
    void Start()
    {
        // Subscribe to detection results
        DetectionResultsReceiver.Instance.OnDetectionWithWorldReceived += OnDetectionReceived;
    }

    void OnDetectionReceived(DetectionResultWithWorld result)
    {
        foreach (var cube in result.CubesWithWorldCoords)
        {
            if (cube.HasWorldPosition)
            {
                Vector3 targetPos = cube.WorldPosition;

                // Check if position came from stereo depth
                if (result.OriginalResult.metadata?.detection_mode == "stereo_3d")
                {
                    Debug.Log($"Using stereo depth: {cube.OriginalDetection.color} cube at {targetPos}");
                    // Navigate robot to exact 3D position
                    RobotManager.Instance.SetTarget("Robot1", targetPos);
                }
            }
        }
    }

    void Update()
    {
        // Send stereo pair on key press
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Camera leftCam = Camera.main;
            Camera rightCam = GameObject.Find("RightCamera").GetComponent<Camera>();
            StereoImageSender.Instance.CaptureAndSendStereoPair(leftCam, rightCam, "detect cubes");
        }
    }
}
```

## Configuration

### Camera Calibration

**Baseline Distance:**
- Measure the physical distance between camera centers
- Typical values: 0.05m - 0.15m
- Larger baseline = better depth accuracy at distance
- Smaller baseline = better accuracy at close range

**Field of View:**
- Must match both cameras
- Unity: Camera component → Field of View
- Python: Pass via `--fov` argument

**Validation:**
```python
# Test depth estimation with known object
python LLMcommunication/DepthEstimator.py \
    --left test_left.png \
    --right test_right.png \
    --x 320 --y 240 \
    --baseline 0.1 --fov 60
```

### Performance Tuning

**Stereo Matching Parameters** (in `StereoImageReconstruction/config.py`):
- `window_size` - Matching window size (default: 2)
- `min_disparity` - Minimum disparity (default: 0)
- `max_disparity` - Maximum disparity (auto-estimated if None)
- `uniqueness_ratio` - Uniqueness constraint (default: 40)

**Processing Interval:**
- Stereo check interval: `0.5s` (configurable in `config.py`)
- Balance between responsiveness and CPU usage

## Troubleshooting

### "No valid disparity values found"
- Cameras too far apart or too close
- Insufficient texture in scene
- Check stereo calibration

### "Stereo image size mismatch"
- Ensure both cameras have identical resolution
- Check Camera Controller resolution settings

### "StereoImageSender not available or not connected"
- Verify Python `RunStereoDetector.py` is running
- Check port 5009 is not in use
- Verify firewall settings

### Depth estimates seem incorrect
- Verify baseline distance is accurate
- Check camera FOV matches between Unity and Python
- Ensure cameras are properly aligned (parallel)

## Benefits Over Raycasting

✅ **Accurate depth without scene geometry** - Works even if Unity colliders are imperfect
✅ **Real-world coordinates** - Direct metric measurements, not dependent on simulation accuracy
✅ **Transfer to real robots** - Same stereo vision algorithms work on physical robots
✅ **Training data quality** - Logs contain true 3D positions for better robot learning
✅ **Robust to complex scenes** - Works where raycasting might hit wrong objects

## File Structure

```
ACRLPython/
├── LLMcommunication/
│   ├── DepthEstimator.py          # NEW - Stereo depth estimation
│   ├── ObjectDetector.py          # UPDATED - Stereo detection support
│   ├── StereoDetectionServer.py  # NEW - Stereo image server
│   ├── RunStereoDetector.py      # NEW - Main orchestrator
│   ├── core/
│   │   └── UnityProtocol.py      # UPDATED - Stereo protocol docs
│   └── config.py                  # UPDATED - Stereo ports/params
└── StereoImageReconstruction/     # Existing stereo reconstruction library
    ├── Reconstruct.py
    ├── FeatureMatching.py
    └── config.py

ACRLUnity/Assets/Scripts/
├── SimulationScripts/
│   ├── StereoImageSender.cs           # NEW - Send stereo pairs
│   ├── CameraController.cs            # UPDATED - Stereo mode
│   └── DetectionResultsReceiver.cs    # UPDATED - Use world_position
├── LLMCommunication/
│   └── DetectionDataModels.cs         # UPDATED - WorldPosition class
└── Logging/
    └── DataModels.cs                   # UPDATED - Stereo logging fields
```

## Next Steps

1. ✅ Integrate stereo depth estimation
2. ⏭️ Calibrate your stereo camera setup
3. ⏭️ Test with known object positions
4. ⏭️ Train robot navigation with accurate 3D targets
5. ⏭️ Extend to real AR4 robot hardware

## References

- [Semi-Global Block Matching (SGBM)](https://core.ac.uk/download/pdf/11134866.pdf)
- [OpenCV Stereo Vision](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- Unity ML-Agents Documentation
