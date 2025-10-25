# Object Detection System for Unity Cube Detection

## Overview

This system detects red and blue cubes in camera images using color-based HSV segmentation and returns both pixel coordinates and Unity world coordinates via raycasting.

## Architecture

### Python Side (Detection Processing)
- **ObjectDetector.py** - HSV color segmentation for red/blue cube detection
- **DetectionServer.py** - TCP server broadcasting results to Unity (port 5007)
- **RunDetector.py** - Main orchestrator monitoring images and running detection

### Unity Side (Result Handling)
- **DetectionDataModels.cs** - Data structures for detection results
- **DetectionResultsReceiver.cs** - TCP client receiving detection results
- **CubeDetectionVisualizer.cs** - Optional visualization component

## Setup Instructions

### 1. Python Environment Setup

The system uses your existing `acrl` virtual environment:

```bash
cd /Users/jan/Code/MS/ACRLPython

# Activate virtual environment (if not already active)
source acrl/bin/activate

# Verify dependencies (should already be installed)
pip list | grep -E "opencv-python|numpy"
```

### 2. Start the Detection System

**Option A: Standalone Detection Server**
```bash
cd /Users/jan/Code/MS/ACRLPython
python LLMcommunication/RunDetector.py
```

**Option B: With Custom Settings**
```bash
# Monitor specific camera only
python LLMcommunication/RunDetector.py --camera AR4Left

# Faster detection interval (0.5s instead of 1s)
python LLMcommunication/RunDetector.py --interval 0.5

# Enable debug mode (saves annotated images)
python LLMcommunication/RunDetector.py --debug
```

**Option C: Run Alongside LLM Analyzer**
```bash
# Terminal 1: LLM Analyzer
python LLMcommunication/RunAnalyzer.py --model gemma3

# Terminal 2: Object Detector
python LLMcommunication/RunDetector.py
```

### 3. Unity Scene Setup

#### Add Required Components to Scene

1. **Add DetectionResultsReceiver GameObject**
   ```
   - Create empty GameObject: "DetectionResultsReceiver"
   - Add component: DetectionResultsReceiver.cs
   - Configure in Inspector:
     - Server Host: 127.0.0.1
     - Server Port: 5007
     - Camera Mappings: Add your camera(s)
       - Camera ID: "AR4Left" (or your camera identifier)
       - Camera: Drag Camera component reference
   ```

2. **Add CubeDetectionVisualizer GameObject** (Optional)
   ```
   - Create empty GameObject: "CubeDetectionVisualizer"
   - Add component: CubeDetectionVisualizer.cs
   - Configure visualization settings as desired
   ```

   OR use menu: `GameObject > ACRL > Cube Detection Visualizer`

#### Register Cameras Programmatically

Alternative to Inspector setup:

```csharp
void Start() {
    Camera myCamera = GetComponent<Camera>();
    DetectionResultsReceiver.Instance.RegisterCamera("AR4Left", myCamera);
}
```

### 4. Send Images for Detection

Use the existing CameraController to send images:

```csharp
// In your code
CameraController cameraController = GetComponent<CameraController>();
cameraController.CaptureAndSend(); // Sends to StreamingServer (port 5005)

// Detection happens automatically via RunDetector.py monitoring ImageStorage
```

## Configuration

### Python Configuration

Edit `config.py` to adjust detection parameters:

```python
# Detection Server Port
DETECTION_SERVER_PORT = 5007

# Color Ranges (HSV)
RED_HSV_LOWER_1 = (0, 100, 100)      # Red lower bound 1
RED_HSV_UPPER_1 = (10, 255, 255)     # Red upper bound 1
RED_HSV_LOWER_2 = (170, 100, 100)    # Red lower bound 2 (wraps)
RED_HSV_UPPER_2 = (180, 255, 255)    # Red upper bound 2

BLUE_HSV_LOWER = (100, 100, 100)     # Blue lower bound
BLUE_HSV_UPPER = (130, 255, 255)     # Blue upper bound

# Detection Filters
MIN_CUBE_AREA_PX = 100        # Minimum bounding box area
MAX_CUBE_AREA_PX = 100000     # Maximum bounding box area
MIN_ASPECT_RATIO = 0.5        # Minimum width/height ratio
MAX_ASPECT_RATIO = 2.0        # Maximum width/height ratio
MIN_CONFIDENCE = 0.6          # Minimum detection confidence

# Processing
DETECTION_CHECK_INTERVAL = 1.0  # Check for new images every N seconds
ENABLE_DEBUG_IMAGES = False     # Save annotated images
```

### Tuning Color Detection

If cubes aren't being detected properly:

1. **Test color ranges**:
   ```bash
   # Take a screenshot of your scene
   # Test the detector on a saved image
   python LLMcommunication/ObjectDetector.py path/to/image.jpg
   ```

2. **Enable debug mode** to see what's being detected:
   ```bash
   python LLMcommunication/RunDetector.py --debug
   # Check debug images in: ACRLPython/LLMCommunication/debug_detections/
   ```

3. **Adjust HSV ranges** in `config.py`:
   - H (Hue): 0-180 in OpenCV (color)
   - S (Saturation): 0-255 (color intensity)
   - V (Value): 0-255 (brightness)

4. **Adjust filters**:
   - Increase `MIN_CUBE_AREA_PX` if detecting too many small objects
   - Decrease `MIN_CONFIDENCE` if missing valid cubes
   - Adjust `MIN/MAX_ASPECT_RATIO` for cubes at extreme angles

## Usage Examples

### Example 1: Subscribe to Detection Events

```csharp
using LLMCommunication;

public class CubeTracker : MonoBehaviour {
    void Start() {
        // Subscribe to detection events
        DetectionResultsReceiver.Instance.OnDetectionWithWorldReceived += HandleDetection;
    }

    void HandleDetection(DetectionResultWithWorld result) {
        Debug.Log($"Received {result.CubesWithWorldCoords.Length} cubes");

        foreach (var cube in result.CubesWithWorldCoords) {
            if (cube.HasWorldPosition) {
                Debug.Log($"{cube.OriginalDetection.color} cube at {cube.WorldPosition}");

                // Do something with the position
                // e.g., move robot to cube.WorldPosition
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

### Example 2: Get Pixel and World Coordinates

```csharp
void HandleDetection(DetectionResultWithWorld result) {
    foreach (var cube in result.CubesWithWorldCoords) {
        // Pixel coordinates
        Vector2Int pixelPos = cube.OriginalDetection.center_px;
        BoundingBoxPx bbox = cube.OriginalDetection.bbox_px;

        // World coordinates (if raycast hit)
        if (cube.HasWorldPosition) {
            Vector3 worldPos = cube.WorldPosition;
            GameObject hitObject = cube.HitObject;

            Debug.Log($"Cube at pixel ({pixelPos.x}, {pixelPos.y}) is at world {worldPos}");
            Debug.Log($"Hit object: {hitObject?.name}");
        }

        // Confidence and color
        float confidence = cube.OriginalDetection.confidence;
        string color = cube.OriginalDetection.color;
    }
}
```

### Example 3: Filter by Color

```csharp
void HandleDetection(DetectionResultWithWorld result) {
    // Get only red cubes
    var redCubes = System.Array.FindAll(
        result.CubesWithWorldCoords,
        cube => cube.OriginalDetection.color == "red" && cube.HasWorldPosition
    );

    Debug.Log($"Found {redCubes.Length} red cubes");

    foreach (var redCube in redCubes) {
        Debug.Log($"Red cube at {redCube.WorldPosition}");
    }
}
```

## Data Flow

1. **Unity** → Image sent via `CameraController.CaptureAndSend()` → `StreamingServer (port 5005)`
2. **Python** → `ImageStorage` stores the image
3. **Python** → `RunDetector.py` detects new image → runs `CubeDetector`
4. **Python** → Detection results → `DetectionServer (port 5007)` → broadcasts to Unity
5. **Unity** → `DetectionResultsReceiver` receives results → converts pixel→world via raycasting
6. **Unity** → Fires `OnDetectionWithWorldReceived` event
7. **Your Code** → Handles detection results

## Result Format

### Python Detection Result (JSON)
```json
{
  "success": true,
  "camera_id": "AR4Left",
  "timestamp": "2025-10-21T12:34:56.789",
  "image_width": 1000,
  "image_height": 1000,
  "detections": [
    {
      "id": 0,
      "color": "red",
      "bbox_px": {"x": 120, "y": 340, "width": 80, "height": 85},
      "center_px": {"x": 160, "y": 382},
      "confidence": 0.92
    },
    {
      "id": 1,
      "color": "blue",
      "bbox_px": {"x": 300, "y": 280, "width": 75, "height": 78},
      "center_px": {"x": 337, "y": 319},
      "confidence": 0.88
    }
  ]
}
```

### Unity Enhanced Result
```csharp
DetectionResultWithWorld {
  OriginalResult: DetectionResult (raw Python data)
  SourceCamera: Camera component
  CubesWithWorldCoords: [
    DetectedCubeWithWorld {
      OriginalDetection: DetectionObject (pixel data)
      WorldPosition: Vector3(0.15, 0.05, 0.45)
      HitObject: GameObject reference
      HasWorldPosition: true
    },
    ...
  ]
}
```

## Troubleshooting

### No Detections Received

1. **Check if DetectionServer is running**:
   ```bash
   lsof -nP -iTCP:5007
   # Should show Python process listening on port 5007
   ```

2. **Check Unity connection**:
   - Inspector → DetectionResultsReceiver → Server Status should show "Connected"

3. **Check if images are being sent**:
   - Verify StreamingServer (port 5005) is running
   - Check CameraController is sending images

4. **Check Python logs**:
   - Should see "🔍 DETECTING CUBES IN IMAGE FROM: ..."
   - Should see detection results printed

### Wrong World Coordinates

1. **Verify camera registration**:
   ```csharp
   Debug.Log($"Camera registered: {DetectionResultsReceiver.Instance != null}");
   ```

2. **Check if raycast is hitting**:
   - Enable CubeDetectionVisualizer to see where raycasts are landing
   - Ensure cubes have colliders

3. **Check coordinate system**:
   - Image Y=0 is top, Unity screen Y=0 is bottom (code handles this)
   - Verify camera FOV matches between Unity and config

### Cubes Not Detected

1. **Enable debug mode**:
   ```bash
   python LLMcommunication/RunDetector.py --debug
   ```
   Check saved images in `debug_detections/` folder

2. **Test on saved image**:
   ```bash
   python LLMcommunication/ObjectDetector.py path/to/screenshot.jpg
   ```

3. **Adjust color ranges** in `config.py`
4. **Adjust size/confidence thresholds** in `config.py`

### Port Already in Use

```bash
# Find process using port 5007
lsof -nP -iTCP:5007

# Kill the process
kill -9 <PID>
```

## Performance Notes

- **Detection Speed**: ~10-50ms per frame (depends on image size and cube count)
- **Network Latency**: ~5-20ms (local TCP connection)
- **Total Latency**: ~50-100ms from image capture to world coordinates
- **Recommended Check Interval**: 0.5-1.0 seconds for real-time operation

## Advanced: Coordinate System Details

### Pixel Coordinates
- Origin: Top-left corner (0, 0)
- X-axis: Right (increases left to right)
- Y-axis: Down (increases top to bottom)

### Unity Viewport Coordinates
- Origin: Bottom-left corner (0, 0)
- X-axis: Right (0 to 1)
- Y-axis: Up (0 to 1)
- **Note**: Y is flipped in conversion (handled automatically)

### Unity World Coordinates
- Depends on camera orientation
- Obtained via `Physics.Raycast()` from camera through pixel position
- Returns actual 3D position of the detected object

## Files Created

### Python (4 files):
1. `ACRLPython/LLMcommunication/ObjectDetector.py` - Core detection logic
2. `ACRLPython/LLMcommunication/DetectionServer.py` - TCP server
3. `ACRLPython/LLMcommunication/RunDetector.py` - Main orchestrator
4. `ACRLPython/LLMcommunication/config.py` - Configuration (modified)

### Unity C# (3 files):
5. `ACRLUnity/Assets/Scripts/LLMCommunication/DetectionDataModels.cs` - Data models
6. `ACRLUnity/Assets/Scripts/SimulationScripts/DetectionResultsReceiver.cs` - TCP client
7. `ACRLUnity/Assets/Scripts/SimulationScripts/CubeDetectionVisualizer.cs` - Visualization

## Integration with Existing Systems

This detection system integrates seamlessly with:
- **StreamingServer** (port 5005) - Shares the ImageStorage singleton
- **LLM Analyzer** (RunAnalyzer.py) - Can run simultaneously
- **CameraController** - Uses existing image capture
- **MainLogger** - Can be extended to log detection results

## Future Enhancements

Possible improvements:
- [ ] Add depth map support for more accurate 3D positions
- [ ] Support for more colors/object types
- [ ] Machine learning-based detection (YOLO)
- [ ] Tracking objects across frames
- [ ] Velocity estimation
- [ ] Integration with robot control system

## License

Part of the ACRL (Auto-Cooperative Robot Learning) project.
