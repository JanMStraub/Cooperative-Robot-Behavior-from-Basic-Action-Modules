# LLM Integration Guide

## Overview

This guide explains how to integrate LLM (Large Language Model) vision analysis into your Unity simulation. The system allows Unity cameras to send images to an Ollama LLM for analysis, and receive text responses back in real-time.

## Architecture

```
Unity Simulation
    ↓ (sends images via port 5005)
Python: StreamingServer
    ↓ (stores images)
Python: AnalyzeImage
    ↓ (processes with Ollama)
Python: ResultsServer
    ↓ (sends results via port 5006)
Unity: LLMResultsReceiver
    ↓ (triggers OnResultReceived event)
Your Custom Script
```

## Components

### Python Side

1. **StreamingServer** (port 5005)
   - Receives camera images from Unity
   - Stores images by camera ID
   - Handles multiple camera streams

2. **AnalyzeImage**
   - Monitors for new images with prompts
   - Sends images to Ollama for analysis
   - Generates text responses

3. **ResultsServer** (port 5006)
   - Sends analysis results back to Unity
   - Supports multiple Unity clients
   - Queues results if Unity not connected

4. **RunAnalyzer**
   - Combines all three servers in one process
   - Automatically started by PythonServerManager

### Unity Side

1. **CameraController** (existing)
   - Captures camera screenshots
   - Sends to Python StreamingServer (port 5005)
   - Includes optional prompt with images

2. **LLMResultsReceiver** (NEW)
   - Connects to Python ResultsServer (port 5006)
   - Receives analysis results
   - Fires `OnResultReceived` event

3. **PythonServerManager** (existing)
   - Auto-starts RunAnalyzer.py on Unity Play
   - Manages Python process lifecycle

## Setup Instructions

### Step 1: Install Python Dependencies

```bash
cd /Users/jan/Code/MS
source ACRLPython/acrl/bin/activate
pip install ollama opencv-python numpy
```

### Step 2: Install and Start Ollama

```bash
# Install Ollama (if not installed)
# Download from: https://ollama.com/download

# Start Ollama service
ollama serve

# Pull a vision model (in another terminal)
ollama pull llava
# or
ollama pull gemma3
```

### Step 3: Add LLMResultsReceiver to Unity Scene

1. Open your Unity scene (e.g., `1xAR4Scene.unity`)
2. In Hierarchy, right-click → Create Empty
3. Name it "LLMResultsReceiver"
4. Select the GameObject
5. Click "Add Component" → Search for "LLM Results Receiver"
6. Configure in Inspector:
   - **Server Host**: `127.0.0.1`
   - **Server Port**: `5006`
   - **Auto Connect**: ✓ (checked)
   - **Retry On Failure**: ✓ (checked)
   - **Retry Delay**: `5`
   - **Log Results**: ✓ (checked for debugging)

### Step 4: Update PythonServerManager Arguments

1. Select GameObject with PythonServerManager
2. Find the "RunAnalyzer" server configuration
3. Update **Arguments** to include results port:
   ```
   --model gemma3 --server-host 127.0.0.1 --server-port 5005 --results-port 5006
   ```

### Step 5: Create a Script to Handle Results

Create a new C# script to handle incoming LLM results:

```csharp
using UnityEngine;

public class LLMResponseHandler : MonoBehaviour
{
    void Start()
    {
        // Subscribe to results
        if (LLMResultsReceiver.Instance != null)
        {
            LLMResultsReceiver.Instance.OnResultReceived += HandleLLMResult;
        }
    }

    void OnDestroy()
    {
        // Unsubscribe
        if (LLMResultsReceiver.Instance != null)
        {
            LLMResultsReceiver.Instance.OnResultReceived -= HandleLLMResult;
        }
    }

    private void HandleLLMResult(LLMResult result)
    {
        Debug.Log($"LLM saw: {result.response}");

        // Do something with the result
        // Examples:
        // - Update UI text
        // - Trigger robot actions
        // - Log to file
        // - Send to other systems

        // Access result data:
        string cameraId = result.camera_id;
        string response = result.response;
        string model = result.metadata.model;
        float processingTime = result.metadata.duration_seconds;
        string prompt = result.metadata.prompt;
    }
}
```

## Usage Examples

### Example 1: Ask LLM to Identify Objects

From your camera controller script:

```csharp
public class MyCameraController : MonoBehaviour
{
    private CameraController _cameraController;

    void Start()
    {
        _cameraController = GetComponent<CameraController>();
    }

    public void AnalyzeCurrentView()
    {
        // Capture and send with prompt
        _cameraController.CaptureAndSend("What objects do you see in this image?");
    }
}
```

### Example 2: Robot Vision Feedback Loop

```csharp
public class VisionGuidedRobot : MonoBehaviour
{
    private RobotController _robot;
    private CameraController _camera;

    void Start()
    {
        _robot = GetComponent<RobotController>();
        _camera = GetComponentInChildren<CameraController>();

        // Subscribe to LLM results
        LLMResultsReceiver.Instance.OnResultReceived += OnVisionResult;

        // Start vision loop
        StartCoroutine(VisionLoop());
    }

    IEnumerator VisionLoop()
    {
        while (true)
        {
            // Ask LLM to analyze scene
            _camera.CaptureAndSend("Describe the closest object to the robot gripper.");

            // Wait for response (will come via OnVisionResult)
            yield return new WaitForSeconds(5f);
        }
    }

    void OnVisionResult(LLMResult result)
    {
        if (result.camera_id == _camera.CameraId)
        {
            Debug.Log($"Robot sees: {result.response}");

            // Example: Parse response and act on it
            if (result.response.Contains("cube"))
            {
                // Move towards cube
                Debug.Log("Moving towards cube...");
            }
        }
    }
}
```

### Example 3: Multi-Camera Analysis

```csharp
public class MultiCameraAnalyzer : MonoBehaviour
{
    private Dictionary<string, LLMResult> _latestResults = new Dictionary<string, LLMResult>();

    void Start()
    {
        LLMResultsReceiver.Instance.OnResultReceived += OnResult;
    }

    void OnResult(LLMResult result)
    {
        // Store latest result per camera
        _latestResults[result.camera_id] = result;

        Debug.Log($"[{result.camera_id}] LLM: {result.response}");
    }

    public string GetLatestResponse(string cameraId)
    {
        if (_latestResults.TryGetValue(cameraId, out LLMResult result))
        {
            return result.response;
        }
        return null;
    }
}
```

## Testing

### Test 1: Verify Python Server Starts

1. Press Play in Unity
2. Check Console for:
   ```
   [PythonServerManager] Started server 'RunAnalyzer' (PID: 1)
   [PythonCaller] Started Python process 1: RunAnalyzer.py...
   ```

### Test 2: Verify LLM Connection

1. Check Console for:
   ```
   [LLMResultsReceiver] Connecting to ResultsServer at 127.0.0.1:5006...
   [LLMResultsReceiver] ✓ Connected to ResultsServer
   ```

### Test 3: Send Test Image

1. Trigger a camera capture with prompt
2. Check Python console output (in terminal if running manually):
   ```
   🔍 PROCESSING NEW IMAGE FROM: AR4Left
   📝 Prompt: 'What do you see?'
   🤖 OLLAMA RESPONSE FOR AR4Left
   ================================================================================
   I see a robotic arm...
   ================================================================================
   📤 Sent result to Unity for camera: AR4Left
   ```

3. Check Unity Console:
   ```
   [LLMResultsReceiver] 📥 LLM Result for AR4Left:
     Response: I see a robotic arm...
     Model: gemma3
     Duration: 1.23s
   ```

## Data Flow Diagram

```
┌─────────────────┐
│  Unity Camera   │
│  (C# Script)    │
└────────┬────────┘
         │ Screenshot + Prompt
         │ TCP → Port 5005
         ↓
┌─────────────────┐
│ StreamingServer │ ← Stores images by camera ID
│  (Python)       │
└────────┬────────┘
         │ Image available
         ↓
┌─────────────────┐
│  AnalyzeImage   │ ← Monitors for new images
│  (Python)       │
└────────┬────────┘
         │ Image + Prompt
         ↓
┌─────────────────┐
│     Ollama      │ ← LLM vision model
│   (llava/gemma) │
└────────┬────────┘
         │ Text response
         ↓
┌─────────────────┐
│  ResultsServer  │ ← Sends results to Unity
│  (Python)       │
└────────┬────────┘
         │ JSON result
         │ TCP → Port 5006
         ↓
┌─────────────────┐
│LLMResultsReceiv │ ← Receives and parses
│     er          │
│  (C# Script)    │
└────────┬────────┘
         │ OnResultReceived event
         ↓
┌─────────────────┐
│  Your Custom    │
│     Script      │
└─────────────────┘
```

## API Reference

### LLMResult Class

```csharp
public class LLMResult
{
    public bool success;           // Whether analysis succeeded
    public string response;        // LLM's text response
    public string camera_id;       // Camera that sent the image
    public string timestamp;       // ISO timestamp
    public LLMMetadata metadata;   // Additional info
}

public class LLMMetadata
{
    public string model;              // Model name (e.g., "gemma3")
    public float duration_seconds;    // Processing time
    public int image_count;           // Number of images analyzed
    public string[] camera_ids;       // Array of camera IDs
    public string prompt;             // User's prompt
    public string full_prompt;        // Full prompt sent to LLM
}
```

### LLMResultsReceiver API

**Methods:**
```csharp
void Connect()                    // Manually connect to server
void Disconnect()                 // Disconnect from server
bool IsConnected()                // Check connection status
string GetConnectionInfo()        // Get "host:port" string
```

**Events:**
```csharp
event Action<LLMResult> OnResultReceived  // Fired when result arrives
```

**Inspector Fields:**
- `Server Host` - ResultsServer host (default: 127.0.0.1)
- `Server Port` - ResultsServer port (default: 5006)
- `Auto Connect` - Connect automatically on Start
- `Retry On Failure` - Reconnect if connection lost
- `Retry Delay` - Seconds between reconnect attempts
- `Log Results` - Log all results to console

## Troubleshooting

### No Results Received

**Problem**: Camera sends images but no results come back.

**Solutions**:
1. Check LLMResultsReceiver is connected:
   - Look for: `[LLMResultsReceiver] ✓ Connected to ResultsServer`
   - If not connected, check port 5006 is not in use

2. Check Python analyzer is processing:
   - Python should log: `🔍 PROCESSING NEW IMAGE FROM: ...`
   - If not, check image has a prompt attached

3. Check Ollama is running:
   ```bash
   ollama list
   ```

4. Check model exists:
   ```bash
   ollama pull gemma3
   ```

### Results Delayed

**Problem**: Long delay between image send and result.

**Reasons**:
1. **LLM processing time** - Vision models take 1-5 seconds
2. **Model not pulled** - First run downloads model (slow)
3. **CPU-only mode** - Install GPU support for faster inference

**Solution**: Use smaller/faster model like `llava:7b` instead of `llava:34b`

### Connection Refused

**Problem**: `[LLMResultsReceiver] Failed to connect to ResultsServer`

**Solutions**:
1. Check RunAnalyzer is running (check PythonServerManager)
2. Check firewall allows port 5006
3. Check `--results-port 5006` in RunAnalyzer arguments
4. Check no other app is using port 5006

### JSON Parse Error

**Problem**: `Failed to parse JSON result`

**Solution**: Python and Unity may have incompatible JSON format. Check the exact error message and the JSON string in logs.

## Performance Tips

1. **Don't analyze every frame** - LLM processing is slow (1-5s per image)
2. **Use debouncing** - Wait for camera to stop moving before analyzing
3. **Batch similar requests** - Process multiple cameras together
4. **Choose appropriate models**:
   - Fast: `llava:7b` (~1-2s)
   - Balanced: `llava:13b` (~2-3s)
   - Accurate: `llava:34b` (~4-5s)
5. **Use clear prompts** - Specific questions get faster, better answers

## Advanced: Custom Result Processing

You can create a specialized result processor:

```csharp
public class ObjectDetectionProcessor : MonoBehaviour
{
    [System.Serializable]
    public class DetectedObject
    {
        public string name;
        public Vector3 position;
        public float confidence;
    }

    public List<DetectedObject> detectedObjects = new List<DetectedObject>();

    void Start()
    {
        LLMResultsReceiver.Instance.OnResultReceived += ProcessDetection;
    }

    void ProcessDetection(LLMResult result)
    {
        // Parse LLM response to extract structured data
        // Example: "I see a red cube at position (1, 0, 0)"

        string response = result.response.ToLower();

        if (response.Contains("cube"))
        {
            detectedObjects.Add(new DetectedObject
            {
                name = "Cube",
                position = ExtractPosition(response),
                confidence = 0.8f
            });
        }

        // Use detected objects for robot control, UI, etc.
    }

    Vector3 ExtractPosition(string text)
    {
        // Implement position extraction from LLM response
        // This is a simplified example
        return Vector3.zero;
    }
}
```

## File Locations

- **Python Scripts**: `/Users/jan/Code/MS/ACRLPython/LLMcommunication/`
  - `StreamingServer.py` - Image receiver
  - `AnalyzeImage.py` - LLM processor
  - `ResultsServer.py` - Result sender
  - `RunAnalyzer.py` - Combined server

- **Unity Scripts**: `/Users/jan/Code/MS/ACRLUnity/Assets/Scripts/SimulationScripts/`
  - `LLMResultsReceiver.cs` - Result receiver
  - `CameraController.cs` - Image sender (existing)
  - `PythonServerManager.cs` - Auto-start manager

## Related Documentation

- [PythonServerManager.md](PythonServerManager.md) - Python server auto-start
- [CLAUDE.md](../../CLAUDE.md) - Project architecture overview

## Version History

- **v1.0** (2025-10-20): Initial LLM integration
  - ResultsServer for sending results to Unity
  - LLMResultsReceiver for receiving results
  - Integration with RunAnalyzer.py
  - Event-based result handling
