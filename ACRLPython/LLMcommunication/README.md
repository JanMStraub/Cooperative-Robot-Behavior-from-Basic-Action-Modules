# LLM Vision Integration for Unity Robots

Real-time vision analysis for Unity robot cameras using either **Claude API** (cloud-based) or **Ollama** (local LLM). Stream camera images from Unity via TCP and send them to an LLM for instant AI-powered analysis.

## Architecture

The system supports two LLM backends:

### Option 1: Claude API (Cloud)
1. **Unity (C#)** - `CameraController` + `ImageSender` capture and stream images via TCP
2. **StreamingServer.py** - Receives images from Unity and stores them in memory
3. **SendScreenshots.py** - Fetches images from StreamingServer and sends to Claude API

```
Unity Camera → ImageSender (TCP) → StreamingServer → SendScreenshots → Claude API
```

### Option 2: Ollama (Local)
1. **Unity (C#)** - `CameraController` + `ImageSender` capture and stream images via TCP with prompts
2. **RunAnalyzer.py** - Combined server + analyzer that processes images automatically when prompts are attached

```
Unity Camera → ImageSender (TCP with prompt) → RunAnalyzer → Ollama (local)
```

## Setup

### Choose Your Backend

#### For Claude API (Cloud-based, High Quality)

**1. Install Python Dependencies**
```bash
cd ACRLPython/LLMcommunication
pip install anthropic opencv-python numpy python-dotenv
```

**2. Configure API Key**
1. Get your API key from [Anthropic Console](https://console.anthropic.com/settings/keys)
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-...
   ```

**Security Note:** The `.env` file is gitignored. Never commit your API key to version control.

#### For Ollama (Local, Free, Private)

**1. Install Ollama**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download
```

**2. Install Python Dependencies**
```bash
pip install ollama opencv-python numpy
```

**3. Pull a Vision Model**
```bash
# Start Ollama server
ollama serve

# In another terminal, pull a vision model
ollama pull llava          # 7B model (recommended for testing)
ollama pull llava:13b      # 13B model (better quality)
ollama pull llama3.2-vision # Alternative vision model
```

### 3. Unity Setup

Add to your Unity scene:
1. **ImageSender GameObject** - Connects to StreamingServer (port 5005)
2. **CameraController component** - Attached to robot cameras
3. Enable `_usePythonServer` in CameraController inspector

## Usage

### Using Ollama (Local) - RECOMMENDED for Development

**1. Start Ollama** (if not already running)
```bash
ollama serve
```

**2. Start RunAnalyzer** (combines server + analyzer)
```bash
python RunAnalyzer.py
```

**3. Run Unity**
- Unity will automatically connect and stream camera images to port 5005
- When Unity sends an image with a prompt attached, RunAnalyzer will automatically:
  - Receive the image
  - Send it to Ollama with the prompt
  - Display and save the response

**Examples:**
```bash
# Use default model (llava)
python RunAnalyzer.py

# Use a specific model
python RunAnalyzer.py --model llava:13b

# Monitor only specific cameras
python RunAnalyzer.py --camera AR4Left AR4Right

# Adjust check interval and age window
python RunAnalyzer.py --interval 1.0 --min-age 0.5 --max-age 30.0
```

### Using Claude API (Cloud) - For Production Quality

**Terminal 1: Start StreamingServer**
```bash
python StreamingServer.py
```

**Terminal 2: Run Unity**
- Unity will automatically connect and stream camera images to port 5005
- CameraController captures images on demand or via script

**Terminal 3: Send to Claude API**
```bash
python SendScreenshots.py --camera AR4Left --prompt "Describe what you see"
```

### Basic Commands

List available cameras:
```bash
python SendScreenshots.py --list-cameras
```

Analyze single camera:
```bash
python SendScreenshots.py --camera AR4Left --prompt "Describe the scene"
```

Compare multiple cameras:
```bash
python SendScreenshots.py --camera AR4Left AR4Right --prompt "Compare both perspectives"
```

Analyze all cameras:
```bash
python SendScreenshots.py --all-cameras --prompt "Provide a complete scene analysis"
```

### Model Selection

Use different Claude models (default: claude-3-5-haiku-20241022):

```bash
# Fast and cheap - good for basic analysis
python SendScreenshots.py --model claude-3-5-haiku-20241022 --camera AR4Left --prompt "Describe"

# Most capable - best for complex analysis
python SendScreenshots.py --model claude-3-5-sonnet-20241022 --camera AR4Left --prompt "Detailed analysis"

# Maximum capability - deepest understanding
python SendScreenshots.py --model claude-3-opus-20240229 --camera AR4Left --prompt "Expert analysis"
```

### Output Options

Save response to specific location:
```bash
python SendScreenshots.py --camera AR4Left --prompt "Analyze" --output results
```

This creates:
- `results.json` - Full response with metadata
- `results.txt` - Just the text response

Don't save to files (only print to console):
```bash
python SendScreenshots.py --camera AR4Left --prompt "Analyze" --no-save
```

Quiet mode (minimal output):
```bash
python SendScreenshots.py --camera AR4Left --prompt "Analyze" --quiet
```

## Integration with Unity

### Triggering from C# Scripts

#### Manual Screenshot Analysis

```csharp
using UnityEngine;

public class VisionAnalyzer : MonoBehaviour
{
    [SerializeField] private CameraController _cameraController;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.V))
        {
            // Capture and send image
            _cameraController.CaptureAndSave();
        }
    }
}
```

#### Automated Analysis via PythonCaller

```csharp
using UnityEngine;
using SimulationScripts;

public class ClaudeVisionAnalyzer : MonoBehaviour
{
    void AnalyzeCurrentView()
    {
        if (PythonCaller.Instance != null && PythonCaller.Instance.IsActive())
        {
            string scriptPath = "ACRLPython/LLMcommunication/SendScreenshots.py";
            string args = "--camera AR4Left --prompt \"Describe what the robot sees\" --no-save";

            int processId = PythonCaller.Instance.ExecuteAsync(
                scriptPath,
                args,
                (result) => {
                    if (result.Success)
                    {
                        Debug.Log($"Claude analysis: {result.Output}");
                        // Parse and use result...
                    }
                    else
                    {
                        Debug.LogError($"Analysis failed: {result.Error}");
                    }
                },
                timeoutSeconds: 60
            );
        }
    }
}
```

#### Periodic Vision Checks

```csharp
public class PeriodicVisionCheck : MonoBehaviour
{
    [SerializeField] private float _checkInterval = 5.0f;
    private float _timer = 0f;

    void Update()
    {
        _timer += Time.deltaTime;
        if (_timer >= _checkInterval)
        {
            _timer = 0f;
            RequestVisionAnalysis();
        }
    }

    void RequestVisionAnalysis()
    {
        string args = "--camera AR4Left --prompt \"List all visible objects\" --quiet";
        PythonCaller.Instance.ExecuteAsync(
            "ACRLPython/LLMcommunication/SendScreenshots.py",
            args,
            OnAnalysisComplete
        );
    }

    void OnAnalysisComplete(PythonResult result)
    {
        if (result.Success)
        {
            // Process Claude's response
            Debug.Log($"Objects detected: {result.Output}");
        }
    }
}
```

## StreamingServer API

The `ImageServer` singleton provides programmatic access to camera images:

```python
from StreamingServer import ImageServer

# Get server instance
server = ImageServer.get_instance()

# List all active cameras
camera_ids = server.get_all_camera_ids()
print(f"Active cameras: {camera_ids}")

# Get image from specific camera
image = server.get_camera_image("AR4Left")
if image is not None:
    print(f"Image shape: {image.shape}")  # numpy array (H, W, C)

# Check image freshness
age = server.get_camera_age("AR4Left")
print(f"Image age: {age:.1f} seconds")
```

## Command-Line Reference

### Input Options

| Option | Short | Description |
|--------|-------|-------------|
| `--camera` | `-c` | Specific camera ID(s) to use (e.g., AR4Left AR4Right) |
| `--all-cameras` | `-a` | Use all available cameras |
| `--list-cameras` | `-l` | List available cameras and exit |

### API Options

| Option | Short | Description |
|--------|-------|-------------|
| `--prompt` | `-p` | **(Required)** Question/instruction for Claude |
| `--model` | `-m` | Claude model (default: claude-3-5-haiku-20241022) |
| `--max-tokens` | | Maximum response tokens (default: 4096) |
| `--temperature` | | Sampling temperature 0.0-1.0 (default: 1.0) |
| `--api-key` | | API key (overrides ANTHROPIC_API_KEY env var) |

### Output Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output path for response (without extension) |
| `--no-save` | | Don't save response to files |
| `--quiet` | `-q` | Minimize output (only show Claude's response) |

## Pricing

Approximate costs per API call (as of January 2025):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3.5 Haiku | $0.80 | $4.00 |
| Claude 3 Opus | $15.00 | $75.00 |

**Note:** Vision requests (images) consume significant input tokens. A typical 1000x1000 screenshot is approximately 1,500-2,000 tokens.

The script displays estimated cost after each request.

## Use Cases

### 1. Real-Time Object Detection
```bash
python SendScreenshots.py \
  --camera AR4Left \
  --prompt "List all objects visible in the scene with their approximate positions. Format as JSON: {\"objects\": [{\"name\": \"...\", \"position\": \"...\"}]}"
```

### 2. Gripper Pose Verification
```bash
python SendScreenshots.py \
  --camera AR4Left \
  --prompt "Is the gripper properly positioned to grasp the target object? Provide yes/no and confidence level."
```

### 3. Multi-Robot Coordination
```bash
python SendScreenshots.py \
  --camera AR4Left AR4Right \
  --prompt "Analyze both robot perspectives. Are they positioned optimally for collaborative manipulation?"
```

### 4. Quality Assurance
```bash
python SendScreenshots.py \
  --all-cameras \
  --prompt "Check for any collisions, unusual poses, or workspace anomalies. Report any issues found."
```

### 5. Scene Understanding
```bash
python SendScreenshots.py \
  --camera AR4Left \
  --prompt "Describe the complete scene including: robot pose, target objects, workspace boundaries, and any obstacles."
```

### 6. Training Feedback
```bash
python SendScreenshots.py \
  --camera AR4Left \
  --prompt "Evaluate this robot configuration. Suggest improvements for better task performance."
```

## Advanced Usage

### Custom Python Integration

Access images directly from your Python code:

```python
from StreamingServer import ImageServer
import cv2

# Get latest image
server = ImageServer.get_instance()
image = server.get_camera_image("AR4Left")

if image is not None:
    # Process image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save processed image
    cv2.imwrite("processed.png", gray)

    # Check freshness
    age = server.get_camera_age("AR4Left")
    print(f"Image is {age:.1f} seconds old")
```

### Batch Processing

Analyze multiple cameras in sequence:

```bash
for camera in AR4Left AR4Right MainCamera; do
    python SendScreenshots.py \
      --camera $camera \
      --prompt "Describe this view" \
      --output "analysis_${camera}" \
      --quiet
done
```

### Integration with ML Training

Monitor training episodes:

```python
import time
from StreamingServer import ImageServer
from SendScreenshots import ScreenshotSender

sender = ScreenshotSender()
server = ImageServer.get_instance()

while training:
    # Get current view
    image = server.get_camera_image("AR4Left")

    # Analyze every N episodes
    if episode % 100 == 0:
        result = sender.send_images(
            images=[image],
            camera_ids=["AR4Left"],
            prompt="Evaluate the robot's current strategy. Is it learning effectively?"
        )

        log_analysis(result["response"])

    time.sleep(1.0)
```

## Troubleshooting

### StreamingServer Not Receiving Images

1. Check Unity console for ImageSender connection logs
2. Verify server is running: `python StreamingServer.py`
3. Check firewall allows port 5005
4. Confirm ImageSender is enabled in Unity scene

### "No cameras available" Error

1. Verify Unity is running and ImageSender is connected
2. Check StreamingServer logs for incoming images
3. Use `--list-cameras` to see what's available
4. Ensure CameraController has `_usePythonServer` enabled

### Import Errors

If you see "anthropic package not found":
```bash
pip install anthropic opencv-python numpy python-dotenv
```

### API Key Errors

If you see "ANTHROPIC_API_KEY not found":
1. Verify `.env` file exists in ACRLPython/LLMcommunication/
2. Check that it contains `ANTHROPIC_API_KEY=sk-ant-...`
3. Or set environment variable: `export ANTHROPIC_API_KEY=your_key_here`

### Rate Limits

Anthropic API has rate limits. If you hit them:
- Add delays between requests
- Use `--model claude-3-5-haiku-20241022` (higher rate limits)
- Upgrade your API tier at console.anthropic.com

### Performance Issues

If image transmission is slow:
- Reduce image resolution in CameraController inspector
- Use JPEG format instead of PNG (`_imageFormat = ImageFormat.JPG`)
- Lower JPEG quality (default: 85)

## System Requirements

- **Python**: 3.8+
- **Unity**: 6000.2.5f1
- **Network**: Port 5005 available
- **Memory**: ~100MB per active camera
- **Internet**: Required for Claude API access

## Key Differences: Ollama vs Claude API

| Feature | Ollama (Local) | Claude API (Cloud) |
|---------|---------------|-------------------|
| **Cost** | Free | Pay per request (~$0.001-0.01/image) |
| **Privacy** | Fully local, no data sent to cloud | Data sent to Anthropic servers |
| **Speed** | Fast (local inference) | Depends on network, typically slower |
| **Quality** | Good (7B-90B models) | Excellent (state-of-the-art) |
| **Setup** | Requires local GPU/CPU resources | Just API key needed |
| **Workflow** | Automatic (prompt embedded in stream) | Manual (call script per request) |
| **Best For** | Development, testing, private data | Production, highest quality analysis |

## Files

| File | Purpose |
|------|---------|
| `StreamingServer.py` | TCP server receiving images from Unity |
| `RunAnalyzer.py` | **Combined server + Ollama analyzer (recommended)** |
| `AnalyzeImage.py` | Ollama vision client (standalone) |
| `SendScreenshots.py` | Claude API client for vision analysis |
| `requirements.txt` | Python dependencies |
| `.env` | API key configuration (gitignored) |
| `README.md` | This file |

## Related Unity Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `ImageSender.cs` | `ACRLUnity/Assets/Scripts/LLMcommunication/` | Sends images to StreamingServer |
| `CameraController.cs` | `ACRLUnity/Assets/Scripts/SimulationScripts/` | Captures robot camera images |
| `PythonCaller.cs` | `ACRLUnity/Assets/Scripts/SimulationScripts/` | Executes Python scripts from Unity |

## License

Part of the ACRL (Auto-Cooperative Robot Learning) project. See project root for license information.
