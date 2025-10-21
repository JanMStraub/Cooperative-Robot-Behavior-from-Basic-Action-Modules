# PythonServerManager

## Overview

`PythonServerManager` automatically starts and manages background Python server processes when Unity starts. It provides a simple, extensible way to launch Python scripts (like image processing servers, ML training processes, or API servers) that need to run alongside your Unity simulation.

**Key Features:**
- ✅ Automatic startup on Unity Play
- ✅ Extensible design - add new servers via Inspector (no code changes)
- ✅ Built on `PythonCaller` infrastructure
- ✅ Non-blocking execution (async processes)
- ✅ Auto-cleanup on Unity shutdown
- ✅ Editor controls for manual start/stop
- ✅ Process monitoring and status display

## How It Works

### Architecture

```
Unity Startup
    ↓
SimulationManager.Start()
    ↓
PythonServerManager.Start()
    ↓ (after startup delay)
PythonServerManager.StartAllServers()
    ↓
PythonCaller.ExecuteAsync() ← for each server
    ↓
Python processes run in background
    ↓
Unity Shutdown → Auto cleanup via OnApplicationQuit()
```

### Integration with PythonCaller

`PythonServerManager` doesn't replace `PythonCaller` - it **uses** `PythonCaller` to manage processes:

- **PythonCaller**: Low-level Python process execution (single process control)
- **PythonServerManager**: High-level server orchestration (multiple servers, automatic startup)

All Python execution goes through `PythonCaller.Instance.ExecuteAsync()` with infinite timeout (`timeoutSeconds: -1`), meaning servers run until explicitly stopped.

## Setup Instructions

### 1. Scene Setup

Add the `PythonServerManager` component to your scene (typically on the same GameObject as `SimulationManager`):

1. Open your Unity scene (e.g., `1xAR4Scene.unity`)
2. Select the GameObject with `SimulationManager` attached
3. Click **Add Component** → Search for "Python Server Manager"
4. The component will auto-initialize with no servers configured

### 2. Configure Your First Server

The default setup is for `RunAnalyzer.py` (StreamingServer + Ollama analyzer):

1. Select the GameObject with `PythonServerManager`
2. In the Inspector, expand **Servers** → Set size to `1`
3. Configure **Element 0**:
   - **Server Name**: `RunAnalyzer`
   - **Script Path**: `ACRLPython/LLMcommunication/RunAnalyzer.py`
   - **Arguments**: `--model gemma3 --server-host 127.0.0.1 --server-port 5005`
   - **Auto Start**: ✓ (checked)
   - **Enabled**: ✓ (checked)
   - **Description**: `Combined StreamingServer + Ollama image analyzer`

4. Under **Settings**:
   - **Startup Delay**: `1.0` (wait 1 second after Unity starts before launching servers)

### 3. Verify Python Environment

Make sure your Python environment is configured in `PythonCaller`:

1. Select the GameObject with `PythonCaller` component
2. Check **Python Environment** settings:
   - **Base Path**: Should auto-detect (typically parent of Unity project)
   - **Python Env Path**: `roboscan/bin/python` (macOS/Linux) or `roboscan/Scripts/python.exe` (Windows)

### 4. Test It

1. Press **Play** in Unity
2. Check the Console - you should see:
   ```
   [PythonServerManager] PythonServerManager initialized with 1 server(s)
   [PythonServerManager] Starting all auto-start servers...
   [PythonCaller] Started Python process 1: RunAnalyzer.py --model gemma3...
   [PythonServerManager] Started server 'RunAnalyzer' (PID: 1)
   ```

3. The Python server is now running in the background!

## Adding New Python Scripts

### Example: Adding a Custom Data Logger

Let's say you want to add a Python script that logs robot telemetry data.

#### Step 1: Create Your Python Script

Create `ACRLPython/telemetry/DataLogger.py`:

```python
#!/usr/bin/env python3
import socket
import time

def main():
    print("Telemetry logger started")
    while True:
        # Your logging logic here
        time.sleep(1)

if __name__ == "__main__":
    main()
```

#### Step 2: Add to PythonServerManager

1. Open Unity and select the GameObject with `PythonServerManager`
2. In the Inspector, expand **Servers**
3. Increase the **Size** from `1` to `2`
4. Configure the new **Element 1**:
   - **Server Name**: `TelemetryLogger`
   - **Script Path**: `ACRLPython/telemetry/DataLogger.py`
   - **Arguments**: `` (empty if no arguments needed)
   - **Auto Start**: ✓ (checked)
   - **Enabled**: ✓ (checked)
   - **Description**: `Logs robot telemetry data to file`

#### Step 3: Done!

Press Play - both servers will start automatically. No code changes required!

### Adding Command-Line Arguments

If your Python script needs arguments:

**Example Arguments**:
```
--port 8080 --log-level DEBUG
--output-dir /path/to/output --interval 0.5
```

Just paste them into the **Arguments** field in the Inspector.

### Disabling a Server Temporarily

To temporarily disable a server without deleting it:

1. Uncheck **Enabled** in the Inspector
2. Or uncheck **Auto Start** to keep it configured but not auto-starting

## Editor Controls

### Custom Inspector

The `PythonServerManager` has a custom inspector with real-time controls:

**Global Controls:**
- **Start All Servers** - Start all enabled servers with autoStart=true
- **Stop All Servers** - Stop all running servers

**Individual Server Controls:**
Each enabled server shows:
- **Status Indicator**: ✓ (green) = running, ○ (gray) = stopped
- **Server Info**: Name and elapsed time (updates in real-time)
- **Start Button**: Manually start this server (disabled if running)
- **Stop Button**: Manually stop this server (disabled if not running)
- **Description**: Shows the server's description below controls

**Note**: Controls only work in Play mode. Servers cannot be started in Edit mode.

## API Reference

### Public Methods

#### `StartAllServers()`
Starts all servers that have `autoStart=true` and `enabled=true`.

```csharp
PythonServerManager.Instance.StartAllServers();
```

#### `StartServer(string serverName)`
Starts a specific server by name. Returns `true` if successful.

```csharp
bool started = PythonServerManager.Instance.StartServer("RunAnalyzer");
```

#### `StopServer(string serverName)`
Stops a specific server by name. Returns `true` if successful.

```csharp
bool stopped = PythonServerManager.Instance.StopServer("RunAnalyzer");
```

#### `StopAllServers()`
Stops all running servers. Called automatically on Unity shutdown.

```csharp
PythonServerManager.Instance.StopAllServers();
```

#### `IsServerRunning(string serverName)`
Check if a server is currently running.

```csharp
if (PythonServerManager.Instance.IsServerRunning("RunAnalyzer"))
{
    Debug.Log("Analyzer is running!");
}
```

#### `GetServerProcessInfo(string serverName, out string scriptPath, out float elapsedSeconds)`
Get runtime information about a running server.

```csharp
if (PythonServerManager.Instance.GetServerProcessInfo("RunAnalyzer", out string path, out float elapsed))
{
    Debug.Log($"RunAnalyzer has been running for {elapsed:F1}s");
}
```

#### `GetServerNames()`
Get list of all registered server names.

```csharp
List<string> servers = PythonServerManager.Instance.GetServerNames();
foreach (string name in servers)
{
    Debug.Log($"Registered server: {name}");
}
```

### PythonServerConfig Class

Each server is configured via `PythonServerConfig`:

```csharp
[System.Serializable]
public class PythonServerConfig
{
    public string serverName;      // Unique identifier
    public string scriptPath;      // Relative path from project root
    public string arguments;       // Command-line arguments
    public bool autoStart;         // Start automatically on Unity Play
    public bool enabled;           // Enable/disable this server
    public string description;     // What this server does

    // Runtime state (read-only)
    public int processId;          // PythonCaller process ID (-1 if not running)
    public bool isRunning;         // Current running state
}
```

## Troubleshooting

### Server doesn't start

**Problem**: Server shows as stopped, no process starts.

**Solutions**:
1. Check Console for error messages
2. Verify **Script Path** is correct (relative to project root)
3. Verify Python environment in `PythonCaller` settings
4. Make sure **Enabled** and **Auto Start** are checked
5. Check that Python script is executable: `chmod +x script.py` (macOS/Linux)

### Server starts but immediately stops

**Problem**: Server appears to start but stops immediately.

**Solutions**:
1. Check Console for Python errors
2. Test script manually: `python ACRLPython/LLMcommunication/RunAnalyzer.py`
3. Check script has proper `if __name__ == "__main__":` guard
4. Verify Python dependencies are installed: `pip install -r requirements.txt`
5. Check script doesn't exit immediately (should have a main loop or server)

### PythonCaller not found

**Problem**: Console shows "PythonCaller not found".

**Solution**:
- Make sure `PythonCaller` GameObject exists in scene
- `PythonCaller` must initialize **before** `PythonServerManager`
- Add `PythonCaller` to scene if missing

### Server runs but nothing happens

**Problem**: Server starts successfully but doesn't seem to do anything.

**Solutions**:
1. Check server is actually running: Look for status ✓ in Inspector
2. Verify server is listening on correct port
3. Check firewall isn't blocking connections
4. For `RunAnalyzer`: Make sure Unity `CameraController` is sending images to port 5005
5. Check Python script output in Console (may need to add print statements)

### Multiple servers conflict

**Problem**: Starting multiple servers causes issues.

**Solutions**:
1. Make sure servers use different ports (configure via arguments)
2. Check for duplicate **Server Name** entries
3. Verify servers don't write to same files/resources
4. Check `PythonCaller` max concurrent processes (default: 3)

## Current Setup (Default Configuration)

By default, the project includes one server:

### RunAnalyzer Server

- **What it does**: Runs both `StreamingServer` (receives camera images from Unity) and `AnalyzeImage` (sends images to Ollama for AI analysis) in a single Python process
- **Script**: `ACRLPython/LLMcommunication/RunAnalyzer.py`
- **Default Arguments**: `--model gemma3 --server-host 127.0.0.1 --server-port 5005`
- **Port**: 5005 (StreamingServer listens here)
- **Dependencies**: `ollama`, `opencv-python`, `numpy`
- **Requirements**: Ollama must be installed and running locally

## Advanced Usage

### Conditional Server Starting

Start servers based on simulation mode:

```csharp
void Start()
{
    if (SimulationManager.Instance.config.coordinationMode == RobotCoordinationMode.Collaborative)
    {
        // Only start analyzer in collaborative mode
        PythonServerManager.Instance.StartServer("RunAnalyzer");
    }
}
```

### Custom Startup Sequence

Control startup order manually:

```csharp
IEnumerator StartServersSequentially()
{
    // Start StreamingServer first
    PythonServerManager.Instance.StartServer("StreamingServer");
    yield return new WaitForSeconds(2f);

    // Then start analyzer
    PythonServerManager.Instance.StartServer("ImageAnalyzer");
}
```

### Monitoring Server Health

Check server status periodically:

```csharp
void Update()
{
    if (Time.frameCount % 300 == 0)  // Every 5 seconds at 60fps
    {
        if (!PythonServerManager.Instance.IsServerRunning("RunAnalyzer"))
        {
            Debug.LogWarning("RunAnalyzer stopped unexpectedly!");
            // Optionally restart it
            PythonServerManager.Instance.StartServer("RunAnalyzer");
        }
    }
}
```

## Best Practices

1. **Use descriptive names**: Server names should clearly indicate what they do
2. **Document arguments**: Use the Description field to explain what arguments do
3. **Test scripts independently**: Always test Python scripts outside Unity first
4. **Keep servers lightweight**: Long-running heavy processing should be in separate processes
5. **Handle errors gracefully**: Python scripts should catch exceptions and log errors
6. **Use startup delay**: Give Unity time to initialize before starting servers (default 1s)
7. **Monitor resource usage**: Multiple Python processes can consume significant memory/CPU

## File Locations

- **Manager Script**: `ACRLUnity/Assets/Scripts/SimulationScripts/PythonServerManager.cs`
- **Python Scripts**: `ACRLPython/` directory (outside Unity project)
- **This Documentation**: `ACRLUnity/Assets/Scripts/SimulationScripts/PythonServerManager.md`

## Related Components

- **PythonCaller**: Low-level Python process execution ([PythonCaller.cs](PythonCaller.cs:1))
- **SimulationManager**: Top-level simulation orchestrator ([SimulationManager.cs](SimulationManager.cs:1))
- **MainLogger**: Unified logging system ([MainLogger.cs](../Logging/MainLogger.cs:1))

## Version History

- **v1.0** (2025-10-19): Initial implementation with extensible server configuration
  - Auto-start functionality
  - Custom inspector with runtime controls
  - Integration with PythonCaller
  - Support for multiple servers
  - Default RunAnalyzer configuration
