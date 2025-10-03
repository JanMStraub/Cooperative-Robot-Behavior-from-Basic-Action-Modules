# Getting Started with Simplified Robot Logging

**Goal**: Get from zero to logging robot actions for LLM training in **under 5 minutes**.

## Step 1: Add Logger to Scene (1 minute)

### Option A: Through Unity Inspector (Easiest)
1. In Unity Hierarchy, right-click → Create Empty
2. Rename to "RobotLogger"
3. With RobotLogger selected, click "Add Component"
4. Search for "SimplifiedRobotLogger"
5. Click to add it

### Option B: Through Code
```csharp
// In any MonoBehaviour Start() method:
var loggerObj = new GameObject("RobotLogger");
loggerObj.AddComponent<SimplifiedRobotLogger>();
```

**That's it!** Your system is now logging. Run your simulation and check:
`Application.persistentDataPath/RobotLogs/`

## Step 2: Add Auto-Logging (Optional, 2 minutes)

Want automatic logging without manual calls? Add this to your robots:

### For Each Robot:
1. Select your RobotController GameObject
2. Add Component → "SimplifiedAutoLogger"
3. Done! It will automatically log:
   - Movement when target changes
   - Gripper actions
   - Object interactions

### Or via Code:
```csharp
// On each robot with RobotController
robotController.gameObject.AddComponent<SimplifiedAutoLogger>();
```

## Step 3: Run and Export (2 minutes)

### Run Your Simulation
1. Press Play in Unity
2. Let your robots do their thing
3. Stop when done

### Export Logs for LLM Training
**From Unity Menu** (Easiest):
- **Tools → Robot Logging → Export to JSONL**

**Or from Code**:
```csharp
SimplifiedLLMExporter.QuickExport("jsonl");
```

**Or from Context Menu**:
- Right-click QuickStartExample component
- Select "Export Logs"

### Find Your Logs
```csharp
// Logs are saved here:
string logPath = Application.persistentDataPath + "/RobotLogs/";
Debug.Log("Logs at: " + logPath);
```

Or use menu: **Tools → Robot Logging → Open Log Directory**

## You're Done! 🎉

Your robots are now:
- ✅ Logging all actions
- ✅ Capturing quality metrics
- ✅ Generating human-readable descriptions
- ✅ Tracking multi-robot coordination
- ✅ Recording trajectories
- ✅ Monitoring environment state
- ✅ Exporting in LLM-ready JSONL format

## Quick Test

Want to verify it's working? Run this test:

```csharp
[ContextMenu("Test Logging")]
void TestLogging()
{
    var logger = SimplifiedRobotLogger.Instance;

    if (logger == null)
    {
        Debug.LogError("Logger not found! Add SimplifiedRobotLogger to scene.");
        return;
    }

    // Log a simple action
    string actionId = logger.StartAction(
        "test_action",
        ActionType.Movement,
        new[] { "TestRobot" },
        description: "Testing the logging system"
    );

    logger.CompleteAction(actionId, success: true, qualityScore: 1.0f);

    Debug.Log("Test successful! Check log file.");
}
```

## Next Steps

### For Automatic Logging (Recommended)
Just attach `SimplifiedAutoLogger` to your robots and forget about it. Actions are logged automatically.

### For Manual Control
Use the API in your robot scripts:

```csharp
var logger = SimplifiedRobotLogger.Instance;

// Start action
string id = logger.StartAction(
    "move_to_target",
    ActionType.Movement,
    new[] { robotId },
    targetPos: targetPosition
);

// When done
logger.CompleteAction(id, success: true, qualityScore: 0.9f);
```

### For Advanced Usage
Check out:
- **QuickStartExample.cs** - Complete pick-and-place example
- **README.md** - Full feature documentation
- **INDEX.md** - Quick reference guide

## Common Questions

### Q: Where are my logs?
**A:** `Application.persistentDataPath/RobotLogs/robot_actions_TIMESTAMP.jsonl`

### Q: How do I export for LLM training?
**A:** Menu: Tools → Robot Logging → Export to JSONL

### Q: Can I use the old RobotActionLogger API?
**A:** Yes! SimplifiedRobotLogger is backward compatible.

### Q: Do I need to change existing code?
**A:** No! Existing `robotActionLogger.LogAction()` calls still work.

### Q: How do I log multi-robot coordination?
**A:** Use `logger.LogCoordination(name, robotIds, description)`

### Q: How do I capture environment state?
**A:** Set `logger.captureEnvironment = true` (enabled by default)

### Q: What format should I use for LLM training?
**A:** JSONL (the standard). Export with Tools → Robot Logging → Export to JSONL

### Q: Can I customize what gets logged?
**A:** Yes! Edit `SimplifiedRobotLogger.cs` - it's just one file, easy to modify.

## Troubleshooting

### "Logger not found"
✅ **Solution**: Add SimplifiedRobotLogger component to a GameObject in your scene

### "No log files generated"
✅ **Solution**:
- Check `logger.enableLogging = true`
- Make sure you're completing actions (not just starting them)
- Run simulation for a few seconds

### "Export fails"
✅ **Solution**:
- Run simulation first to generate logs
- Check console for specific error message
- Verify log directory exists and is writable

### "Log file is empty"
✅ **Solution**:
- Actions must be completed to be written
- Check you're calling `CompleteAction()` or using SimplifiedAutoLogger

## Example Output

After running your robots, your JSONL file will look like:

```json
{"logId":"movement_move_to_target_12.34_567","timestamp":"2024-01-15T10:30:45.123Z","gameTime":12.34,"logType":"action","action":{"actionId":"movement_move_to_target_12.34_567","actionName":"move_to_target","type":"Movement","status":"Completed","robotIds":["LeftRobot"],"success":true,"qualityScore":0.9,"duration":2.5,"humanReadable":"LeftRobot moving: Moving to target (successfully completed in 2.5s, quality: 0.90)","capabilities":["movement"],"complexityLevel":1},"trainingPrompt":"Task: move_to_target with robot(s) LeftRobot. Required capabilities: movement. What should happen?","trainingResponse":"LeftRobot moving: Moving to target (successfully completed in 2.5s, quality: 0.90)","difficultyLevel":"simple"}
```

Each line is a complete training example with:
- ✅ **trainingPrompt**: What the LLM should understand
- ✅ **trainingResponse**: What the LLM should generate
- ✅ **action**: Full context (success, quality, timing, etc.)
- ✅ **learningPoints**: Key takeaways

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│  SIMPLIFIED ROBOT LOGGING - QUICK REFERENCE             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  SETUP (1 minute):                                      │
│    1. Add SimplifiedRobotLogger to scene               │
│    2. (Optional) Add SimplifiedAutoLogger to robots     │
│                                                          │
│  MANUAL LOGGING:                                        │
│    var logger = SimplifiedRobotLogger.Instance;         │
│    string id = logger.StartAction(...);                │
│    logger.CompleteAction(id, success, quality);        │
│                                                          │
│  EXPORT:                                                │
│    Tools → Robot Logging → Export to JSONL             │
│                                                          │
│  LOGS LOCATION:                                         │
│    Application.persistentDataPath/RobotLogs/           │
│                                                          │
│  MENU COMMANDS:                                         │
│    • Export to JSONL                                    │
│    • Export to Conversational                           │
│    • Generate Statistics                                │
│    • Open Log Directory                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Training Your LLM

Once you have logs:

1. **Export to JSONL** (Tools → Robot Logging → Export)
2. **Load into your training pipeline**:
   ```python
   import json

   # Read JSONL
   with open('robot_actions.jsonl') as f:
       for line in f:
           entry = json.loads(line)
           prompt = entry['trainingPrompt']
           response = entry['trainingResponse']
           # Use for training...
   ```
3. **Train your model** with prompt-response pairs
4. **Done!** Your LLM now understands robot actions

## What's Next?

✅ **You're ready!** Start running robots and collecting data.

For more advanced usage:
- See **README.md** for full API documentation
- Check **QuickStartExample.cs** for code examples
- Read **INDEX.md** for quick reference
- Review **FINAL_COMPARISON.md** to see what you gained

**Happy logging! 🤖📊**
