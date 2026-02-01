# SimulationScripts

Core simulation orchestration and multi-robot coordination system for ACRL.

## Overview

The SimulationScripts directory contains the top-level simulation management and multi-robot coordination strategies. It implements the Strategy Pattern for flexible coordination modes and integrates with WorkspaceManager for spatial coordination.

## Architecture

```
SimulationScripts/
├── SimulationManager.cs          # Top-level simulation orchestrator
├── WorkspaceManager.cs            # Workspace region allocation and tracking
└── CoordinationStrategies/        # Strategy Pattern implementations
    ├── ICoordinationStrategy.cs           # Strategy interface
    ├── IndependentStrategy.cs             # All robots move simultaneously
    ├── SequentialStrategy.cs              # Robots take turns (round-robin)
    ├── CollaborativeStrategy.cs           # Workspace-aware coordination with collision avoidance
    ├── ICollisionAvoidancePlanner.cs      # Collision avoidance interface
    └── WaypointCollisionAvoidancePlanner.cs  # Waypoint-based path replanning
```

## Core Components

### SimulationManager

**Purpose**: Top-level singleton orchestrator controlling simulation lifecycle and robot coordination.

**States**:
- `Initializing` - Setting up simulation configuration
- `Running` - Active simulation with robot updates
- `Paused` - Simulation stopped, robots inactive
- `Resetting` - Resetting robots to initial poses
- `Error` - Error state, optional auto-reset

**Key Features**:
- Singleton pattern with DontDestroyOnLoad
- State machine with event-driven transitions
- Coordination strategy initialization and execution
- Robot tracking and target status management
- Physics-safe reset with coroutine timing
- Custom Unity Editor inspector with runtime controls

**Configuration**:
```csharp
[SerializeField] public SimulationConfig config;
[SerializeField] private CoordinationConfig _coordinationConfig;
```

**Public API**:
```csharp
// State control
void StartSimulation()
void PauseSimulation()
void ResumeSimulation()
void ResetSimulation()

// Robot coordination queries
string GetActiveRobotId()
bool IsRobotActive(string robotId)
void NotifyTargetReached(string robotId, bool reached)

// Properties
SimulationState CurrentState
bool IsRunning
bool IsPaused
bool ShouldStopRobots
```

**Event System**:
```csharp
event System.Action<SimulationState, SimulationState> OnStateChanged;
```

**Usage Example**:
```csharp
// In scene setup
SimulationManager.Instance.config = mySimulationConfig;

// Start/pause from code
SimulationManager.Instance.StartSimulation();
SimulationManager.Instance.PauseSimulation();

// Query robot activity
bool canMove = SimulationManager.Instance.IsRobotActive("Robot1");
```

### WorkspaceManager

**Purpose**: Manages workspace region allocation and coordination for multi-robot systems.

**Key Features**:
- Workspace region definition with named bounds
- Region allocation and release tracking
- Collision zone marking
- Safety separation enforcement
- Debug visualization with Gizmos
- Allocation consistency validation

**Data Model**:
```csharp
[Serializable]
public class WorkspaceRegion
{
    public string regionName;
    public Vector3 minBounds;
    public Vector3 maxBounds;
    public string allocatedRobotId;
    public Color debugColor;
}
```

**Configuration Parameters**:
- `_minRobotSeparation` - Minimum distance between robot end effectors (default: 0.2m)
- `_allowMovementInVoid` - Allow movement outside defined regions (default: true)
- `_enableDebugVisualization` - Show workspace regions in Scene view

**Default Regions** (auto-created if none configured):
- `left_workspace` - Left robot working area (-0.65 to -0.1 in X)
- `right_workspace` - Right robot working area (0.1 to 0.65 in X)
- `shared_zone` - Central handoff area (-0.1 to 0.1 in X)
- `center` - Common task space (-0.3 to 0.3 in all axes)

**Public API**:
```csharp
// Region allocation
bool AllocateRegion(string robotId, string regionName)
void ReleaseRegion(string robotId, string regionName)
void ReleaseAllRegions(string robotId)
bool IsRegionAvailable(string regionName, string requestingRobotId = null)

// Position queries
List<WorkspaceRegion> GetRegionsAtPosition(Vector3 position)
WorkspaceRegion GetRegionAtPosition(Vector3 position)
bool IsInRobotWorkspace(string robotId, Vector3 position)
bool IsPositionAllowedForRobot(string robotId, Vector3 position)

// Safety checks
bool IsSafeSeparation(Vector3 pos1, Vector3 pos2)

// Collision zones
void MarkCollisionZone(string regionName)
void ClearCollisionZone(string regionName)
bool IsCollisionZone(string regionName)

// State management
Dictionary<string, HashSet<string>> GetAllocationState()
HashSet<string> GetRobotAllocations(string robotId)
void ResetAllocations()
bool ValidateAndRepairAllocations()
```

**Usage Example**:
```csharp
// Allocate workspace for robot
WorkspaceManager.Instance.AllocateRegion("Robot1", "left_workspace");

// Check if position is allowed
bool allowed = WorkspaceManager.Instance.IsPositionAllowedForRobot("Robot1", targetPos);

// Check safety separation
bool safe = WorkspaceManager.Instance.IsSafeSeparation(robot1Pos, robot2Pos);

// Release when done
WorkspaceManager.Instance.ReleaseAllRegions("Robot1");
```

## Coordination Strategies

### Strategy Pattern

All coordination strategies implement `ICoordinationStrategy`:

```csharp
public interface ICoordinationStrategy
{
    void Update(RobotController[] robotControllers, Dictionary<string, bool> robotTargetReached);
    bool IsRobotActive(string robotId);
    string GetActiveRobotId();
    void Reset();
}
```

### IndependentStrategy

**Implementation Status**: ✅ COMPLETE

**Behavior**: All robots operate simultaneously without coordination.

**Use Cases**:
- Single robot scenarios
- Non-overlapping workspaces
- Independent task execution

**Characteristics**:
- All robots always active (`IsRobotActive()` returns true)
- No coordination overhead
- Fallback for unimplemented modes (MasterSlave, Distributed)

**Usage**:
```csharp
config.coordinationMode = RobotCoordinationMode.Independent;
```

### SequentialStrategy

**Implementation Status**: ✅ COMPLETE

**Behavior**: Robots take turns in round-robin fashion based on sorted name order.

**Features**:
- Deterministic turn order (robots sorted by GameObject name)
- Timeout mechanism (default: 30s per robot)
- Automatic switch when target reached or timeout
- State tracking: `_activeRobotIndex`, `_robotActivationTime`

**Configuration**:
```csharp
// Default timeout
SequentialStrategy strategy = new SequentialStrategy();

// Custom timeout
SequentialStrategy strategy = new SequentialStrategy(robotTimeout: 60f);
```

**Turn Switching Conditions**:
1. Robot reaches target (`robotTargetReached[robotId] == true`)
2. Timeout exceeded (`Time.time - _robotActivationTime > _robotTimeout`)

**Usage**:
```csharp
config.coordinationMode = RobotCoordinationMode.Sequential;
```

### CollaborativeStrategy

**Implementation Status**: ⚠️ PARTIAL (Unity side complete, Python verification TODO)

**Behavior**: Workspace-aware coordination with collision detection and path replanning.

**Features**:
- WorkspaceManager integration for region allocation
- Three-level conflict detection:
  1. Target collision (targets too close)
  2. Path collision (trajectories intersect)
  3. Workspace conflict (both targeting same non-shared region)
- Priority-based conflict resolution (alphabetical robotId ordering)
- Waypoint-based path replanning via `WaypointCollisionAvoidancePlanner`
- Periodic coordination checks (500ms interval)

**Configuration** (via `CoordinationConfig` ScriptableObject):
```csharp
public class CoordinationConfig : ScriptableObject
{
    public float minSafeSeparation = 0.2f;    // Minimum robot separation
    public float verticalOffset = 0.15f;      // Vertical clearance for path replanning
    public float lateralOffset = 0.1f;        // Lateral avoidance offset
    public int maxWaypoints = 5;              // Maximum waypoints per path
}
```

**Coordination Check Flow**:
1. Update planned targets for all robots
2. Check all robot pairs for conflicts
3. Detect conflicts: target distance, path collision, workspace overlap
4. Apply priority system: lower alphabetical ID yields to higher ID
5. Attempt path replanning for low-priority robot
6. Block robot if no alternative path found

**Path Replanning Strategies** (attempted in order):
1. **Vertical offset**: Lift over obstacle, approach from above
2. **Lateral offset**: Move around obstacle (try both sides)
3. **Combined offset**: Combination of vertical + lateral

**Workspace Integration**:
```csharp
// Auto-allocate regions as robots enter them
UpdateWorkspaceAllocations(robotControllers);

// Check if target region is available
bool isAvailable = _workspaceManager.IsRegionAvailable(targetRegion.regionName, robotId);
```

**Usage**:
```csharp
config.coordinationMode = RobotCoordinationMode.Collaborative;
[SerializeField] private CoordinationConfig _coordinationConfig; // Assign in inspector
```

**TODO**:
- Python `CoordinationVerifier` integration for server-side validation
- WorldState synchronization for Python visibility

### WaypointCollisionAvoidancePlanner

**Purpose**: Generates collision-free waypoint paths for robots.

**Strategy Attempts** (in order):
1. **Vertical Offset**:
   - Lift to `current + Vector3.up * verticalOffset`
   - Move horizontally to `target + Vector3.up * verticalOffset`
   - Descend to target

2. **Lateral Offset**:
   - Move perpendicular to movement direction
   - Try right side first, then left side
   - Offset magnitude: `lateralOffset`

3. **Combined Offset**:
   - Combination of vertical + lateral
   - Try both right and left sides with vertical clearance

**Path Validation**:
```csharp
// Check if path is clear of obstacles
bool IsPathClear(List<Vector3> path, List<Vector3> obstacles, Vector3 currentPosition)

// Distance from point to line segment
float DistanceToSegment(Vector3 point, Vector3 segmentStart, Vector3 segmentEnd)
```

**Configuration Parameters**:
- `verticalOffset` - Height clearance (default: 0.15m)
- `lateralOffset` - Side clearance (default: 0.1m)
- `minSafeSeparation` - Safety margin (default: 0.2m)
- `maxWaypoints` - Path complexity limit (default: 5)

**Usage**:
```csharp
var planner = new WaypointCollisionAvoidancePlanner(
    verticalOffset: 0.15f,
    lateralOffset: 0.1f,
    minSafeSeparation: 0.2f,
    maxWaypoints: 5
);

List<Vector3> waypoints = planner.PlanAlternativePath(
    robotId: "Robot1",
    current: robotPos,
    target: targetPos,
    obstacles: new List<Vector3> { otherRobotPos1, otherRobotPos2 }
);
```

## Scene Setup

**Required GameObjects**:
```
Scene
├── SimulationManager (singleton)
│   └── SimulationConfig (ScriptableObject reference)
│   └── CoordinationConfig (optional, for Collaborative mode)
├── WorkspaceManager (singleton, optional)
├── RobotManager (singleton)
└── Robots (RobotController instances)
```

**Configuration Files**:
- `Assets/Configuration/SimulationConfig.asset` - Simulation settings
- `Assets/Configuration/DefaultCoordinationConfig.asset` - Coordination parameters

## Coordination Mode Comparison

| Mode | Status | All Robots Active | Collision Avoidance | Workspace Management | Use Case |
|------|--------|-------------------|---------------------|---------------------|----------|
| **Independent** | ✅ Complete | Yes | No | No | Non-overlapping workspaces |
| **Sequential** | ✅ Complete | No (one at a time) | Implicit (no overlap) | No | Simple turn-taking |
| **Collaborative** | ⚠️ Partial | Yes (when safe) | Yes (waypoint replanning) | Yes (region allocation) | Complex multi-robot tasks |
| **MasterSlave** | ❌ Not Implemented | - | - | - | Falls back to Independent |
| **Distributed** | ❌ Not Implemented | - | - | - | Falls back to Independent |

## Performance Considerations

**SimulationManager**:
- Single `Update()` call per frame for coordination
- Pre-allocated dictionaries for robot tracking
- Physics-safe reset with `WaitForFixedUpdate()`

**WorkspaceManager**:
- Periodic validation (5s interval) instead of every frame
- Allocation state caching
- Zero-allocation `IsPositionAllowedForRobot()` overload for pathfinding

**CollaborativeStrategy**:
- Coordination checks every 500ms (configurable)
- Pre-allocated path buffer in `WaypointCollisionAvoidancePlanner`
- Early exit for single-robot scenarios

## Custom Editor

**SimulationManager Inspector**:
- Current state display
- Active robot ID display
- Runtime control buttons: Start, Pause, Resume, Reset
- Editable configuration references

**WorkspaceManager Scene View**:
- Colored region visualization (Gizmos)
- Region labels with allocation status
- Alpha transparency indicates allocation state (0.5 = allocated, 0.2 = free)

## Integration Points

**RobotController**:
- Queries `IsRobotActive(robotId)` before moving
- Calls `NotifyTargetReached(robotId, reached)` on target events
- Respects `ShouldStopRobots` flag

**Python Backend**:
- WorldStatePublisher sends robot positions for server-side coordination
- CoordinationVerifier (TODO) validates multi-robot movements
- SequenceServer orchestrates coordinated sequences

## Testing

**Key Test Scenarios**:
1. **Independent Mode**: Multiple robots move simultaneously without interference
2. **Sequential Mode**: Robots take turns, switch on target reached or timeout
3. **Collaborative Mode**: Collision detection, path replanning, workspace allocation
4. **Workspace Management**: Region allocation, release, validation, desync repair
5. **State Transitions**: Proper state changes, event firing, reset behavior

**Test Files**:
- `ACRLUnity/Assets/Tests/PlayMode/CoordinationIntegrationTests.cs`

## Troubleshooting

**Issue**: Robots not moving in Sequential mode
- **Check**: Verify robot name sorting order
- **Check**: Check `_activeRobotIndex` in debugger
- **Check**: Ensure `NotifyTargetReached()` is called

**Issue**: Collaborative mode blocking all robots
- **Check**: WorkspaceManager instance exists in scene
- **Check**: `_blockedRobots` set size in debugger
- **Check**: Coordination check interval (may need faster checks)

**Issue**: Workspace allocation desync
- **Solution**: Manual validation via `WorkspaceManager.Instance.ValidateAndRepairAllocations()`
- **Prevention**: Automatic validation runs every 5s

**Issue**: Path replanning fails repeatedly
- **Check**: `minSafeSeparation` too large for workspace size
- **Check**: `maxWaypoints` limit too restrictive
- **Adjust**: Increase `verticalOffset` or `lateralOffset` in CoordinationConfig

## Future Enhancements

**Planned Features**:
- ✅ Workspace-based coordination (implemented)
- ✅ Waypoint collision avoidance (implemented)
- ⏳ Python CoordinationVerifier integration (TODO)
- ⏳ MasterSlave coordination mode (not implemented)
- ⏳ Distributed coordination mode (not implemented)
- ⏳ Dynamic workspace region creation
- ⏳ Machine learning-based path planning

## References

**Related Documentation**:
- Motion control redesign: `ACRLPython/documents/RobotControlRedesign.md`
- Project overview: `CLAUDE.md`
- Configuration: `ACRLUnity/Assets/Scripts/ConfigScripts/`
- Robot control: `ACRLUnity/Assets/Scripts/RobotScripts/`

**Key Files**:
- SimulationManager.cs:100 - Singleton initialization
- SimulationManager.cs:226 - Coordination strategy factory
- WorkspaceManager.cs:196 - Default workspace initialization
- CollaborativeStrategy.cs:145 - Coordination check logic
- WaypointCollisionAvoidancePlanner.cs:42 - Path planning algorithm
