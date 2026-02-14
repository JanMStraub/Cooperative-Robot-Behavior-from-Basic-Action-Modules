# Robot Scripts

This directory contains the core robot control and manipulation scripts for the ACRL Unity simulation.

## Architecture Overview

The robot control system is organized into three main layers:

1. **Motion Control** - Inverse kinematics, trajectory generation, and joint control
2. **Grasp Planning** - Multi-stage grasp candidate generation, scoring, and execution
3. **Management** - Robot lifecycle, coordination, and state tracking

## Core Components

### Motion Control

#### `RobotController.cs`
Main robot controller implementing velocity-level IK with PD control for stable, oscillation-free motion.

**Key Features:**
- Three-waypoint grasp execution (pre-grasp → grasp → retreat)
- Handoff detection and execution between robots
- Moving target tracking with smooth position filtering
- Fallback to SimpleRobotController when advanced planning fails
- Integration with GraspPlanningPipeline for intelligent grasping

**Usage:**
```csharp
// Basic movement to position
robotController.SetTarget(targetPosition, GraspOptions.MoveOnly);

// Intelligent grasping with advanced planning
robotController.SetTarget(targetObject, GraspOptions.Advanced);

// Custom grasp options
var options = new GraspOptions
{
    useAdvancedPlanning = true,
    openGripperOnSet = true,
    closeGripperOnReach = true,
    approach = null  // Auto-determine best approach
};
robotController.SetTarget(targetObject, options);

// Movement with specific orientation
robotController.SetTarget(position, rotation, GraspOptions.MoveOnly);
```

#### `SimpleRobotController.cs`
Simplified IK-based controller for basic movement and fallback scenarios.

**Key Features:**
- Velocity-level IK with PD control (Kp=2.0, Kd=4.0)
- Leash/carrot approach for stable convergence (max 10cm ahead)
- Critical damping tuning for ArticulationBody drives
- Can operate autonomously or as a backup for RobotController

**Usage:**
```csharp
// Direct position control (default top-down approach)
simpleController.SetTarget(position);

// With specific approach direction
simpleController.SetTarget(position, GripperApproach.TopDown);
// or GripperApproach.Front, GripperApproach.Side, GripperApproach.Current

// GameObject tracking
simpleController.SetTarget(targetGameObject);

// With position and rotation
simpleController.SetTarget(position, rotation);

// Check status
bool hasReached = simpleController.HasReachedTarget;
float distance = simpleController.DistanceToTarget;
```

#### `IKSolver.cs`
Pure C# inverse kinematics solver using damped least-squares method.

**Key Features:**
- 6-DOF control (3 position + 3 orientation)
- Velocity-level IK: `combined_error = Kp * pos_error + Kd * vel_error`
- Joint velocity clamping (±5.0 rad/s) prevents singularity spikes
- Pre-allocated matrices for GC-free operation
- Separate position and orientation convergence thresholds

**Parameters:**
- `convergenceThreshold`: Position convergence (default: 0.02m = 2cm)
- `orientationConvergenceThreshold`: Rotation convergence (default: 0.3 rad ≈ 17°)
- `dampingFactor`: Pseudo-inverse regularization (default: 0.5)

#### `TrajectoryController.cs`
PD-controlled trajectory following with trapezoidal velocity profiles.

**Key Features:**
- Smooth acceleration/cruise/deceleration phases
- Feedforward velocity and acceleration terms
- Synchronized with FixedUpdate to prevent jitter
- Adaptive speed near targets

**PD Gains:**
- Position gain (Kp): 10.0
- Velocity gain (Kd): 2.0

#### `CartesianPath.cs` & `CartesianPathGenerator.cs`
Path representation and generation for smooth motion planning.

**Features:**
- Linear path generation with configurable waypoint spacing (default: 3cm)
- Distance-based waypoint interpolation
- Support for arc and obstacle-avoidance paths (future)

### Grasp Planning

#### `GraspPlanningPipeline.cs`
MoveIt2-inspired grasp planning with multi-stage filtering and scoring.

**Pipeline Stages:**
1. **Candidate Generation** - Generate multiple grasp poses per approach type
2. **Collision Filtering** - Remove candidates that collide with environment
3. **IK Validation** - Filter candidates that are unreachable
4. **Scoring** - Rank candidates by quality (IK ease, stability, antipodal quality)

**Scoring Weights:**
- IK Score: 30% (prefers easy-to-reach poses)
- Approach Score: 20% (prefers collision-free approach paths)
- Depth Score: 15% (prefers optimal finger penetration)
- Stability Score: 20% (prefers centered, gravity-aligned grasps)
- Antipodal Score: 15% (prefers opposing contact points)

#### `GraspCandidateGenerator.cs`
Generates diverse grasp candidates with stochastic sampling.

**Approach Types:**
- **Top**: Vertical approach from above (most stable)
- **Front**: Horizontal approach from robot side
- **Side**: Perpendicular approach from object side

**Variation Parameters:**
- Distance variation: ±20% of base pre-grasp distance
- Angle variation: ±5° from nominal approach
- Depth variation: Finger penetration into grasp region

#### `GraspScorer.cs`
Multi-criteria grasp quality evaluation.

**Scoring Criteria:**
- **IK Score**: Joint displacement from current configuration
- **Approach Score**: Clearance along approach trajectory
- **Depth Score**: Optimal finger contact depth
- **Stability Score**: Center of mass alignment, gravity consideration
- **Antipodal Score**: Opposition of contact forces

#### `GraspCollisionFilter.cs`
Filters grasp candidates that collide with environment or self.

**Features:**
- Physics-based collision detection using Unity's OverlapSphere
- Configurable clearance margins
- Self-collision avoidance

#### `GraspIKFilter.cs`
Validates grasp reachability using inverse kinematics.

**Features:**
- Fast IK pre-validation before execution
- Joint limit checking
- Convergence quality assessment
- Caches joint positions for execution

### ROS 2 Integration

The ROS 2 integration provides MoveIt-based motion planning with Unity execution. All ROS-related scripts are organized in the `Ros/` subdirectory.

#### `Ros/ROSJointStatePublisher.cs`
Publishes Unity joint states to ROS 2 at 50Hz for MoveIt state synchronization.

**Key Features:**
- Automatic joint discovery from ArticulationBody chain
- State publishing to `/joint_states` topic
- Configurable update rate (default: 50Hz)
- Automatic connection management

#### `Ros/ROSTrajectorySubscriber.cs`
Subscribes to MoveIt-generated joint trajectories and executes them in Unity.

**Key Features:**
- Listens on `/arm_controller/joint_trajectory` topic
- Converts ROS trajectories to Unity ArticulationBody targets
- Sends execution feedback to `/arm_controller/feedback`
- Integrates with TrajectoryController for smooth execution

#### `Ros/ROSGripperSubscriber.cs`
Controls gripper based on ROS gripper commands.

**Key Features:**
- Subscribes to `/gripper/command` topic
- Publishes gripper state to `/gripper/state` topic
- Integrates with GripperController for execution
- Position and effort control support

#### `Ros/ROSControlModeManager.cs`
Manages switching between Unity IK control and ROS MoveIt control modes.

**Control Modes:**
- **Unity** (default): Unity's IKSolver handles all motion planning
- **ROS**: MoveIt generates trajectories, Unity executes them
- **Hybrid**: ROS takes priority, falls back to Unity IK on failure

**Usage:**
```csharp
// Switch control modes
controlModeManager.SetControlMode(ROSControlMode.ROS);
controlModeManager.SetControlMode(ROSControlMode.Unity);
controlModeManager.SetControlMode(ROSControlMode.Hybrid);

// Check current mode
ROSControlMode currentMode = controlModeManager.CurrentMode;

// Mode automatically manages trajectory execution routing
```

**Integration Notes:**
- MoveIt is used for **planning only** - Unity executes all trajectories
- ROS plans are converted to `JointTrajectory` messages
- Unity's TrajectoryController provides smooth execution with PD control
- Feedback published to ROS for monitoring and validation

### Gripper Control

#### `Gripper/GripperController.cs`
ArticulationBody-based parallel-jaw gripper with object attachment.

**Key Features:**
- Position control with stiffness=5000, damping=500
- Anti-tunneling clamping (max 5mm target lead during closing)
- Automatic object attachment/detachment with physics state management
- Handoff support (transfers objects between grippers)
- ROS integration via ROSGripperSubscriber
- Custom editor for runtime testing

**Usage:**
```csharp
// Basic control
gripperController.OpenGrippers();
gripperController.CloseGrippers();

// Set position directly (0.0 = closed, 1.0 = open)
gripperController.SetGripperPosition(0.5f);

// With automatic attachment
gripperController.SetTargetObject(targetObject);
gripperController.CloseGrippers(); // Object attaches when closed

// Release object
gripperController.ReleaseObject(); // Detaches and re-enables physics

// Check state
bool isHolding = gripperController.IsHoldingObject;
GameObject heldObject = gripperController.GraspedObject;
bool isMoving = gripperController.IsMoving;
```

#### `Gripper/GripperContactSensor.cs`
Multi-criteria grasp verification using contact detection and force estimation.

**Verification Criteria:**
- **Contact Detection**: Both fingers touching target (100ms minimum duration)
- **Force Estimation**: Moving average force > 5N (handles Unity physics noise)
- **Closure Position**: Gripper not fully open or fully closed

**Usage:**
```csharp
// Check if both fingers are touching object (with duration check)
bool hasContact = contactSensor.HasContact(targetObject);

// Estimate grasp force (averaged over 5 frames to handle Unity physics noise)
float force = contactSensor.EstimateGraspForce();

// Multi-criteria stability check (contact + force + duration)
bool isStable = contactSensor.IsGraspStable(targetObject, minForce: 5f);

// Get all contacted objects
List<GameObject> contactedObjects = contactSensor.GetContactedObjects();

// Reset force history (call when starting new grasp attempt)
contactSensor.ResetForceHistory();
```

#### `Gripper/GripperCollisionForwarder.cs`
Forwards trigger events from finger colliders to GripperContactSensor.

**Setup:**
- Attach to each finger GameObject with trigger collider
- Assign parent GripperContactSensor and finger type (Left/Right)

### Management & Utilities

#### `RobotManager.cs`
Singleton manager for robot lifecycle and configuration.

**Features:**
- Auto-discovery of RobotController and SimpleRobotController components
- Per-robot configuration profiles (RobotConfig)
- Target change detection and event broadcasting
- Global speed multiplier for simulation control

**Usage:**
```csharp
// Register robot (auto-generates ID if not provided)
robotManager.RegisterRobot("Robot1", robotGameObject, targetGameObject, customProfile);

// Register with auto-discovery (null robotId generates one)
robotManager.RegisterRobot(null, robotGameObject);

// Get robot profile
RobotConfig profile = robotManager.GetRobotProfile("Robot1");

// Check registration
bool isRegistered = robotManager.IsRobotRegistered("Robot1");

// Unregister robot
robotManager.UnregisterRobot("Robot1");

// Access robot instances
var instances = robotManager.RobotInstances; // Read-only dictionary
int activeCount = robotManager.ActiveRobotCount;
```

#### `CollisionDetector.cs`
Trigger-based collision detection with cooldown and filtering.

**Features:**
- Configurable cooldown period (prevents duplicate events)
- Layer mask and tag filtering
- Collision logging with timestamp and approach speed
- Automatic target-reached notification

#### `CartesianPath.cs`
Data structures for Cartesian paths and velocity profiles.

**Classes:**
- `CartesianPath`: Waypoint sequence with interpolation
- `CartesianWaypoint`: Position, rotation, distance, and time
- `VelocityProfile`: Trapezoidal velocity profile calculations

#### `GraspOptions.cs`
Configuration struct for grasp execution behavior.

**Presets:**
```csharp
// Default: intelligent grasping enabled (simple planning)
var defaultOpts = GraspOptions.Default;

// Move only: no gripper control
var moveOpts = GraspOptions.MoveOnly;

// Advanced: full MoveIt2-inspired pipeline with multi-criteria scoring
var advancedOpts = GraspOptions.Advanced;
```

**Fields:**
- `useGraspPlanning`: Enable grasp planning
- `useAdvancedPlanning`: Enable full pipeline
- `openGripperOnSet`: Open gripper when setting target
- `closeGripperOnReach`: Close gripper when target reached
- `approach`: Optional approach override
- `graspConfig`: Custom grasp configuration
- `overridePreGraspDistance`: Custom pre-grasp distance
- `customApproachVector`: Custom approach direction

## Motion Control System

### Three-Layer Control Architecture

The motion control system implements a three-layer architecture designed to eliminate oscillation and improve grasp reliability:

1. **TrajectoryController** - PD control with velocity feedback
   - Generates smooth trajectories with trapezoidal velocity profiles
   - Provides target position, velocity, and acceleration
   - Synchronized with FixedUpdate
   - Used for both Unity IK and ROS trajectory execution

2. **IKSolver** - Velocity-level inverse kinematics
   - Combines position error and velocity error: `Kp * pos_error + Kd * vel_error`
   - Joint velocity clamping prevents singularity spikes
   - Convergence detection with position and velocity thresholds
   - Active in Unity and Hybrid control modes

3. **ArticulationBody** - Physics integration
   - Critical damping tuning: `damping = 2 * sqrt(stiffness * inertia)`
   - Stiffness: 2000 for all joints
   - `matchAnchors = true` prevents IK/physics conflicts

### ROS 2 Control Integration

The system supports three control modes via `Ros/ROSControlModeManager.cs`:

**Unity Mode (Default):**
- IKSolver computes joint velocities locally
- TrajectoryController applies PD control
- No ROS communication required
- Best for: Real-time reactive tasks, simple grasps, low-latency control

**ROS Mode:**
- MoveIt2 plans trajectories in Docker container
- Python ROSBridge (port 5020) sends motion requests to Docker MoveIt
- MoveIt plans and publishes `JointTrajectory` to `/arm_controller/joint_trajectory`
- Unity's ROSTrajectorySubscriber receives and converts to ArticulationBody targets
- TrajectoryController executes with PD control
- Best for: Complex collision avoidance, multi-waypoint paths, validated planning

**Hybrid Mode:**
- Attempts ROS planning first with timeout
- Falls back to Unity IK on timeout or planning failure
- Best for: Robust execution with intelligent fallback, production environments

**Architecture:**
```
[Python Backend] --TCP:5020--> [Docker ROS 2 + MoveIt2]
       |                                 |
   ROSBridge                    ROS-TCP-Endpoint (port 10000)
   (Python)                              |
       |                                 |
       +----------- Unity ROS Components (Ros/ directory) -----------+
                                         |
                              ROSControlModeManager
                                         |
                 +---------------+---------------+
                 |                               |
          ROSTrajectorySubscriber        ROSJointStatePublisher
                 |                               |
          TrajectoryController            Joint States (50Hz)
                 |
          ArticulationBody
```

### Oscillation Prevention

**Key Improvements:**
- Velocity feedback in IK solver provides natural damping
- Reduced IK step size (0.05 rad) prevents overshoot
- Increased damping factor (λ=0.5) for better regularization
- Convergence requires both position convergence AND velocity < 0.05 m/s

### Grasp Reliability

**Contact Verification:**
- Multi-criteria grasp success check (contact + force + closure)
- Moving average force estimation (5-frame window) handles Unity physics noise
- Contact duration tracking (100ms minimum) filters transient collisions

**Grasp Position Calculation:**
- Surface point: `objPos + (approachDir * (size/2))`
- Grasp point: `surfacePoint - (approachDir * fingerDepth)`
- Ensures fingers make contact rather than hovering above object

**IK Validation:**
- Tight threshold: 0.002m (2mm) for grasp IK validation
- Ensures precision: 2mm error = 4% of 5cm object

## Configuration

### `RobotConfig` (ScriptableObject)
Per-robot joint configuration and performance limits.

**Fields:**
- `joints`: Array of JointConfig (stiffness, damping, forceLimit, angleLimit)
- `convergenceThreshold`: IK position threshold (default: 0.02m)
- `maxJointStepRad`: Max joint delta per step (default: 0.05 rad)
- `adjustmentSpeed`: Speed multiplier for IK steps
- `maxReachDistance`: Maximum reach envelope
- `maxVelocity`: Cartesian velocity limit
- `acceleration`: Cartesian acceleration limit

### `IKConfig` (ScriptableObject)
Inverse kinematics solver configuration.

**Fields:**
- `dampingFactor`: Pseudo-inverse regularization (default: 0.5)
- `convergenceThreshold`: Position convergence (default: 0.02m)
- `orientationThresholdDegrees`: Rotation convergence (default: 5°)
- `maxIterations`: Max IK iterations per step (default: 50)
- `objectFindingRadius`: Radius for object snapping (default: 0.15m)
- `objectDistanceThreshold`: Distance threshold for object snapping (default: 0.1m)
- `graspTimeoutSeconds`: Timeout for grasp execution (default: 30s)

### `GraspConfig` (ScriptableObject)
Grasp planning pipeline configuration.

**Fields:**
- `candidatesPerApproach`: Number of candidates per approach type (default: 3)
- `enabledApproaches`: List of enabled approach types with weights
- `minPreGraspDistance`: Minimum pre-grasp distance (default: 0.05m)
- `maxPreGraspDistance`: Maximum pre-grasp distance (default: 0.15m)
- `targetGraspDepth`: Target finger penetration depth (default: 0.5 = 50%)
- `enableRetreat`: Enable retreat motion after grasp (default: true)
- `retreatDistance`: Retreat distance multiplier (default: 1.5x pre-grasp)
- `ikValidationThreshold`: IK convergence threshold for validation (default: 0.002m)

### `TrajectoryConfig` (ScriptableObject)
Trajectory controller PD gains.

**Fields:**
- `positionGains`: Position gain vector (default: [10, 10, 10])
- `velocityGains`: Velocity gain vector (default: [2, 2, 2])

## Grasp Planning Pipeline

### Pipeline Flow

```
1. Candidate Generation
   ├─> Generate N candidates per approach type (Top/Front/Side)
   ├─> Apply stochastic variations (distance, angle, depth)
   └─> Compute initial antipodal scores

2. Collision Filtering
   ├─> Check pre-grasp position for collisions
   ├─> Check approach trajectory for obstacles
   └─> Filter out collision-prone candidates

3. IK Validation
   ├─> Validate pre-grasp pose reachability
   ├─> Validate grasp pose reachability
   ├─> Cache joint positions for execution
   └─> Compute IK quality scores

4. Scoring & Selection
   ├─> Compute composite score (weighted sum)
   ├─> Sort candidates by total score
   └─> Select best valid candidate
```

### Approach Types

#### Top Approach
- **Use Case**: Most stable, gravity-aligned
- **Gripper Orientation**: Fingers pointing down (-Y axis)
- **Best For**: Objects on horizontal surfaces
- **Scoring**: Emphasizes horizontal centering and vertical alignment

#### Front Approach
- **Use Case**: Approaching from robot's side
- **Gripper Orientation**: Fingers horizontal, perpendicular to approach
- **Best For**: Objects at robot's workspace edge
- **Scoring**: Traditional antipodal score (opposing contact points)

#### Side Approach
- **Use Case**: Tight spaces, shelf picking
- **Gripper Orientation**: Fingers perpendicular to approach axis
- **Best For**: Objects in constrained spaces
- **Scoring**: Traditional antipodal score with side-grasp centering

### Fallback Behavior

If advanced grasp planning fails, the system falls back to SimpleRobotController:
1. Generate single grasp candidate using simple heuristics
2. Set `useSimplifiedExecution = true` on candidate
3. RobotController calls `ExecuteSimplifiedGrasp()`
4. SimpleRobotController's IK algorithm drives motion
5. Gripper closes after reaching target

## Integration with Simulation

### Event-Driven Communication

**RobotController Events:**
```csharp
// Subscribe to target reached event
robotController.OnTargetReached += () => {
    Debug.Log("Target reached!");
};

// Subscribe to coordination state changes
robotController.OnCoordinationStateChanged += (isActive) => {
    Debug.Log($"Robot active state: {isActive}");
};
```

**RobotManager Events:**
```csharp
// Subscribe to target changes
robotManager.OnTargetChanged += (robotId, target) => {
    Debug.Log($"Robot {robotId} assigned new target: {target.name}");
};
```

**GripperController Events:**
```csharp
// Subscribe to gripper action completion
gripperController.OnGripperActionComplete += () => {
    Debug.Log("Gripper finished moving");
};
```

### Coordination with SimulationManager

RobotController queries SimulationManager for:
- `ShouldStopRobots`: Emergency stop flag
- `IsRobotActive(robotId)`: Coordination system active state
- `NotifyTargetReached(robotId, reached)`: Notify coordination system

### Handoff Detection

When `SetTarget()` is called with a GameObject currently held by another gripper:
1. RobotController detects handoff scenario using `FindGripperHoldingObject()`
2. Executes `ExecuteHandoffGrasp()` coroutine
3. Receiving robot moves to object's current position
4. Receiving gripper closes (automatic transfer via `GripperController.AttachObject()`)
5. Original gripper force-releases without physics state change

## Performance Considerations

### Memory Optimization

- **Pre-allocated matrices**: IKSolver uses pre-allocated MathNet matrices (GC-free)
- **Cached temporary objects**: RobotController reuses temp GameObjects for targets
- **Cached joint info**: JointInfo arrays allocated once at startup
- **MethodImpl.AggressiveInlining**: Hot path methods in GraspCandidateGenerator

### Update Frequency

- **FixedUpdate**: IK computation (physics rate, default 50Hz)
- **Update**: Gripper position control (visual rate, uncapped)
- **0.1s intervals**: Robot active state caching (reduces dictionary lookups)

### Caching Strategy

- **Robot active state**: Cached for 100ms (reduces SimulationManager queries)
- **Joint drive updates**: Only written if changed (reduces ArticulationBody overhead)
- **IK frame transforms**: Cached and reused during IK computation

## Debugging

### Debug Visualization

Enable `_enableDebugVisualization` in RobotController to see:
- **Blue line**: End effector to target position
- **Gizmos**: Target sphere (green=reached, yellow=moving)

Enable `debugLogging` in GripperContactSensor to see:
- Contact events (enter/stay/exit)
- Force estimation values
- Grasp verification results

### Common Issues

**Robot oscillates near target:**
- Increase velocity gain (Kd) in TrajectoryConfig
- Reduce position gain (Kp) if overshoot persists
- Check ArticulationBody damping (should use critical damping)

**Grasp misses object:**
- Verify GraspConfig.ikValidationThreshold (should be 0.002m)
- Check finger depth calculation in GraspCandidateGenerator
- Ensure GripperContactSensor has correct finger assignments

**IK fails to converge:**
- Increase IKConfig.maxIterations
- Relax IKConfig.convergenceThreshold
- Check joint limits in RobotConfig

**Handoff fails:**
- Verify both grippers have GripperController attached
- Check GripperController.attachmentPoint is assigned
- Ensure target object has Rigidbody component

## Testing

### Unit Tests
Located in `ACRLUnity/Assets/Tests/PlayMode/`:
- `GraspPipelineTests.cs`: Grasp planning pipeline validation

### Manual Testing

**RobotController:**
- Assign `debugTarget` in Inspector
- Press Play to test automatic grasping

**GripperController:**
- Use custom editor buttons (Open/Close)
- Adjust slider for precise position control

## References

- Motion control redesign: `ACRLPython/documents/RobotControlRedesign.md`
- Project overview: `CLAUDE.md` (repository root)
- Configuration defaults: `ACRLUnity/Assets/Configuration/`

## ROS 2 Communication Flow

### Complete Architecture

```
┌──────────────────┐         ┌───────────────────────────┐         ┌──────────────┐
│  Unity           │         │  Docker Container         │         │  Python      │
│  (Simulation)    │         │  (ROS 2 + MoveIt2)        │         │  Backend     │
├──────────────────┤         ├───────────────────────────┤         ├──────────────┤
│ Ros/             │         │                           │         │              │
│ - JointState     │───50Hz──┤ /joint_states             │         │              │
│   Publisher      │         │                           │         │              │
│                  │         │ MoveIt2                   │◄──TCP───┤ ROSBridge    │
│ - Trajectory     │◄────────┤ /arm_controller/          │  5020   │ (ros2/)      │
│   Subscriber     │         │  joint_trajectory         │         │              │
│                  │         │                           │         │              │
│ - Gripper        │◄────────┤ /gripper/command          │         │              │
│   Subscriber     │         │                           │         │              │
│                  │─────────►/gripper/state             │         │              │
│                  │         │                           │         │              │
│ ControlMode      │         │ ROS-TCP-Endpoint          │         │              │
│ Manager          │◄──────►│ (port 10000)              │         │              │
└──────────────────┘         └───────────────────────────┘         └──────────────┘
```

### Topic Architecture

**Unity → ROS (via ROS-TCP-Endpoint on port 10000):**
- `/joint_states` (50Hz): Current joint positions/velocities for MoveIt state synchronization
- `/arm_controller/feedback`: Trajectory execution status and progress
- `/gripper/state`: Gripper position, velocity, and contact state

**ROS → Unity (via ROS-TCP-Endpoint on port 10000):**
- `/arm_controller/joint_trajectory`: MoveIt-planned trajectories (JointTrajectory messages)
- `/gripper/command`: Gripper position/effort commands

**Python ↔ Docker (TCP port 5020):**
- ROSBridge: Python backend sends motion planning requests to MoveIt
- Returns: MoveIt publishes trajectories to Unity via ROS topics

### Enabling ROS Integration

**In Python Backend** (`ACRLPython/config/ROS.py`):
```python
ROS_ENABLED = True  # Enable ROS communication
DEFAULT_CONTROL_MODE = "ros"  # or "unity" or "hybrid"
```

**Start with ROS:**
```bash
cd ACRLPython
./start_servers.sh --with-ros
```

**Per-Operation Override:**
```python
# In Python operations
result = move_to_coordinate(x, y, z, use_ros=True)
```

**In Unity:**
- Add `ROSConnectionInitializer` to scene (singleton, located in `SimulationScripts/`)
- Add ROS components (from `Ros/` directory) to robot:
  - `ROSJointStatePublisher` (publishes joint states)
  - `ROSTrajectorySubscriber` (receives MoveIt trajectories)
  - `ROSGripperSubscriber` (receives gripper commands)
  - `ROSControlModeManager` (manages control mode switching)
- Configure ROS endpoint in Inspector (default: localhost:10000)

**Docker Integration:**
- Docker container runs ROS 2 Humble + MoveIt2
- ROS-TCP-Endpoint in container bridges Unity ↔ ROS topics
- Python ROSBridge (port 5020) forwards motion requests to Docker
- See `ros_unity_integration/` directory for Docker configuration

## File Organization

```
RobotScripts/
├── README.md                        (This file)
│
├── Motion Control Layer
│   ├── RobotController.cs           (Main controller - orchestrates IK + physics)
│   ├── SimpleRobotController.cs     (Fallback controller - basic IK)
│   ├── IKSolver.cs                  (Damped least-squares IK solver)
│   ├── TrajectoryController.cs      (PD control with velocity profiles)
│   ├── CartesianPath.cs             (Path data structures)
│   └── CartesianPathGenerator.cs    (Path generation utilities)
│
├── Grasp Planning Layer
│   ├── Grasp/
│   │   ├── GraspOptions.cs              (Configuration struct)
│   │   ├── GraspPlanningPipeline.cs     (Main pipeline - orchestrates all stages)
│   │   ├── GraspCandidateGenerator.cs   (Stochastic candidate generation)
│   │   ├── GraspScorer.cs               (Multi-criteria scoring)
│   │   ├── GraspCollisionFilter.cs      (SphereCast collision filtering)
│   │   ├── GraspIKFilter.cs             (IK reachability validation)
│   │   ├── GraspCandidate.cs            (Candidate data structure)
│   │   ├── GraspApproach.cs             (Approach enum: Top/Front/Side)
│   │   └── GraspUtilities.cs            (Object size, approach determination)
│
├── Gripper Control Layer
│   └── Gripper/
│       ├── GripperController.cs         (ArticulationBody-based gripper)
│       ├── GripperContactSensor.cs      (Contact & force detection)
│       └── GripperCollisionForwarder.cs (Trigger event forwarding)
│
├── ROS 2 Integration Layer
│   └── Ros/
│       ├── ROSJointStatePublisher.cs    (Publishes joint states at 50Hz)
│       ├── ROSTrajectorySubscriber.cs   (Executes MoveIt trajectories)
│       ├── ROSGripperSubscriber.cs      (ROS gripper control)
│       └── ROSControlModeManager.cs     (Switches Unity/ROS/Hybrid modes)
│
└── Management Layer
    ├── RobotManager.cs              (Robot lifecycle & configuration)
    └── CollisionDetector.cs         (Trigger-based collision detection)
```

**Total:** 23 C# files implementing a complete robot manipulation system with ROS 2 integration

**Directory Structure:**
- **Grasp/**: Grasp planning pipeline (9 files)
- **Gripper/**: Gripper control and sensing (3 files)
- **Ros/**: ROS 2 communication layer (4 files)
- **Root**: Motion control and management (7 files)
