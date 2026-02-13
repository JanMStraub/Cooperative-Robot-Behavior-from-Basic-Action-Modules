# ROS 2 Integration for ACRL

Docker-based ROS 2 environment providing **collision-aware motion planning** for the AR4 mk3 robot arm via MoveIt 2.

## Key Features

- ✅ **Plan-Only Architecture** — MoveIt plans, Unity executes (no ros2_control needed)
- ✅ **Hybrid Control** — Switch between Unity IK and ROS planning at runtime
- ✅ **Feature Flags** — ROS integration coexists with existing TCP control path
- ✅ **Zero Code Changes Required** — Default config uses Unity IK (ROS disabled)
- ✅ **Full Docker Isolation** — All ROS dependencies containerized

## Quick Start

```bash
# 1. Start ROS services
cd ros_unity_integration
./start_ros_endpoint.sh up

# 2. Verify services are running
./start_ros_endpoint.sh status

# 3. Start Python backend with ROS support
cd ../ACRLPython
./start_servers.sh --with-ros

# 4. Open Unity and press Play
# (Add ROS components to robot - see QUICKSTART.md)
```

## Services

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| ros_tcp_endpoint | acrl_ros_endpoint | 10000 | Unity <-> ROS 2 bridge |
| moveit | acrl_moveit | - | MoveIt 2 motion planning |
| robot_state_publisher | acrl_robot_state_publisher | - | TF transforms from URDF |
| ros_bridge | acrl_ros_bridge | 5020 | Python backend <-> ROS bridge |

## Architecture Overview

```
Python Backend → ROSBridge (TCP:5020) → ROSMotionClient (Docker)
                                               ↓
                                         MoveIt 2 (plan_only=True)
                                               ↓
                                         Publish JointTrajectory to ROS topic
                                               ↓
                                         ros_tcp_endpoint (port 10000)
                                               ↓
                                         Unity ROSTrajectorySubscriber
                                               ↓
                                         ArticulationBody execution
```

**Key Point**: MoveIt only **plans** trajectories (no execution), Unity is the physics executor.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed data flows and component diagrams.

## Unity Setup

Add these components to each robot GameObject:
- `ROSJointStatePublisher` - publishes joint states to `/joint_states`
- `ROSTrajectorySubscriber` - executes trajectories from `/arm_controller/joint_trajectory`
- `ROSGripperSubscriber` - receives gripper commands on `/gripper/command`
- `ROSControlModeManager` - switches between Unity IK and ROS control

Add to scene:
- `ROSConnectionInitializer` - manages ROS connection (singleton)

## Control Modes

Set via `ROSControlModeManager` component in Unity Inspector:

| Mode | Unity IK | ROS Planning | Use Case |
|------|----------|--------------|----------|
| **Unity** (default) | ✓ Active | ✗ Disabled | Testing, no Docker dependency |
| **ROS** | ✗ Bypassed | ✓ Active | Collision-aware planning |
| **Hybrid** | ✓ Dynamic | ✓ Dynamic | Best of both (auto-switching) |

**Default behavior**: ROS disabled (`ROS_ENABLED = False` in `ACRLPython/config/ROS.py`)

## Commands

```bash
./start_ros_endpoint.sh up       # Start services
./start_ros_endpoint.sh down     # Stop services
./start_ros_endpoint.sh logs     # View logs
./start_ros_endpoint.sh status   # Check status
./start_ros_endpoint.sh restart  # Restart services
```

## ROS Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/joint_states` | sensor_msgs/JointState | Unity -> ROS | Current joint positions (50Hz) |
| `/arm_controller/joint_trajectory` | trajectory_msgs/JointTrajectory | ROS -> Unity | Planned trajectories |
| `/arm_controller/feedback` | std_msgs/String | Unity -> ROS | Execution feedback |
| `/gripper/command` | sensor_msgs/JointState | ROS -> Unity | Gripper commands |
| `/gripper/state` | sensor_msgs/JointState | Unity -> ROS | Gripper state feedback |

## Coordinate Systems

**Important**: The ROS integration automatically handles coordinate transformations between Unity world space and robot-local `base_link` frames.

### Robot Positions in Unity
- **Robot1**: Unity world position `(-0.475, 0, 0)` meters
- **Robot2**: Unity world position `(0.475, 0, 0)` meters

### Using World Coordinates (Recommended)
When sending commands via Python, use **Unity world coordinates**:

```python
from ros2.ROSBridge import ROSBridge

bridge = ROSBridge.get_instance()
bridge.connect()

# Send world coordinates - transformation happens automatically
bridge.plan_and_execute(
    position={"x": -0.2, "y": 0.05, "z": 0.0},  # Unity world coords
    robot_id="Robot1"
)
```

### Coordinate Transformation
The `ROSMotionClient` automatically transforms coordinates:
- **Input**: Unity world coordinates (what you send from Python)
- **Planning**: MoveIt plans in `base_link` local coordinates
- **Execution**: Unity executes in world space

Example for Robot1:
- World position: `(x=-0.2, y=0.05, z=0.0)`
- Robot1 base at: `(-0.475, 0, 0)`
- MoveIt receives: `(x=0.275, y=0.05, z=0.0)` ← automatic transformation
