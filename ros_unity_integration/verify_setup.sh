#!/bin/bash
# Verify Dual-Robot ROS Setup
# This script validates the configuration before starting services

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Dual-Robot ROS Setup Verification"
echo "============================================================"
echo ""

# Check docker-compose syntax
echo "1. Validating docker-compose.yml..."
if docker compose config > /dev/null 2>&1; then
    echo "   ✓ docker-compose.yml syntax is valid"
else
    echo "   ✗ docker-compose.yml has syntax errors"
    docker compose config
    exit 1
fi
echo ""

# Check launch files exist
echo "2. Checking launch files..."
if [ -f "ar4_moveit_config/launch/move_group.launch.py" ]; then
    echo "   ✓ move_group.launch.py exists"
else
    echo "   ✗ move_group.launch.py not found"
    exit 1
fi

if [ -f "ar4_moveit_config/launch/robot_state_publisher.launch.py" ]; then
    echo "   ✓ robot_state_publisher.launch.py exists"
else
    echo "   ✗ robot_state_publisher.launch.py not found"
    exit 1
fi
echo ""

# Check Python files
echo "3. Checking Python ROS bridge files..."
if [ -f "../ACRLPython/ros2/ROSMotionClient.py" ]; then
    echo "   ✓ ROSMotionClient.py exists"
else
    echo "   ✗ ROSMotionClient.py not found"
    exit 1
fi

if [ -f "../ACRLPython/ros2/ROSBridge.py" ]; then
    echo "   ✓ ROSBridge.py exists"
else
    echo "   ✗ ROSBridge.py not found"
    exit 1
fi
echo ""

# Check for robot_id parameter in launch files
echo "4. Verifying robot_id parameter in launch files..."
if grep -q "robot_id" "ar4_moveit_config/launch/move_group.launch.py"; then
    echo "   ✓ move_group.launch.py has robot_id parameter"
else
    echo "   ✗ move_group.launch.py missing robot_id parameter"
    exit 1
fi

if grep -q "robot_id" "ar4_moveit_config/launch/robot_state_publisher.launch.py"; then
    echo "   ✓ robot_state_publisher.launch.py has robot_id parameter"
else
    echo "   ✗ robot_state_publisher.launch.py missing robot_id parameter"
    exit 1
fi
echo ""

# Verify expected services in docker-compose
echo "5. Verifying expected services in docker-compose.yml..."
SERVICES=$(docker compose config --services)

for service in "moveit_robot1" "moveit_robot2" "robot_state_publisher_robot1" "robot_state_publisher_robot2" "ros_bridge"; do
    if echo "$SERVICES" | grep -q "^${service}$"; then
        echo "   ✓ Service '$service' defined"
    else
        echo "   ✗ Service '$service' not found"
        exit 1
    fi
done
echo ""

# Check for namespace configuration
echo "6. Checking namespace configuration..."
if docker compose config | grep -q "robot_id:=Robot1"; then
    echo "   ✓ Robot1 namespace configured"
else
    echo "   ✗ Robot1 namespace not configured"
    exit 1
fi

if docker compose config | grep -q "robot_id:=Robot2"; then
    echo "   ✓ Robot2 namespace configured"
else
    echo "   ✗ Robot2 namespace not configured"
    exit 1
fi
echo ""

# Summary
echo "============================================================"
echo "  ✓ All verification checks passed!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Build Docker images (if first time):"
echo "     ./start_ros_endpoint.sh build"
echo ""
echo "  2. Start services:"
echo "     ./start_ros_endpoint.sh up"
echo ""
echo "  3. Verify services are running:"
echo "     docker compose ps"
echo ""
echo "  4. Run tests:"
echo "     cd ../ACRLPython && python tests/TestDualRobotROS.py"
echo ""
