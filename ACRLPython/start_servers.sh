#!/bin/bash
# Standalone script to run the RobotController in the background
# Run this ONCE before starting Unity - it will keep running until you stop it
# IMPORTANT: Make sure LM Studio is running with the server started before running this script

# Get absolute path to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing server processes
echo "Checking for existing server processes..."
echo "Stopping any running Python servers..."

# Kill servers by process name pattern
pkill -9 -f "RunRobotController" 2>/dev/null && echo "  ✓ Killed RunRobotController"
pkill -9 -f "RunSequenceServer.py" 2>/dev/null && echo "  ✓ Killed RunSequenceServer.py"

# Also kill by orchestrator module pattern
pkill -9 -f "orchestrators.RunRobotController" 2>/dev/null
pkill -9 -f "orchestrators.RunSequenceServer" 2>/dev/null

# Wait for processes to fully terminate
sleep 2
echo "All previous server processes terminated."
echo ""

echo "Starting RobotController..."
echo "============================================================"
echo ""

# Check for --with-ros flag to optionally start ROS Docker services
ROS_INTEGRATION=false
for arg in "$@"; do
    if [ "$arg" = "--with-ros" ]; then
        ROS_INTEGRATION=true
    fi
done

ROS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/ros_unity_integration"
ROS_PID=""

if [ "$ROS_INTEGRATION" = true ]; then
    if [ -d "$ROS_DIR" ] && command -v docker &>/dev/null; then
        echo "Starting ROS 2 Docker services..."
        "$ROS_DIR/start_ros_endpoint.sh" up
        echo ""

        # Wait for ros_tcp_endpoint to be ready
        echo "Waiting for ROS TCP endpoint (port 10000)..."
        for i in $(seq 1 15); do
            if nc -z 127.0.0.1 10000 2>/dev/null; then
                echo "  ROS TCP endpoint ready."
                break
            fi
            sleep 1
            printf "."
        done
        echo ""
    else
        echo "WARNING: --with-ros specified but Docker or ros_unity_integration/ not found."
        echo "         Continuing without ROS integration."
        echo ""
    fi
fi

# Set PYTHONPATH to include root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run RobotController - it starts all required servers internally:
# - ImageServer (ports 5005/5006) - receives single and stereo images
# - CommandServer (port 5010) - bidirectional commands and completions
# - SequenceServer (port 5013) - sequence orchestration with RAG
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/acrl/bin/python" -u -m orchestrators.RunRobotController &
CONTROLLER_PID=$!

# Function to kill all processes on exit
cleanup() {
    echo ""
    echo "============================================================"
    echo "Stopping server..."
    kill $CONTROLLER_PID 2>/dev/null
    if [ "$ROS_INTEGRATION" = true ] && [ -d "$ROS_DIR" ]; then
        echo "Stopping ROS 2 Docker services..."
        "$ROS_DIR/start_ros_endpoint.sh" down 2>/dev/null
    fi
    echo "Server stopped."
    echo "============================================================"
    echo ""
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT SIGTERM

# Wait for background process
wait
