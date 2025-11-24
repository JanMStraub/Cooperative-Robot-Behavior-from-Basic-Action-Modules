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

# Set PYTHONPATH to include root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run RobotController - it starts all required servers internally:
# - ImageServer (ports 5005/5006) - receives single and stereo images
# - CommandServer (port 5010) - bidirectional commands and completions
# - SequenceServer (port 5013) - sequence orchestration with RAG
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/acrl/bin/python" -u -m orchestrators.RunRobotController &
CONTROLLER_PID=$!

sleep 5

echo ""
echo "============================================================"
echo "Started RunRobotController (PID: $CONTROLLER_PID)"
echo "  - ImageServer: ports 5005/5006 (receives images)"
echo "  - CommandServer: port 5010 (bidirectional commands)"
echo "  - SequenceServer: port 5013 (sequences + RAG)"
echo ""
echo "Server is running. Logs will appear below."
echo "Press Ctrl+C to stop."
echo "============================================================"
echo ""

# Function to kill all processes on exit
cleanup() {
    echo ""
    echo "============================================================"
    echo "Stopping server..."
    kill $CONTROLLER_PID 2>/dev/null
    echo "Server stopped."
    echo "============================================================"
    echo ""
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT SIGTERM

# Wait for background process
wait
