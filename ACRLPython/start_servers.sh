#!/bin/bash
# Standalone script to run the SequenceServer in the background
# Run this ONCE before starting Unity - it will keep running until you stop it
# IMPORTANT: Make sure LM Studio is running with the server started before running this script

# Get absolute path to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing server processes
echo "Checking for existing server processes..."
echo "Stopping any running Python servers..."

# Kill servers by process name pattern
pkill -9 -f "RunSequenceServer.py" 2>/dev/null && echo "  ✓ Killed RunSequenceServer.py"

# Also kill by orchestrator module pattern
pkill -9 -f "orchestrators.RunSequenceServer" 2>/dev/null

# Wait for processes to fully terminate
sleep 2
echo "All previous server processes terminated."
echo ""

echo "Starting SequenceServer..."
echo "============================================================"
echo ""

# Set PYTHONPATH to include root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run SequenceServer - it starts all required servers internally:
# - StreamingServer (port 5005) - receives images from Unity
# - ResultsServer (port 5010) - sends commands to Unity
# - StatusServer (port 5012) - receives completion signals from Unity
# - SequenceServer (port 5013) - receives commands from Unity
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/acrl/bin/python" -u -m orchestrators.RunSequenceServer --model gemma-3-12b &
SEQUENCE_PID=$!

sleep 5

echo ""
echo "============================================================"
echo "Started RunSequenceServer.py (PID: $SEQUENCE_PID)"
echo "  - StreamingServer: port 5005 (receives images)"
echo "  - ResultsServer: port 5010 (sends commands)"
echo "  - StatusServer: port 5012 (completion signals)"
echo "  - SequenceServer: port 5013 (receives NL commands)"
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
    kill $SEQUENCE_PID 2>/dev/null
    echo "Server stopped."
    echo "============================================================"
    echo ""
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT SIGTERM

# Wait for background process
wait
