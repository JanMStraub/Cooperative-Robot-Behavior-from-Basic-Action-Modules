#!/bin/bash
# Standalone script to run LLM servers in the background
# Run this ONCE before starting Unity - it will keep running until you stop it
# IMPORTANT: Make sure LM Studio is running with the server started before running this script

# Get absolute path to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing server processes
echo "Checking for existing server processes..."
pkill -f "RunAnalyzer.py" 2>/dev/null
pkill -f "RunStereoDetector.py" 2>/dev/null
pkill -f "RunDetector.py" 2>/dev/null
sleep 1

echo "Starting LLM servers..."
echo "Both servers will run in the background. Press Ctrl+C to stop all."
echo ""

# Set PYTHONPATH to include ACRLPython root
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/ACRLPython:$PYTHONPATH"

# Run servers in background with &
# RunAnalyzer uses ports 5005 (StreamingServer) and 5006 (ResultsServer)
cd "$SCRIPT_DIR/ACRLPython"
"$SCRIPT_DIR/ACRLPython/acrl/bin/python" -u -m LLMCommunication.orchestrators.RunAnalyzer --model llama-3.2-vision --interval 2.0 &
ANALYZER_PID=$!
cd "$SCRIPT_DIR"
echo "Started RunAnalyzer.py with LM Studio (PID: $ANALYZER_PID)"

# RunStereoDetector uses ports 5009 (StereoDetectionServer) and 5007 (ResultsServer)
cd "$SCRIPT_DIR/ACRLPython"
"$SCRIPT_DIR/ACRLPython/acrl/bin/python" -u -m LLMCommunication.orchestrators.RunStereoDetector --baseline 0.05 --fov 60 --results-port 5007 &
STEREO_PID=$!
cd "$SCRIPT_DIR"
echo "Started RunStereoDetector.py (PID: $STEREO_PID)"

echo ""
echo "Both servers are running. Logs will appear below."
echo "Press Ctrl+C to stop all servers."
echo ""

# Function to kill both processes on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $ANALYZER_PID 2>/dev/null
    kill $STEREO_PID 2>/dev/null
    echo "Servers stopped."
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT SIGTERM

# Wait for both background processes
wait