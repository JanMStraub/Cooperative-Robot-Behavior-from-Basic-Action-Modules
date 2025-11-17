#!/bin/bash
# Standalone script to run the servers in the background
# Run this ONCE before starting Unity - it will keep running until you stop it
# IMPORTANT: Make sure LM Studio is running with the server started before running this script

# Get absolute path to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing server processes
echo "Checking for existing server processes..."
echo "Stopping any running Python servers..."

# Kill servers by process name pattern
pkill -9 -f "RunAnalyzer.py" 2>/dev/null && echo "  ✓ Killed RunAnalyzer.py"
pkill -9 -f "RunStereoDetector.py" 2>/dev/null && echo "  ✓ Killed RunStereoDetector.py"
pkill -9 -f "RunDetector.py" 2>/dev/null && echo "  ✓ Killed RunDetector.py"
pkill -9 -f "RunRAGServer.py" 2>/dev/null && echo "  ✓ Killed RunRAGServer.py"
pkill -9 -f "RunStatusServer.py" 2>/dev/null && echo "  ✓ Killed RunStatusServer.py"

# Also kill by orchestrator module pattern
pkill -9 -f "orchestrators.RunAnalyzer" 2>/dev/null
pkill -9 -f "orchestrators.RunStereoDetector" 2>/dev/null
pkill -9 -f "orchestrators.RunDetector" 2>/dev/null
pkill -9 -f "orchestrators.RunRAGServer" 2>/dev/null
pkill -9 -f "orchestrators.RunStatusServer" 2>/dev/null

# Wait for processes to fully terminate
sleep 2
echo "All previous server processes terminated."
echo ""

echo "Starting servers..."
echo "All servers will run in the background."
echo "======================================================================"   
echo ""

# Set PYTHONPATH to include root
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR:$PYTHONPATH"

# Run servers in background with &
# RunAnalyzer uses ports 5005 (StreamingServer) and 5010 (ResultsServer for LLM results)
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/acrl/bin/python" -u -m orchestrators.RunAnalyzer --model llama-3.2-vision --interval 2.0 &
ANALYZER_PID=$!
cd "$SCRIPT_DIR"

# RunStereoDetector uses ports 5006 (StereoDetectionServer) and 5007 (ResultsServer for depth results)
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/acrl/bin/python" -u -m orchestrators.RunStereoDetector &
STEREO_PID=$!
cd "$SCRIPT_DIR"

# RunRAGServer uses port 5011 (RAGServer for semantic search)
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/acrl/bin/python" -u -m orchestrators.RunRAGServer &
RAG_PID=$!
cd "$SCRIPT_DIR"

# Wait for servers to fully initialize before starting StatusServer
# This ensures RunAnalyzer's ResultsServer is bound to port 5010
sleep 2

# RunStatusServer uses port 5012 (StatusServer for robot status queries)
# It connects to ResultsServer (port 5010) as a TCP client to send commands
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/acrl/bin/python" -u -m orchestrators.RunStatusServer &
STATUS_PID=$!
cd "$SCRIPT_DIR"

sleep 2

echo ""
echo "======================================================================"
echo "Started RunAnalyzer.py with LM Studio (PID: $ANALYZER_PID)"
echo "  - StreamingServer: port 5005"
echo "  - ResultsServer (LLM): port 5010"

echo "Started RunStereoDetector.py (PID: $STEREO_PID)"
echo "  - StereoDetectionServer: port 5006"
echo "  - ResultsServer (Depth): port 5007"

echo "Started RunRAGServer.py (PID: $RAG_PID)"
echo "  - RAGServer: port 5011"

echo "Started RunStatusServer.py (PID: $STATUS_PID)"
echo "  - StatusServer: port 5012 (receives status queries from Unity)"
echo "  - Connects to ResultsServer (port 5010) as TCP client"

echo ""
echo "All servers are running. Logs will appear below."
echo "Press Ctrl+C to stop all servers."
echo "======================================================================"
echo ""

# Function to kill all processes on exit
cleanup() {
    echo ""
    echo "======================================================================"
    echo "Stopping servers..."
    kill $ANALYZER_PID 2>/dev/null
    kill $STEREO_PID 2>/dev/null
    kill $RAG_PID 2>/dev/null
    kill $STATUS_PID 2>/dev/null
    echo "Servers stopped."
    echo "======================================================================"
    echo ""
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT SIGTERM

# Wait for both background processes
wait