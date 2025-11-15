#!/bin/bash
# Standalone script to run the servers in the background
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
pkill -f "RunRAGServer.py" 2>/dev/null
pkill -f "RunStatusServer.py" 2>/dev/null
sleep 1

echo "Starting servers..."
echo "All servers will run in the background. Press Ctrl+C to stop all."
echo ""

# Set PYTHONPATH to include root
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR:$PYTHONPATH"

# Run servers in background with &
# RunAnalyzer uses ports 5005 (StreamingServer) and 5010 (ResultsServer for LLM results)
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/acrl/bin/python" -u -m orchestrators.RunAnalyzer --model llama-3.2-vision --interval 2.0 &
ANALYZER_PID=$!
cd "$SCRIPT_DIR"
echo "Started RunAnalyzer.py with LM Studio (PID: $ANALYZER_PID)"
echo "  - StreamingServer: port 5005"
echo "  - ResultsServer (LLM): port 5010"

# RunStereoDetector uses ports 5006 (StereoDetectionServer) and 5007 (ResultsServer for depth results)
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/acrl/bin/python" -u -m orchestrators.RunStereoDetector &
STEREO_PID=$!
cd "$SCRIPT_DIR"
echo "Started RunStereoDetector.py (PID: $STEREO_PID)"
echo "  - StereoDetectionServer: port 5006"
echo "  - ResultsServer (Depth): port 5007"

# RunRAGServer uses port 5011 (RAGServer for semantic search)
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/acrl/bin/python" -u -m orchestrators.RunRAGServer &
RAG_PID=$!
cd "$SCRIPT_DIR"
echo "Started RunRAGServer.py (PID: $RAG_PID)"
echo "  - RAGServer: port 5011"

# RunStatusServer uses port 5012 (StatusServer for robot status queries)
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/acrl/bin/python" -u -m orchestrators.RunStatusServer &
STATUS_PID=$!
cd "$SCRIPT_DIR"
echo "Started RunStatusServer.py (PID: $STATUS_PID)"
echo "  - StatusServer: port 5012"

echo ""
echo "All servers are running. Logs will appear below."
echo "Press Ctrl+C to stop all servers."
echo ""

# Function to kill all processes on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $ANALYZER_PID 2>/dev/null
    kill $STEREO_PID 2>/dev/null
    kill $RAG_PID 2>/dev/null
    kill $STATUS_PID 2>/dev/null
    echo "Servers stopped."
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT SIGTERM

# Wait for both background processes
wait