#!/bin/bash
# Standalone script to run LLM servers in the background
# Run this ONCE before starting Unity - it will keep running until you stop it
# IMPORTANT: Make sure LM Studio is running with the server started before running this script

cd "$(dirname "$0")"

echo "Starting LLM servers..."
echo "Both servers will run in the background. Press Ctrl+C to stop all."
echo ""

# Run servers in background with &
./ACRLPython/acrl/bin/python -u ACRLPython/LLMCommunication/orchestrators/RunAnalyzer.py --model llama-3.2-vision --interval 2.0 &
ANALYZER_PID=$!
echo "Started RunAnalyzer.py with LM Studio (PID: $ANALYZER_PID)"

./ACRLPython/acrl/bin/python -u ACRLPython/LLMCommunication/orchestrators/RunStereoDetector.py --baseline 0.1 --fov 60 --results-port 5007 &
STEREO_PID=$!
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