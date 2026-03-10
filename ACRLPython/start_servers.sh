#!/bin/bash
# Standalone script to run the RobotController in the background
# Run this ONCE before starting Unity - it will keep running until you stop it
# IMPORTANT: Make sure LM Studio is running with the server started before running this script

# Exit on error, treat unset variables as errors, and propagate pipeline failures
set -euo pipefail

# Get absolute path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ---
PYTHON_EXEC="$SCRIPT_DIR/acrl/bin/python"
CONTROLLER_PATTERN="orchestrators.RunRobotController"
SEQUENCE_SERVER_PATTERN="orchestrators.RunSequenceServer"
ROS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/rosUnityIntegration"
CONTROLLER_PID=""
ROS_INTEGRATION=true

# --- Functions ---

print_header() {
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

kill_process_by_pattern() {
    local pattern="$1"
    local name="$2"
    # Send SIGTERM (15) for graceful shutdown. -f matches the full command line.
    if pkill -f "$pattern"; then
        echo "  ✓ Sent stop signal to $name"
    fi
}

# Kill any existing server processes
kill_existing_servers() {
    print_header "Checking for existing server processes"
    kill_process_by_pattern "$CONTROLLER_PATTERN" "RunRobotController"
    kill_process_by_pattern "$SEQUENCE_SERVER_PATTERN" "RunSequenceServer"

    # Wait for processes to fully terminate
    echo "Waiting for processes to shut down..."
    sleep 2
    echo "All previous server processes terminated."
    echo ""
}

start_ros() {
    if ! "$ROS_INTEGRATION"; then
        return
    fi

    if [ -d "$ROS_DIR" ] && command -v docker &>/dev/null; then
        # Check if containers are already up and the bridge is responsive
        if echo '{"command":"ping"}' | nc -w 1 127.0.0.1 5020 2>/dev/null | grep -q '"success": true'; then
            echo "ROS 2 Docker services already running — skipping startup."
            echo ""
            return
        fi

        echo "Starting ROS 2 Docker services..."
        "$ROS_DIR/start_ros_endpoint.sh" up
        echo ""

        # Wait for ros_tcp_endpoint (port 10000) AND ros_bridge motion server (port 5020)
        local timeout=60

        echo -n "Waiting for ROS TCP endpoint (port 10000)..."
        for (( i=0; i<timeout; i++ )); do
            if nc -z 127.0.0.1 10000 2>/dev/null; then
                echo " ready."
                break
            fi
            printf "."
            sleep 1
            if (( i == timeout - 1 )); then
                echo " FAILED."
                echo "ERROR: ROS TCP endpoint did not become available in time." >&2
                exit 1
            fi
        done

        echo -n "Waiting for ROS bridge motion server (port 5020)..."
        for (( i=0; i<timeout; i++ )); do
            if echo '{"command":"ping"}' | nc -w 1 127.0.0.1 5020 2>/dev/null | grep -q '"success": true'; then
                echo " ready."
                echo ""
                return
            fi
            printf "."
            sleep 1
        done

        echo " FAILED."
        echo "ERROR: ROS bridge motion server did not become available in time." >&2
        exit 1
    else
        echo "WARNING: ROS enabled but Docker or '$ROS_DIR' not found."
        echo "         Continuing without ROS integration."
        echo ""
    fi
}

start_controller() {
    print_header "Starting RobotController"

    # Set PYTHONPATH to include root
    export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

    # Run RobotController - it starts all required servers internally:
    # - ImageServer (ports 5005/5006) - receives single and stereo images
    # - CommandServer (port 5010) - bidirectional commands and completions
    # - SequenceServer (port 5013) - sequence orchestration with RAG
    cd "$SCRIPT_DIR"

    # Enable job control to manage the background process group
    set -m
    "$PYTHON_EXEC" -u -m orchestrators.RunRobotController &
    CONTROLLER_PID=$!
    set +m

    echo "RobotController started with PID: $CONTROLLER_PID"
    echo "Press Ctrl+C to stop all servers."
}

# Function to kill all processes on exit
cleanup() {
    echo ""
    print_header "Stopping server"

    # Kill the Python controller process group to terminate it and its children
    if [ -n "$CONTROLLER_PID" ]; then
        echo "Stopping RobotController (PID: $CONTROLLER_PID)..."
        # Kill process group first, fall back to individual process if that fails
        kill -SIGTERM -- "-$CONTROLLER_PID" 2>/dev/null || kill -SIGTERM "$CONTROLLER_PID" 2>/dev/null || true
        wait "$CONTROLLER_PID" 2>/dev/null || true
    fi

    echo "Server stopped. (ROS Docker containers left running)"
    exit 0
}

main() {
    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --without-ros)
                ROS_INTEGRATION=false
                shift
                ;;
            *)
                echo "Unknown option: $1" >&2
                echo "Usage: $0 [--without-ros]" >&2
                exit 1
                ;;
        esac
    done

    # Trap Ctrl+C and call cleanup
    trap cleanup SIGINT SIGTERM

    kill_existing_servers
    start_ros
    start_controller

    # Wait for background process to exit. Cleanup is handled by the trap.
    wait "$CONTROLLER_PID"
}

# Execute the main function with all script arguments
main "$@"
