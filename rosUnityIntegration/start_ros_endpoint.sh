#!/bin/bash
# Start the ROS 2 Docker environment for ACRL (Foxglove Edition)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Starting ACRL ROS 2 Integration (Apple Silicon / Foxglove)"
echo "============================================================"
echo ""

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is not installed or not in PATH."
    echo "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

if ! docker info &>/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running."
    echo "Start Docker Desktop and try again."
    exit 1
fi

# Parse arguments
ACTION="${1:-up}"
PROFILE_ARGS=()
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE_ARGS+=("--profile" "$2")
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

case "$ACTION" in
    up|start)

        # Start services using existing images (skip rebuild)
        docker compose "${PROFILE_ARGS[@]}" up -d
        echo ""
        echo "Services started. Waiting for endpoints..."

        # Wait for ros_tcp_endpoint to be ready
        for i in $(seq 1 20); do
            if docker compose "${PROFILE_ARGS[@]}" logs ros_tcp_endpoint 2>/dev/null | grep -q "Starting server"; then
                echo "  [OK] ros_tcp_endpoint is ready on port 10000"
                break
            fi
            sleep 1
            printf "."
        done

        # Wait for foxglove_bridge to be ready
        if docker compose "${PROFILE_ARGS[@]}" ps | grep -q "acrl_foxglove"; then
             echo "  [OK] Foxglove Bridge is running on port 8765"
        fi
        ;;

    down|stop)
        echo "Stopping ROS 2 services..."
        docker compose "${PROFILE_ARGS[@]}" down
        echo "All ROS services stopped."
        ;;

    logs)
        docker compose "${PROFILE_ARGS[@]}" logs -f
        ;;

    status)
        docker compose "${PROFILE_ARGS[@]}" ps
        ;;

    restart)
        echo "Restarting ROS 2 services..."
        docker compose "${PROFILE_ARGS[@]}" down
        docker compose "${PROFILE_ARGS[@]}" up -d
        echo "Services restarted."
        ;;

    build)
        echo "Building ROS 2 Docker images..."
        docker compose "${PROFILE_ARGS[@]}" build
        echo "Build complete."
        ;;

    *)
        echo "Usage: $0 [up|down|logs|status|restart|build] [--profile PROFILE]"
        echo ""
        echo "  up/start  - Start all ROS 2 services (default)"
        echo "  down/stop - Stop all ROS 2 services"
        echo "  logs      - Follow service logs"
        echo "  status    - Show service status"
        echo "  restart   - Restart all services"
        echo "  build     - Rebuild Docker images (only needed after Dockerfile/config changes)"
        echo ""
        echo "Options:"
        echo "  --profile PROFILE  - Enable a Docker Compose profile (e.g. --profile dual)"
        exit 1
        ;;
esac
