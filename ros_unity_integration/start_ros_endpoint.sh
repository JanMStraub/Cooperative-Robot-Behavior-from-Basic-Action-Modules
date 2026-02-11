#!/bin/bash
# Start the ROS 2 Docker environment for ACRL (Foxglove Edition)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  ACRL ROS 2 Integration (Apple Silicon / Foxglove)"
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

case "$ACTION" in
    up|start)
        echo "Starting ROS 2 services..."
        echo ""

        # Start services using existing images (skip rebuild)
        docker compose up -d
        echo ""
        echo "Services started. Waiting for endpoints..."

        # Wait for ros_tcp_endpoint to be ready
        for i in $(seq 1 20); do
            if docker compose logs ros_tcp_endpoint 2>/dev/null | grep -q "Starting server"; then
                echo "  [OK] ros_tcp_endpoint is ready on port 10000"
                break
            fi
            sleep 1
            printf "."
        done
        
        # Wait for foxglove_bridge to be ready
        if docker compose ps | grep -q "acrl_foxglove"; then
             echo "  [OK] Foxglove Bridge is running on port 8765"
        fi
        echo ""

        echo "------------------------------------------------------------"
        echo "ROS 2 Services Status:"
        echo "  - Unity Connection: localhost:10000"
        echo "  - Foxglove Studio:  ws://localhost:8765"
        echo "  - Python Bridge:    localhost:5020"
        echo "------------------------------------------------------------"
        echo ""
        echo "NEXT STEPS:"
        echo "1. Open Foxglove Studio on your Mac."
        echo "2. Open a new 'Foxglove WebSocket' connection to ws://localhost:8765"
        echo ""
        echo "To view logs: docker compose logs -f"
        echo "To stop:      $0 down"
        ;;

    down|stop)
        echo "Stopping ROS 2 services..."
        docker compose down
        echo "All ROS services stopped."
        ;;

    logs)
        docker compose logs -f
        ;;

    status)
        docker compose ps
        ;;

    restart)
        echo "Restarting ROS 2 services..."
        docker compose down
        docker compose up -d
        echo "Services restarted."
        ;;

    build)
        echo "Building ROS 2 Docker images..."
        docker compose build
        echo "Build complete."
        ;;

    *)
        echo "Usage: $0 [up|down|logs|status|restart|build]"
        echo ""
        echo "  up/start  - Start all ROS 2 services (default)"
        echo "  down/stop - Stop all ROS 2 services"
        echo "  logs      - Follow service logs"
        echo "  status    - Show service status"
        echo "  restart   - Restart all services"
        echo "  build     - Rebuild Docker images (only needed after Dockerfile/config changes)"
        exit 1
        ;;
esac