"""
WorldStateServer - Receives robot and object state updates from Unity.

Dedicated streaming server that receives unsolicited world state broadcasts
from Unity's WorldStatePublisher on port 5014. Keeps world state updates
separate from command/response traffic to prevent message correlation conflicts.

Architecture:
- Port 5014: One-way stream (Unity → Python)
- Message Type: STATUS_RESPONSE (0x04) with requestId=0
- No responses sent back to Unity (pure receive-only)

Usage:
    server = WorldStateServer()
    server.start()

    # Get latest state
    state = server.get_latest_state()
    print(f"Robots: {len(state['robots'])}, Objects: {len(state['objects'])}")
"""

import json
import socket
import threading
from typing import Dict, List, Optional

try:
    from core.TCPServerBase import TCPServerBase, ServerConfig
    from core.UnityProtocol import UnityProtocol, MessageType
    from config.Servers import WORLD_STATE_PORT, DEFAULT_HOST, WORLDSTATE_CHECK_INTERVAL
except ImportError:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig
    from ..core.UnityProtocol import UnityProtocol, MessageType
    from ..config.Servers import (
        WORLD_STATE_PORT,
        DEFAULT_HOST,
        WORLDSTATE_CHECK_INTERVAL,
    )


class WorldStateServer(TCPServerBase):
    """
    TCP server that receives world state updates from Unity's WorldStatePublisher.

    Maintains latest world state snapshot for spatial reasoning operations.
    Thread-safe access to state data.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        """
        Initialize WorldStateServer.

        Args:
            config: Server configuration (default: port 5014, host 0.0.0.0)
        """
        if config is None:
            config = ServerConfig(host=DEFAULT_HOST, port=WORLD_STATE_PORT)
        super().__init__(config)

        # Latest world state snapshot
        self._latest_state: Optional[Dict] = None
        self._state_lock = threading.Lock()

        # Statistics
        self._updates_received = 0
        self._last_update_time = None

    def handle_client_connection(self, client: socket.socket, address: tuple) -> None:
        """
        Handle incoming world state update from Unity.
        Runs in dedicated client thread.

        Args:
            client: Connected client socket
            address: Client address tuple (host, port)
        """
        try:
            while self.is_running():
                # Read Protocol V2 header: [Type:1][RequestId:4]
                header = self._recv_exactly(client, UnityProtocol.HEADER_SIZE)
                if not header:
                    break  # Connection closed

                msg_type, request_id, _ = UnityProtocol._decode_header(header)

                # Expect STATUS_RESPONSE with requestId=0 (broadcast)
                if msg_type != MessageType.STATUS_RESPONSE:
                    self._logger.warning(
                        f"Expected STATUS_RESPONSE, got {msg_type.name}. Skipping."
                    )
                    continue

                if request_id != 0:
                    self._logger.warning(
                        f"Expected requestId=0 (broadcast), got {request_id}. Processing anyway."
                    )

                # Read JSON length (4 bytes, little-endian per UnityProtocol.py)
                len_bytes = self._recv_exactly(client, 4)
                if not len_bytes:
                    break
                import struct

                json_length = struct.unpack("<I", len_bytes)[0]

                # Validate length
                if json_length <= 0 or json_length > UnityProtocol.MAX_IMAGE_SIZE:
                    self._logger.error(f"Invalid JSON length: {json_length}")
                    break

                # Read JSON body
                json_bytes = self._recv_exactly(client, json_length)
                if not json_bytes:
                    break

                json_str = json_bytes.decode("utf-8")

                # Parse and store world state
                try:
                    state_update = json.loads(json_str)
                    self._update_world_state(state_update)

                    robots_count = len(state_update.get("robots", []))
                    objects_count = len(state_update.get("objects", []))
                    self._logger.debug(
                        f"Received world state update: {robots_count} robots, "
                        f"{objects_count} objects"
                    )

                except json.JSONDecodeError as e:
                    self._logger.error(f"Failed to parse world state JSON: {e}")
                    continue

        except Exception as e:
            self._logger.error(f"Error handling client {address}: {e}")
        finally:
            self._logger.info(f"Client {address} disconnected")

    def _update_world_state(self, state_update: Dict) -> None:
        """
        Update latest world state snapshot (thread-safe).

        Args:
            state_update: World state update dictionary from Unity
        """
        import time

        with self._state_lock:
            self._latest_state = state_update
            self._updates_received += 1
            self._last_update_time = time.time()

    def get_latest_state(self) -> Optional[Dict]:
        """
        Get latest world state snapshot (thread-safe).

        Returns:
            World state dictionary with 'robots', 'objects', 'timestamp' keys,
            or None if no state received yet.
        """
        with self._state_lock:
            return self._latest_state.copy() if self._latest_state else None

    def get_robot_state(self, robot_id: str) -> Optional[Dict]:
        """
        Get state of specific robot (thread-safe).

        Args:
            robot_id: Robot identifier (e.g., "Robot1")

        Returns:
            Robot state dictionary or None if not found
        """
        with self._state_lock:
            if not self._latest_state:
                return None

            robots = self._latest_state.get("robots", [])
            for robot in robots:
                if robot.get("robot_id") == robot_id:
                    return robot.copy()

            return None

    def get_object_state(self, object_id: str) -> Optional[Dict]:
        """
        Get state of specific object (thread-safe).

        Args:
            object_id: Object identifier (e.g., "RedCube")

        Returns:
            Object state dictionary or None if not found
        """
        with self._state_lock:
            if not self._latest_state:
                return None

            objects = self._latest_state.get("objects", [])
            for obj in objects:
                if obj.get("object_id") == object_id:
                    return obj.copy()

            return None

    def get_all_robot_ids(self) -> List[str]:
        """
        Get list of all known robot IDs (thread-safe).

        Returns:
            List of robot IDs
        """
        with self._state_lock:
            if not self._latest_state:
                return []

            robots = self._latest_state.get("robots", [])
            return [robot.get("robot_id") for robot in robots if "robot_id" in robot]

    def get_all_object_ids(self) -> List[str]:
        """
        Get list of all known object IDs (thread-safe).

        Returns:
            List of object IDs
        """
        with self._state_lock:
            if not self._latest_state:
                return []

            objects = self._latest_state.get("objects", [])
            return [obj.get("object_id") for obj in objects if "object_id" in obj]

    def get_statistics(self) -> Dict:
        """
        Get server statistics.

        Returns:
            Dictionary with 'updates_received', 'last_update_time' keys
        """
        with self._state_lock:
            return {
                "updates_received": self._updates_received,
                "last_update_time": self._last_update_time,
                "has_state": self._latest_state is not None,
            }


# Testing and standalone execution
if __name__ == "__main__":
    import time

    print(f"Starting WorldStateServer on port {WORLD_STATE_PORT}...")
    server = WorldStateServer()
    server.start()

    try:
        print("Server running. Waiting for world state updates from Unity...")
        print("Press Ctrl+C to stop.\n")

        while True:
            time.sleep(WORLDSTATE_CHECK_INTERVAL)

            state = server.get_latest_state()
            if state:
                stats = server.get_statistics()
                print(f"\n--- World State Summary ---")
                print(f"Updates received: {stats['updates_received']}")
                print(f"Robots: {len(state.get('robots', []))}")
                print(f"Objects: {len(state.get('objects', []))}")
                print(f"Timestamp: {state.get('timestamp', 'N/A')}")

                # Print robot details
                for robot in state.get("robots", []):
                    robot_id = robot.get("robot_id", "unknown")
                    pos = robot.get("position", {})
                    gripper = robot.get("gripper_state", "unknown")
                    moving = robot.get("is_moving", False)
                    print(
                        f"  {robot_id}: pos=({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f}), "
                        f"gripper={gripper}, moving={moving}"
                    )
            else:
                print("No world state received yet...")

    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()
        print("Server stopped.")
