"""
UnityInterface.py - Unity Hardware Adapter

Wraps the existing CommandBroadcaster so that operation code can call the
RobotHardwareInterface contract without any knowledge of the TCP protocol
used to talk to the Unity simulation.

This is a zero-behavioural-change adapter: all calls delegate to the same
CommandBroadcaster that operations have always used.
"""

import logging
from hardware.Interface import RobotHardwareInterface

logger = logging.getLogger(__name__)


class UnityHardwareInterface(RobotHardwareInterface):
    """
    Concrete hardware interface that delegates to the Unity CommandBroadcaster.

    The CommandBroadcaster is retrieved lazily via core.Imports to avoid
    circular imports and to respect the module dependency hierarchy.
    """

    def _broadcaster(self):
        """Return the CommandBroadcaster singleton."""
        from core.Imports import get_command_broadcaster
        return get_command_broadcaster()

    def move_to(self, robot_id: str, x: float, y: float, z: float, **kwargs) -> bool:
        """
        Send a move_to command to Unity for the specified robot.

        Delegates to CommandBroadcaster.send_command_to_robot with a
        move_to payload containing the Cartesian target.
        """
        cmd = {"type": "move_to", "x": x, "y": y, "z": z, **kwargs}
        try:
            return self._broadcaster().send_command_to_robot(robot_id, cmd)
        except Exception as e:
            logger.error(f"UnityInterface.move_to failed: {e}")
            return False

    def set_gripper(self, robot_id: str, open: bool) -> bool:
        """
        Send a gripper open/close command to Unity.

        Maps to the "gripper" command type expected by the Unity PythonCommandHandler.
        """
        cmd = {"type": "gripper", "action": "open" if open else "close"}
        try:
            return self._broadcaster().send_command_to_robot(robot_id, cmd)
        except Exception as e:
            logger.error(f"UnityInterface.set_gripper failed: {e}")
            return False

    def get_joint_states(self, robot_id: str) -> list[float]:
        """
        Return joint states from WorldState singleton (populated by Unity streaming).

        Falls back to an empty list if the robot is not yet tracked.
        """
        try:
            from core.Imports import get_world_state
            robot_state = get_world_state().get_robot_state(robot_id)
            if robot_state:
                return robot_state.get("joint_positions", [])
            return []
        except Exception as e:
            logger.error(f"UnityInterface.get_joint_states failed: {e}")
            return []

    def emergency_stop(self) -> bool:
        """Send an estop halt_all command via the CommandBroadcaster."""
        cmd = {"type": "estop", "action": "halt_all"}
        try:
            return self._broadcaster().send_command(cmd)
        except Exception as e:
            logger.error(f"UnityInterface.emergency_stop failed: {e}")
            return False
