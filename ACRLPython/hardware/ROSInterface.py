"""
ROSInterface.py - ROS/MoveIt Hardware Adapter (stub)

Wraps ROSBridge for real robot execution.  Currently a stub — methods raise
NotImplementedError until physical hardware integration is implemented in Phase 4
of the Sim-to-Real Decoupling roadmap.

The structure mirrors UnityInterface so that swapping backends only requires
changing the factory in hardware/__init__.py.
"""

import logging
from hardware.Interface import RobotHardwareInterface

logger = logging.getLogger(__name__)


class ROSHardwareInterface(RobotHardwareInterface):
    """
    Concrete hardware interface that delegates to the ROS 2 / MoveIt bridge.

    Phase 4 implementation: replace NotImplementedError stubs with real
    ROSBridge calls once physical AR4 drivers are available.
    """

    def _bridge(self):
        """Return the ROSBridge singleton."""
        from ros2.ROSBridge import ROSBridge
        return ROSBridge.get_instance()

    def move_to(self, robot_id: str, x: float, y: float, z: float, **kwargs) -> bool:
        """
        Plan and execute a Cartesian move via MoveIt.

        Phase 4: Not yet implemented.
        """
        raise NotImplementedError(
            "ROSHardwareInterface.move_to is not yet implemented. "
            "Phase 4 (MoveIt execution on physical AR4) required."
        )

    def set_gripper(self, robot_id: str, open: bool) -> bool:
        """
        Control the gripper via ROS topic.

        Phase 4: Not yet implemented.
        """
        raise NotImplementedError(
            "ROSHardwareInterface.set_gripper is not yet implemented."
        )

    def get_joint_states(self, robot_id: str) -> list[float]:
        """
        Read joint states from the ROS /joint_states topic.

        Phase 4: Not yet implemented.
        """
        raise NotImplementedError(
            "ROSHardwareInterface.get_joint_states is not yet implemented."
        )

    def emergency_stop(self) -> bool:
        """
        Send an emergency stop via ROS.

        Phase 4: Not yet implemented.
        """
        raise NotImplementedError(
            "ROSHardwareInterface.emergency_stop is not yet implemented."
        )
