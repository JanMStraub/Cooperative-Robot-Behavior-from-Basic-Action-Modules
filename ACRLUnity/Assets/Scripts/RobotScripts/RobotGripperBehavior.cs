using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Handles gripper behavior in response to robot events.
    /// Subscribes to RobotController.OnTargetReached and closes the gripper when target is reached.
    /// This implements the event-driven architecture pattern recommended in ARCHITECTURE.md.
    /// </summary>
    [RequireComponent(typeof(RobotController))]
    public class RobotGripperBehavior : MonoBehaviour
    {
        private RobotController _robotController;
        private GripperController _gripperController;

        // Helper variables
        private const string _logPrefix = "[ROBOT_GRIPPER_BEHAVIOR]";

        /// <summary>
        /// Unity Awake callback - find required components and subscribe to events.
        /// </summary>
        private void Awake()
        {
            // Get RobotController on this GameObject
            _robotController = GetComponent<RobotController>();
            if (_robotController == null)
            {
                Debug.LogError($"{_logPrefix} RobotController not found on GameObject!");
                return;
            }

            // Find GripperController in children
            _gripperController = GetComponentInChildren<GripperController>();
            if (_gripperController == null)
            {
                Debug.LogWarning(
                    $"{_logPrefix} No GripperController found in children of {gameObject.name}"
                );
                return;
            }

            // Subscribe to the OnTargetReached event
            _robotController.OnTargetReached += HandleTargetReached;
            Debug.Log($"{_logPrefix} Subscribed to OnTargetReached event for {gameObject.name}");
        }

        /// <summary>
        /// Handles the OnTargetReached event by closing the gripper.
        /// </summary>
        private void HandleTargetReached()
        {
            if (_gripperController != null)
            {
                _gripperController.CloseGrippers();
                Debug.Log($"{_logPrefix} {gameObject.name} closing grippers after reaching target");
            }
        }

        /// <summary>
        /// Unity OnDestroy callback - unsubscribe from events to prevent memory leaks.
        /// </summary>
        private void OnDestroy()
        {
            if (_robotController != null)
            {
                _robotController.OnTargetReached -= HandleTargetReached;
            }
        }
    }
}
