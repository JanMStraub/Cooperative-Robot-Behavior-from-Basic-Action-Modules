using Logging;
using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Handles all logging for RobotController actions.
    /// Encapsulates MainLogger interactions and provides high-level logging methods.
    /// </summary>
    public class RobotActionLogger
    {
        private readonly string _robotId;
        private readonly MainLogger _logger;
        private readonly Transform _endEffectorTransform;

        /// <summary>
        /// Creates a new action logger for a robot
        /// </summary>
        /// <param name="robotId">Robot identifier</param>
        /// <param name="endEffectorTransform">End effector transform for position logging</param>
        public RobotActionLogger(string robotId, Transform endEffectorTransform)
        {
            _robotId = robotId;
            _endEffectorTransform = endEffectorTransform;
            _logger = MainLogger.Instance;
        }

        /// <summary>
        /// Log target assignment action
        /// </summary>
        public void LogTargetSet(string targetName, Vector3 targetPosition, bool useGraspPlanning)
        {
            if (_logger == null)
                return;

            string actionId = _logger.StartAction(
                actionName: "set_target",
                type: ActionType.Movement,
                robotIds: new[] { _robotId },
                startPos: _endEffectorTransform.position,
                targetPos: targetPosition,
                objectIds: targetName != null ? new[] { targetName } : null,
                description: $"Setting target (grasp planning: {useGraspPlanning})"
            );
            _logger.CompleteAction(actionId, success: true, qualityScore: 1f);
        }

        /// <summary>
        /// Log target reached event
        /// </summary>
        public void LogTargetReached(
            string targetName,
            Vector3 targetPosition,
            float distance,
            float convergenceThreshold
        )
        {
            if (_logger == null)
                return;

            string actionId = _logger.StartAction(
                actionName: "reach_target",
                type: ActionType.Movement,
                robotIds: new[] { _robotId },
                startPos: _endEffectorTransform.position,
                targetPos: targetPosition,
                objectIds: new[] { targetName },
                description: $"Reached target {targetName}"
            );

            float quality = Mathf.Max(0f, 1f - distance / convergenceThreshold);
            _logger.CompleteAction(actionId, success: true, qualityScore: quality);
        }

        /// <summary>
        /// Log gripper action (open or close)
        /// </summary>
        public void LogGripperAction(string actionName, string description)
        {
            if (_logger == null)
                return;

            string actionId = _logger.StartAction(
                actionName,
                ActionType.Manipulation,
                new[] { _robotId },
                startPos: _endEffectorTransform.position,
                description: description
            );
            _logger.CompleteAction(actionId, success: true, qualityScore: 1.0f);
        }

        /// <summary>
        /// Log robot initialization
        /// </summary>
        public void LogInitialization(GameObject robotObject)
        {
            if (_logger == null)
                return;

            string actionId = _logger.StartAction(
                actionName: "initialize_robot",
                type: ActionType.Task,
                robotIds: new[] { _robotId },
                startPos: _endEffectorTransform.position,
                objectIds: new[] { robotObject.name },
                description: $"Initialized robot {_robotId}"
            );
            _logger.CompleteAction(actionId, success: true, qualityScore: 1f);
        }
    }
}
