using System.Collections.Generic;
using Robotics;
using UnityEngine;

namespace Simulation.CoordinationStrategies
{
    /// <summary>
    /// Sequential coordination strategy where robots take turns.
    /// </summary>
    public class SequentialStrategy : ICoordinationStrategy
    {
        private int _activeRobotIndex;
        private RobotController[] _robotControllers;
        private float _robotActivationTime;
        private float _robotTimeout;

        private const string _logPrefix = "[SEQUENTIAL_STRATEGY]";
        private const float DEFAULT_ROBOT_TIMEOUT = 30f;

        /// <summary>
        /// Constructor for SequentialStrategy.
        /// </summary>
        public SequentialStrategy()
            : this(DEFAULT_ROBOT_TIMEOUT) { }

        /// <summary>
        /// Constructor for SequentialStrategy with custom timeout.
        /// </summary>
        /// <param name="robotTimeout">Timeout in seconds</param>
        public SequentialStrategy(float robotTimeout)
        {
            _activeRobotIndex = 0;
            _robotTimeout = robotTimeout;
            _robotActivationTime = Time.time;
        }

        /// <summary>
        /// Updates the sequential coordination logic.
        /// </summary>
        public void Update(
            RobotController[] robotControllers,
            Dictionary<string, bool> robotTargetReached
        )
        {
            if (robotControllers == null || robotControllers.Length == 0)
                return;

            _robotControllers = robotControllers;

            if (_activeRobotIndex < 0 || _activeRobotIndex >= robotControllers.Length)
                _activeRobotIndex = 0;

            var currentRobot = robotControllers[_activeRobotIndex];
            if (currentRobot == null)
                return;

            string currentRobotId = currentRobot.robotId;

            float timeSinceActivation = Time.time - _robotActivationTime;
            bool hasTimedOut = timeSinceActivation > _robotTimeout;
            bool hasReachedTarget = robotTargetReached.GetValueOrDefault(currentRobotId, false);

            if (hasReachedTarget || hasTimedOut)
            {
                if (hasTimedOut)
                {
                    Debug.LogWarning(
                        $"{_logPrefix} Robot {currentRobotId} timed out after {timeSinceActivation:F1}s (timeout: {_robotTimeout}s). Switching to next robot."
                    );
                }

                int previousIndex = _activeRobotIndex;
                _activeRobotIndex = (_activeRobotIndex + 1) % robotControllers.Length;
                _robotActivationTime = Time.time;

                Debug.Log(
                    $"{_logPrefix} Robot switch: {currentRobotId} (index {previousIndex}) -> {GetActiveRobotId()} (index {_activeRobotIndex})"
                );
            }
        }

        /// <summary>
        /// Checks if a robot is the currently active robot.
        /// </summary>
        public bool IsRobotActive(string robotId)
        {
            if (_robotControllers == null || _activeRobotIndex >= _robotControllers.Length)
                return false;

            var activeRobot = _robotControllers[_activeRobotIndex];
            return activeRobot != null && activeRobot.robotId == robotId;
        }

        /// <summary>
        /// Gets the ID of the currently active robot.
        /// </summary>
        public string GetActiveRobotId()
        {
            if (_robotControllers == null || _activeRobotIndex >= _robotControllers.Length)
                return "None";

            var activeRobot = _robotControllers[_activeRobotIndex];
            return activeRobot != null ? activeRobot.robotId : "None";
        }

        /// <summary>
        /// Resets the strategy to the first robot.
        /// </summary>
        public void Reset()
        {
            _activeRobotIndex = 0;
            _robotActivationTime = Time.time;
            Debug.Log($"{_logPrefix} Reset to robot 0");
        }
    }
}
