using System.Collections.Generic;
using Robotics;
using UnityEngine;

namespace Simulation.CoordinationStrategies
{
    /// <summary>
    /// Sequential coordination strategy where robots take turns.
    /// Only one robot is active at a time. When the active robot reaches its target,
    /// control switches to the next robot in sequence.
    /// </summary>
    public class SequentialStrategy : ICoordinationStrategy
    {
        private int _activeRobotIndex;
        private RobotController[] _robotControllers;

        /// <summary>
        /// Constructor for SequentialStrategy
        /// </summary>
        public SequentialStrategy()
        {
            _activeRobotIndex = 0;
        }

        /// <summary>
        /// Updates the sequential coordination logic.
        /// Switches to the next robot when the current robot reaches its target.
        /// </summary>
        public void Update(RobotController[] robotControllers, Dictionary<string, bool> robotTargetReached)
        {
            if (robotControllers == null || robotControllers.Length == 0)
                return;

            _robotControllers = robotControllers;

            // Validate active index
            if (_activeRobotIndex < 0 || _activeRobotIndex >= robotControllers.Length)
                _activeRobotIndex = 0;

            var currentRobot = robotControllers[_activeRobotIndex];
            if (currentRobot == null)
                return;

            string currentRobotId = currentRobot.gameObject.name;

            // Check if current robot has reached its target
            if (robotTargetReached.GetValueOrDefault(currentRobotId, true))
            {
                // Switch to next robot
                int previousIndex = _activeRobotIndex;
                _activeRobotIndex = (_activeRobotIndex + 1) % robotControllers.Length;

                Debug.Log(
                    $"[SEQUENTIAL_STRATEGY] Robot switch: {currentRobotId} (index {previousIndex}) -> {GetActiveRobotId()} (index {_activeRobotIndex})"
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
            return activeRobot != null && activeRobot.gameObject.name == robotId;
        }

        /// <summary>
        /// Gets the ID of the currently active robot.
        /// </summary>
        public string GetActiveRobotId()
        {
            if (_robotControllers == null || _activeRobotIndex >= _robotControllers.Length)
                return "None";

            var activeRobot = _robotControllers[_activeRobotIndex];
            return activeRobot != null ? activeRobot.gameObject.name : "None";
        }

        /// <summary>
        /// Resets the strategy to the first robot.
        /// </summary>
        public void Reset()
        {
            _activeRobotIndex = 0;
            Debug.Log("[SEQUENTIAL_STRATEGY] Reset to robot 0");
        }
    }
}
