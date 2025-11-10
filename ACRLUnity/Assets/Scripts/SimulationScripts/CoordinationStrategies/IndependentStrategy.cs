using System.Collections.Generic;
using Robotics;
using UnityEngine;

namespace Simulation.CoordinationStrategies
{
    /// <summary>
    /// Independent coordination strategy where all robots operate independently.
    /// No coordination is performed - all robots can move simultaneously.
    /// This is the default mode for most scenarios.
    /// </summary>
    public class IndependentStrategy : ICoordinationStrategy
    {
        /// <summary>
        /// Updates the independent coordination logic.
        /// In independent mode, no coordination is needed.
        /// </summary>
        public void Update(
            RobotController[] robotControllers,
            Dictionary<string, bool> robotTargetReached
        )
        {
            // No coordination needed - all robots operate independently
        }

        /// <summary>
        /// In independent mode, all robots are always active.
        /// </summary>
        public bool IsRobotActive(string robotId)
        {
            return true; // All robots are active in independent mode
        }

        /// <summary>
        /// In independent mode, there is no single active robot.
        /// </summary>
        public string GetActiveRobotId()
        {
            return "All"; // All robots are active
        }

        /// <summary>
        /// Resets the strategy (no-op for independent mode).
        /// </summary>
        public void Reset()
        {
            // No state to reset in independent mode
        }
    }
}
