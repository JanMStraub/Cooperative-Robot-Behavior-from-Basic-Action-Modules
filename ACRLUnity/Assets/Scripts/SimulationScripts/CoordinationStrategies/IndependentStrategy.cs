using System.Collections.Generic;
using Robotics;
using UnityEngine;

namespace Simulation.CoordinationStrategies
{
    /// <summary>
    /// Independent coordination strategy where all robots operate independently.
    /// </summary>
    public class IndependentStrategy : ICoordinationStrategy
    {
        /// <summary>
        /// Updates the independent coordination logic.
        /// </summary>
        public void Update(
            RobotController[] robotControllers,
            Dictionary<string, bool> robotTargetReached
        ) { }

        /// <summary>
        /// All robots are always active.
        /// </summary>
        public bool IsRobotActive(string robotId)
        {
            return true;
        }

        /// <summary>
        /// Gets the active robot ID.
        /// </summary>
        public string GetActiveRobotId()
        {
            return "All";
        }

        /// <summary>
        /// Resets the strategy.
        /// </summary>
        public void Reset() { }
    }
}
