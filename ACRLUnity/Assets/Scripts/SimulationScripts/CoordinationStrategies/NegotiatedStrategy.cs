using System.Collections.Generic;
using Robotics;
using UnityEngine;

namespace Simulation.CoordinationStrategies
{
    /// <summary>
    /// Negotiated coordination strategy where Python-side LLM agents
    /// determine robot coordination. All robots remain active since
    /// sequencing is controlled by the Python backend via signal/wait
    /// operations and parallel groups.
    /// </summary>
    public class NegotiatedStrategy : ICoordinationStrategy
    {
        /// <summary>
        /// No-op: Python backend controls sequencing via operations.
        /// </summary>
        public void Update(
            RobotController[] robotControllers,
            Dictionary<string, bool> robotTargetReached
        ) { }

        /// <summary>
        /// All robots are always active (Python controls sequencing).
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
            return "All (Negotiated)";
        }

        /// <summary>
        /// Resets the strategy.
        /// </summary>
        public void Reset()
        {
            Debug.Log("[NegotiatedStrategy] Reset");
        }
    }
}
