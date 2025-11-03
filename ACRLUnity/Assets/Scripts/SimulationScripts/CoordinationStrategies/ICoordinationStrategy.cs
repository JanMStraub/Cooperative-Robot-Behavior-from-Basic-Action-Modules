using System.Collections.Generic;
using Robotics;
using UnityEngine;

namespace Simulation.CoordinationStrategies
{
    /// <summary>
    /// Interface for robot coordination strategies.
    /// Implements the Strategy Pattern to allow different coordination modes.
    /// </summary>
    public interface ICoordinationStrategy
    {
        /// <summary>
        /// Updates the coordination logic for the current frame.
        /// Called each frame from SimulationManager.
        /// </summary>
        /// <param name="robotControllers">Array of all robot controllers in the simulation</param>
        /// <param name="robotTargetReached">Dictionary tracking which robots have reached their targets</param>
        void Update(RobotController[] robotControllers, Dictionary<string, bool> robotTargetReached);

        /// <summary>
        /// Checks if a specific robot is allowed to move based on the coordination strategy.
        /// </summary>
        /// <param name="robotId">The ID of the robot to check</param>
        /// <returns>True if the robot can move, false otherwise</returns>
        bool IsRobotActive(string robotId);

        /// <summary>
        /// Gets the ID of the currently active robot (if applicable to the strategy).
        /// </summary>
        /// <returns>The active robot ID, or null if all robots are active</returns>
        string GetActiveRobotId();

        /// <summary>
        /// Resets the coordination strategy state.
        /// Called when the simulation resets.
        /// </summary>
        void Reset();
    }
}
