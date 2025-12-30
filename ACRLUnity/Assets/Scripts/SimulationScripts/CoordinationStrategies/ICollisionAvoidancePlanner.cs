using System.Collections.Generic;
using Robotics;
using UnityEngine;

namespace Simulation.CoordinationStrategies
{
    /// <summary>
    /// Interface for collision avoidance and path replanning strategies.
    /// Implementations provide alternative paths when robot movements conflict.
    /// </summary>
    public interface ICollisionAvoidancePlanner
    {
        /// <summary>
        /// Plan an alternative path that avoids obstacles.
        /// </summary>
        /// <param name="robotId">Robot identifier</param>
        /// <param name="current">Current position</param>
        /// <param name="target">Target position</param>
        /// <param name="obstacles">List of obstacle positions to avoid</param>
        /// <returns>List of waypoints from current to target, or empty list if no path found</returns>
        List<Vector3> PlanAlternativePath(
            string robotId,
            Vector3 current,
            Vector3 target,
            List<Vector3> obstacles
        );

        /// <summary>
        /// Check if replanning is required for a robot's movement.
        /// </summary>
        /// <param name="robotId">Robot identifier</param>
        /// <param name="target">Target position</param>
        /// <param name="otherRobots">Array of other robots to check for conflicts</param>
        /// <returns>True if replanning is needed to avoid collisions</returns>
        bool RequiresReplanning(string robotId, Vector3 target, RobotController[] otherRobots);
    }
}
