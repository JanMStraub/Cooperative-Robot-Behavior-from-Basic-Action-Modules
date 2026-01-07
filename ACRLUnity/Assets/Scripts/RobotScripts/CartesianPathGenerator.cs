using System.Collections.Generic;
using UnityEngine;

namespace RobotScripts
{
    /// <summary>
    /// Generates Cartesian paths for robot motion planning.
    /// Currently supports linear (straight-line) paths with configurable waypoint density.
    /// </summary>
    public static class CartesianPathGenerator
    {
        /// <summary>
        /// Generate a linear Cartesian path from start pose to target pose.
        /// Creates waypoints at regular intervals along a straight line in 3D space.
        /// </summary>
        /// <param name="startPos">Starting position in local coordinates</param>
        /// <param name="startRot">Starting rotation</param>
        /// <param name="targetPos">Target position in local coordinates</param>
        /// <param name="targetRot">Target rotation</param>
        /// <param name="waypointSpacing">Distance between waypoints in meters (default 3cm)</param>
        /// <returns>CartesianPath with linearly interpolated waypoints</returns>
        public static CartesianPath GenerateLinearPath(
            Vector3 startPos,
            Quaternion startRot,
            Vector3 targetPos,
            Quaternion targetRot,
            float waypointSpacing = 0.03f
        )
        {
            float distance = Vector3.Distance(startPos, targetPos);

            // Calculate number of waypoints based on distance and spacing
            // Minimum 2 waypoints (start and end)
            int numWaypoints = Mathf.Max(2, Mathf.CeilToInt(distance / waypointSpacing));

            List<CartesianWaypoint> waypoints = new List<CartesianWaypoint>(numWaypoints + 1);

            for (int i = 0; i <= numWaypoints; i++)
            {
                float t = i / (float)numWaypoints;

                CartesianWaypoint wp = new CartesianWaypoint
                {
                    position = Vector3.Lerp(startPos, targetPos, t),
                    rotation = Quaternion.Slerp(startRot, targetRot, t),
                    distanceFromStart = distance * t,
                    timeFromStart = 0f  // Will be set by velocity profile
                };

                waypoints.Add(wp);
            }

            return new CartesianPath
            {
                waypoints = waypoints,
                totalDistance = distance,
                maxVelocity = 0.2f,  // Default, will be overridden
                acceleration = 0.5f  // Default, will be overridden
            };
        }

        /// <summary>
        /// Generate a circular arc path (future enhancement).
        /// </summary>
        public static CartesianPath GenerateArcPath(
            Vector3 startPos,
            Quaternion startRot,
            Vector3 targetPos,
            Quaternion targetRot,
            Vector3 arcCenter,
            float waypointSpacing = 0.03f
        )
        {
            // TODO: Implement circular arc path generation
            // For now, fall back to linear path
            Debug.LogWarning("[CartesianPathGenerator] Arc path not yet implemented, using linear path");
            return GenerateLinearPath(startPos, startRot, targetPos, targetRot, waypointSpacing);
        }

        /// <summary>
        /// Generate a path that avoids obstacles (future enhancement).
        /// </summary>
        public static CartesianPath GenerateObstacleAvoidingPath(
            Vector3 startPos,
            Quaternion startRot,
            Vector3 targetPos,
            Quaternion targetRot,
            List<Collider> obstacles,
            float waypointSpacing = 0.03f
        )
        {
            // TODO: Implement obstacle avoidance path generation
            // For now, fall back to linear path
            Debug.LogWarning("[CartesianPathGenerator] Obstacle avoidance not yet implemented, using linear path");
            return GenerateLinearPath(startPos, startRot, targetPos, targetRot, waypointSpacing);
        }
    }
}
