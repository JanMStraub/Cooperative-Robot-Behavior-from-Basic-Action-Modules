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
        /// <param name="maxVelocity">Maximum path velocity in m/s (default 0.2)</param>
        /// <param name="acceleration">Path acceleration in m/s² (default 0.5)</param>
        /// <returns>CartesianPath with linearly interpolated waypoints</returns>
        public static CartesianPath GenerateLinearPath(
            Vector3 startPos,
            Quaternion startRot,
            Vector3 targetPos,
            Quaternion targetRot,
            float waypointSpacing = 0.03f,
            float maxVelocity = 0.2f,
            float acceleration = 0.5f
        )
        {
            float distance = Vector3.Distance(startPos, targetPos);

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
                };

                waypoints.Add(wp);
            }

            return new CartesianPath
            {
                waypoints = waypoints,
                totalDistance = distance,
                maxVelocity = maxVelocity,
                acceleration = acceleration,
            };
        }
    }
}
