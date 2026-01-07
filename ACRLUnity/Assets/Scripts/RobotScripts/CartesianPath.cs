using System.Collections.Generic;
using UnityEngine;

namespace RobotScripts
{
    /// <summary>
    /// Represents a Cartesian path through 3D space with position and orientation waypoints.
    /// Used for smooth straight-line motion planning.
    /// </summary>
    public class CartesianPath
    {
        public List<CartesianWaypoint> waypoints;
        public float totalDistance;
        public float maxVelocity;
        public float acceleration;

        /// <summary>
        /// Get the waypoint at a specific distance along the path.
        /// Interpolates between waypoints if distance falls between them.
        /// </summary>
        public CartesianWaypoint GetWaypointAtDistance(float distance)
        {
            if (waypoints == null || waypoints.Count == 0)
            {
                return new CartesianWaypoint
                {
                    position = Vector3.zero,
                    rotation = Quaternion.identity,
                    distanceFromStart = 0f
                };
            }

            // Clamp distance to path bounds
            distance = Mathf.Clamp(distance, 0f, totalDistance);

            // Find the two waypoints that bracket this distance
            for (int i = 0; i < waypoints.Count - 1; i++)
            {
                CartesianWaypoint current = waypoints[i];
                CartesianWaypoint next = waypoints[i + 1];

                if (distance >= current.distanceFromStart && distance <= next.distanceFromStart)
                {
                    // Interpolate between current and next
                    float segmentDistance = next.distanceFromStart - current.distanceFromStart;
                    if (segmentDistance < 0.0001f)
                    {
                        return current;  // Avoid division by zero
                    }

                    float t = (distance - current.distanceFromStart) / segmentDistance;

                    return new CartesianWaypoint
                    {
                        position = Vector3.Lerp(current.position, next.position, t),
                        rotation = Quaternion.Slerp(current.rotation, next.rotation, t),
                        distanceFromStart = distance
                    };
                }
            }

            // If we're at or past the end, return the last waypoint
            return waypoints[waypoints.Count - 1];
        }

        /// <summary>
        /// Get position at a specific time along the path (for future velocity-based queries).
        /// </summary>
        public Vector3 GetPositionAtTime(float time)
        {
            // For now, uses simple constant velocity assumption
            // Can be enhanced with velocity profile integration later
            float distance = time * maxVelocity;
            return GetWaypointAtDistance(distance).position;
        }

        /// <summary>
        /// Get rotation at a specific time along the path (for future velocity-based queries).
        /// </summary>
        public Quaternion GetRotationAtTime(float time)
        {
            // For now, uses simple constant velocity assumption
            // Can be enhanced with velocity profile integration later
            float distance = time * maxVelocity;
            return GetWaypointAtDistance(distance).rotation;
        }
    }

    /// <summary>
    /// Represents a single waypoint along a Cartesian path.
    /// </summary>
    public struct CartesianWaypoint
    {
        public Vector3 position;
        public Quaternion rotation;
        public float distanceFromStart;
        public float timeFromStart;
    }

    /// <summary>
    /// Velocity profile for smooth acceleration and deceleration along a path.
    /// Supports trapezoidal (accel -> cruise -> decel) and triangular (accel -> decel) profiles.
    /// </summary>
    public class VelocityProfile
    {
        public float accelerationPhaseDistance;
        public float cruisePhaseDistance;
        public float decelerationPhaseDistance;
        public float cruiseVelocity;
        public float acceleration;

        /// <summary>
        /// Create a trapezoidal velocity profile for smooth motion.
        /// Automatically handles short distances by creating triangular profiles.
        /// </summary>
        /// <param name="totalDistance">Total path distance in meters</param>
        /// <param name="maxVelocity">Maximum velocity in m/s</param>
        /// <param name="acceleration">Acceleration/deceleration in m/s²</param>
        public static VelocityProfile CreateTrapezoidal(
            float totalDistance,
            float maxVelocity,
            float acceleration
        )
        {
            // Calculate time and distance for acceleration phase
            float accelTime = maxVelocity / acceleration;
            float accelDistance = 0.5f * acceleration * accelTime * accelTime;

            VelocityProfile profile = new VelocityProfile
            {
                acceleration = acceleration
            };

            // Check if we can reach max velocity
            if (accelDistance * 2 > totalDistance)
            {
                // Triangular profile (no cruise phase) - too short to reach max velocity
                accelDistance = totalDistance / 2f;
                float peakVelocity = Mathf.Sqrt(2f * acceleration * accelDistance);

                profile.accelerationPhaseDistance = accelDistance;
                profile.cruisePhaseDistance = 0f;
                profile.decelerationPhaseDistance = accelDistance;
                profile.cruiseVelocity = peakVelocity;
            }
            else
            {
                // Trapezoidal profile (has cruise phase)
                profile.accelerationPhaseDistance = accelDistance;
                profile.cruisePhaseDistance = totalDistance - (2f * accelDistance);
                profile.decelerationPhaseDistance = accelDistance;
                profile.cruiseVelocity = maxVelocity;
            }

            return profile;
        }

        /// <summary>
        /// Get the velocity at a specific distance along the path.
        /// Returns smooth velocity based on the trapezoidal/triangular profile.
        /// </summary>
        /// <param name="distance">Distance from start in meters</param>
        /// <returns>Velocity at that distance in m/s</returns>
        public float GetVelocityAtDistance(float distance)
        {
            if (distance < accelerationPhaseDistance)
            {
                // Acceleration phase: v = sqrt(2 * a * d)
                return Mathf.Sqrt(2f * acceleration * distance);
            }
            else if (distance < accelerationPhaseDistance + cruisePhaseDistance)
            {
                // Cruise phase: constant velocity
                return cruiseVelocity;
            }
            else
            {
                // Deceleration phase
                float totalNonDecelDistance = accelerationPhaseDistance + cruisePhaseDistance;
                float decelDistance = distance - totalNonDecelDistance;
                float remainingDistance = decelerationPhaseDistance - decelDistance;

                // Ensure we don't go negative
                if (remainingDistance <= 0f)
                {
                    return 0f;
                }

                return Mathf.Sqrt(2f * acceleration * remainingDistance);
            }
        }
    }
}
