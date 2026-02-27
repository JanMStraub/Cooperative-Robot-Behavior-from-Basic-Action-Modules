using System;
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
                    distanceFromStart = 0f,
                };
            }

            distance = Mathf.Clamp(distance, 0f, totalDistance);

            for (int i = 0; i < waypoints.Count - 1; i++)
            {
                CartesianWaypoint current = waypoints[i];
                CartesianWaypoint next = waypoints[i + 1];

                if (distance >= current.distanceFromStart && distance <= next.distanceFromStart)
                {
                    float segmentDistance = next.distanceFromStart - current.distanceFromStart;
                    if (segmentDistance < 0.0001f)
                    {
                        return current;
                    }

                    float t = (distance - current.distanceFromStart) / segmentDistance;

                    return new CartesianWaypoint
                    {
                        position = Vector3.Lerp(current.position, next.position, t),
                        rotation = Quaternion.Slerp(current.rotation, next.rotation, t),
                        distanceFromStart = distance,
                    };
                }
            }

            return waypoints[waypoints.Count - 1];
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
            if (acceleration <= 0f)
            {
                throw new ArgumentException(
                    $"[VelocityProfile] acceleration must be positive, got {acceleration}"
                );
            }

            if (maxVelocity <= 0f)
            {
                throw new ArgumentException(
                    $"[VelocityProfile] maxVelocity must be positive, got {maxVelocity}"
                );
            }

            float accelTime = maxVelocity / acceleration;
            float accelDistance = 0.5f * acceleration * accelTime * accelTime;

            VelocityProfile profile = new VelocityProfile { acceleration = acceleration };

            if (accelDistance * 2 > totalDistance)
            {
                accelDistance = totalDistance / 2f;
                float peakVelocity = Mathf.Sqrt(2f * acceleration * accelDistance);

                profile.accelerationPhaseDistance = accelDistance;
                profile.cruisePhaseDistance = 0f;
                profile.decelerationPhaseDistance = accelDistance;
                profile.cruiseVelocity = peakVelocity;
            }
            else
            {
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
                return Mathf.Sqrt(2f * acceleration * distance);
            }
            else if (distance < accelerationPhaseDistance + cruisePhaseDistance)
            {
                return cruiseVelocity;
            }
            else
            {
                float totalNonDecelDistance = accelerationPhaseDistance + cruisePhaseDistance;
                float decelDistance = distance - totalNonDecelDistance;
                float remainingDistance = decelerationPhaseDistance - decelDistance;

                if (remainingDistance <= 0f)
                {
                    return 0f;
                }

                return Mathf.Sqrt(2f * acceleration * remainingDistance);
            }
        }
    }
}
