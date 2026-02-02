using RobotScripts;
using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Trajectory controller with PD control for smooth robot motion.
    /// Eliminates oscillation by adding velocity feedback (damping) to position control.
    ///
    /// Key Features:
    /// - PD control law: correction = K_p*(pos error) + K_d*(vel error)
    /// - Synchronized with FixedUpdate to avoid Update/FixedUpdate jitter
    /// - Feedforward terms from velocity profile for smooth acceleration
    ///
    /// Usage:
    /// - Call GetTrajectoryState() in FixedUpdate to get desired state
    /// - Call ComputeCartesianCorrection() to get damped correction vector
    /// </summary>
    public class TrajectoryController
    {
        // PD gains for Cartesian space control
        private Vector3 _positionGains;
        private Vector3 _velocityGains;
        private float _maxVelocity;
        private float _maxAcceleration;

        // Cached trajectory state (synchronized with FixedUpdate)
        private Vector3 _cachedTargetPosition;
        private Vector3 _cachedTargetVelocity;
        private Vector3 _cachedTargetAcceleration;
        private float _lastUpdateTime = -1f;

        /// <summary>
        /// Creates a new trajectory controller with specified PD gains
        /// </summary>
        /// <param name="positionGains">Position gain (K_p) per axis</param>
        /// <param name="velocityGains">Velocity gain (K_d) per axis for damping</param>
        public TrajectoryController(Vector3? positionGains = null, Vector3? velocityGains = null, float? maxVelocity = null, float? maxAcceleration = null)
        {
            _positionGains = positionGains ?? new Vector3(10f, 10f, 10f);
            _velocityGains = velocityGains ?? new Vector3(2f, 2f, 2f);
            _maxVelocity = maxVelocity ?? 0.5f;
            _maxAcceleration = maxAcceleration ?? 1.0f;
        }

        /// <summary>
        /// Set PD gains for trajectory tracking
        /// </summary>
        public void SetGains(Vector3 positionGains, Vector3 velocityGains)
        {
            _positionGains = positionGains;
            _velocityGains = velocityGains;
        }

        /// <summary>
        /// Get trajectory state at specified time.
        /// CRITICAL: Must be called in FixedUpdate to avoid jitter.
        /// Caches result for use between FixedUpdate calls.
        /// </summary>
        /// <param name="currentTime">Time along trajectory (from trajectory start)</param>
        /// <param name="path">Cartesian path being followed</param>
        /// <param name="velocityProfile">Velocity profile for the path</param>
        /// <returns>Target position, velocity, and acceleration</returns>
        public (Vector3 targetPos, Vector3 targetVel, Vector3 targetAccel) GetTrajectoryState(
            float currentTime,
            CartesianPath path,
            VelocityProfile velocityProfile
        )
        {
            // CRITICAL: Cache trajectory state in FixedUpdate
            // If called in Update(), trajectory will jitter relative to FixedUpdate()
            // Only recompute if time has changed (i.e., new FixedUpdate frame)
            if (Mathf.Abs(currentTime - _lastUpdateTime) > 0.001f)
            {
                _lastUpdateTime = currentTime;

                float distance = CalculateDistanceFromTime(currentTime, velocityProfile);
                float velocity = velocityProfile.GetVelocityAtDistance(distance);

                distance = Mathf.Clamp(distance, 0f, path.totalDistance);

                CartesianWaypoint waypoint = path.GetWaypointAtDistance(distance);

                Vector3 direction = GetPathTangent(path, distance);

                _cachedTargetPosition = waypoint.position;
                Vector3 rawVelocity = direction * velocity;

                if (rawVelocity.magnitude > _maxVelocity)
                {
                    rawVelocity = rawVelocity.normalized * _maxVelocity;
                }
                _cachedTargetVelocity = rawVelocity;

                _cachedTargetAcceleration = GetAccelerationFromProfile(
                    velocityProfile,
                    currentTime,
                    distance
                );
            }

            return (_cachedTargetPosition, _cachedTargetVelocity, _cachedTargetAcceleration);
        }

        /// <summary>
        /// Calculate distance traveled at given time along trapezoidal velocity profile.
        /// Matches the calculation in RobotController.
        /// </summary>
        private float CalculateDistanceFromTime(float time, VelocityProfile profile)
        {
            if (profile == null)
                return 0f;

            float a = profile.acceleration;
            float vMax = profile.cruiseVelocity;
            float tAccel = vMax / a;

            if (time <= tAccel)
            {
                return 0.5f * a * time * time;
            }
            else if (profile.cruisePhaseDistance > 0f)
            {
                float tCruiseEnd = tAccel + (profile.cruisePhaseDistance / vMax);
                if (time <= tCruiseEnd)
                {
                    return profile.accelerationPhaseDistance + vMax * (time - tAccel);
                }
                else
                {
                    float tDecel = time - tCruiseEnd;
                    return profile.accelerationPhaseDistance
                        + profile.cruisePhaseDistance
                        + (vMax * tDecel - 0.5f * a * tDecel * tDecel);
                }
            }
            else
            {
                float tTotal = 2f * tAccel;
                if (time <= tAccel)
                {
                    return 0.5f * a * time * time;
                }
                else if (time < tTotal)
                {
                    float tDecel = time - tAccel;
                    return profile.accelerationPhaseDistance
                        + (vMax * tDecel - 0.5f * a * tDecel * tDecel);
                }
                else
                {
                    return profile.accelerationPhaseDistance + profile.decelerationPhaseDistance;
                }
            }
        }

        /// <summary>
        /// Compute Cartesian correction using PD control law.
        /// This is the "secret sauce" that eliminates oscillation.
        /// </summary>
        /// <param name="currentPos">Current end effector position</param>
        /// <param name="targetPos">Target position from trajectory</param>
        /// <param name="currentVel">Current end effector velocity (from ArticulationBody)</param>
        /// <param name="targetVel">Target velocity from trajectory</param>
        /// <returns>Correction vector to apply</returns>
        public Vector3 ComputeCartesianCorrection(
            Vector3 currentPos,
            Vector3 targetPos,
            Vector3 currentVel,
            Vector3 targetVel
        )
        {
            Vector3 posError = targetPos - currentPos;
            Vector3 velError = targetVel - currentVel;

            Vector3 posCorrection = Vector3.Scale(_positionGains, posError);
            Vector3 velCorrection = Vector3.Scale(_velocityGains, velError);

            return posCorrection + velCorrection;
        }

        /// <summary>
        /// Get tangent direction to path at specified distance
        /// </summary>
        private Vector3 GetPathTangent(CartesianPath path, float distance)
        {
            if (path.waypoints.Count < 2)
                return Vector3.forward;

            // Find segment containing this distance
            for (int i = 0; i < path.waypoints.Count - 1; i++)
            {
                float d1 = path.waypoints[i].distanceFromStart;
                float d2 = path.waypoints[i + 1].distanceFromStart;

                if (distance >= d1 && distance <= d2)
                {
                    Vector3 p1 = path.waypoints[i].position;
                    Vector3 p2 = path.waypoints[i + 1].position;
                    return (p2 - p1).normalized;
                }
            }

            // Default to direction of last segment
            int lastIdx = path.waypoints.Count - 1;
            return (
                path.waypoints[lastIdx].position - path.waypoints[lastIdx - 1].position
            ).normalized;
        }

        /// <summary>
        /// Get acceleration from velocity profile at specified time/distance
        /// </summary>
        private Vector3 GetAccelerationFromProfile(
            VelocityProfile profile,
            float time,
            float distance
        )
        {
            if (profile == null)
                return Vector3.zero;

            float accelScalar = 0f;

            // Determine phase based on distance
            if (distance < profile.accelerationPhaseDistance)
            {
                accelScalar = profile.acceleration;
            }
            else if (distance >= profile.accelerationPhaseDistance + profile.cruisePhaseDistance)
            {
                accelScalar = -profile.acceleration;
            }

            accelScalar = Mathf.Clamp(accelScalar, -_maxAcceleration, _maxAcceleration);

            return Vector3.zero;
        }

        /// <summary>
        /// Reset cached state (call when starting new trajectory)
        /// </summary>
        public void Reset()
        {
            _lastUpdateTime = -1f;
            _cachedTargetPosition = Vector3.zero;
            _cachedTargetVelocity = Vector3.zero;
            _cachedTargetAcceleration = Vector3.zero;
        }

        /// <summary>
        /// Get current cached target velocity (for external use)
        /// </summary>
        public Vector3 GetCachedTargetVelocity()
        {
            return _cachedTargetVelocity;
        }
    }
}
