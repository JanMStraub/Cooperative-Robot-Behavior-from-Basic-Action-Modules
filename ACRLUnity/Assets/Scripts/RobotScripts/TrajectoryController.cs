using RobotScripts;
using UnityEngine;

namespace Robotics
{
    /// <summary>PD trajectory controller: position + velocity feedback for oscillation-free Cartesian motion.</summary>
    public class TrajectoryController
    {
        private Vector3 _positionGains;
        private Vector3 _velocityGains;
        private float _maxVelocity;
        private float _maxAcceleration;

        // Cached per-FixedUpdate to avoid jitter from Update() calls
        private Vector3 _cachedTargetPosition;
        private Vector3 _cachedTargetVelocity;
        private Vector3 _cachedTargetAcceleration;
        private float _lastUpdateTime = -1f;

        public TrajectoryController(
            Vector3? positionGains = null,
            Vector3? velocityGains = null,
            float? maxVelocity = null,
            float? maxAcceleration = null
        )
        {
            _positionGains = positionGains ?? new Vector3(10f, 10f, 10f);
            _velocityGains = velocityGains ?? new Vector3(2f, 2f, 2f);
            _maxVelocity = maxVelocity ?? 0.5f;
            _maxAcceleration = maxAcceleration ?? 1.0f;
        }

        public void SetGains(Vector3 positionGains, Vector3 velocityGains)
        {
            _positionGains = positionGains;
            _velocityGains = velocityGains;
        }

        /// <summary>Sample trajectory at current time. Call from FixedUpdate only — result is cached per-frame.</summary>
        public (Vector3 targetPos, Vector3 targetVel, Vector3 targetAccel) GetTrajectoryState(
            float currentTime,
            CartesianPath path,
            VelocityProfile velocityProfile
        )
        {
            if (Mathf.Abs(currentTime - _lastUpdateTime) > 0.001f)
            {
                if (path == null || velocityProfile == null)
                    return (Vector3.zero, Vector3.zero, Vector3.zero);

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
                    distance,
                    direction
                );
            }

            return (_cachedTargetPosition, _cachedTargetVelocity, _cachedTargetAcceleration);
        }

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

        private Vector3 GetPathTangent(CartesianPath path, float distance)
        {
            if (path.waypoints.Count < 2)
                return Vector3.forward;

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

            int lastIdx = path.waypoints.Count - 1;
            return (
                path.waypoints[lastIdx].position - path.waypoints[lastIdx - 1].position
            ).normalized;
        }

        private Vector3 GetAccelerationFromProfile(
            VelocityProfile profile,
            float time,
            float distance,
            Vector3 direction
        )
        {
            if (profile == null)
                return Vector3.zero;

            float accelScalar = 0f;

            if (distance < profile.accelerationPhaseDistance)
            {
                accelScalar = profile.acceleration;
            }
            else if (distance >= profile.accelerationPhaseDistance + profile.cruisePhaseDistance)
            {
                accelScalar = -profile.acceleration;
            }

            accelScalar = Mathf.Clamp(accelScalar, -_maxAcceleration, _maxAcceleration);

            return direction * accelScalar;
        }

        public void Reset()
        {
            _lastUpdateTime = -1f;
            _cachedTargetPosition = Vector3.zero;
            _cachedTargetVelocity = Vector3.zero;
            _cachedTargetAcceleration = Vector3.zero;
        }

        public Vector3 GetCachedTargetVelocity()
        {
            return _cachedTargetVelocity;
        }
    }
}
