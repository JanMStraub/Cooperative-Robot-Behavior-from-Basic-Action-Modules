using System.Collections.Generic;
using System.Linq;
using Robotics;
using UnityEngine;

namespace Simulation.CoordinationStrategies
{
    /// <summary>
    /// Waypoint-based collision avoidance planner.
    /// </summary>
    public class WaypointCollisionAvoidancePlanner : ICollisionAvoidancePlanner
    {
        private float _verticalOffset;
        private float _lateralOffset;
        private float _minSafeSeparation;
        private int _maxWaypoints;

        private readonly List<Vector3> _pathBuffer = new();

        private const string LOG_PREFIX = "[WAYPOINT_PLANNER]";

        /// <summary>
        /// Constructor with configurable parameters.
        /// </summary>
        /// <param name="verticalOffset">Vertical clearance offset (default 0.15m)</param>
        /// <param name="lateralOffset">Lateral avoidance offset (default 0.1m)</param>
        /// <param name="minSafeSeparation">Minimum safe separation from obstacles (default 0.2m)</param>
        /// <param name="maxWaypoints">Maximum waypoints per path (default 5)</param>
        public WaypointCollisionAvoidancePlanner(
            float verticalOffset = 0.15f,
            float lateralOffset = 0.1f,
            float minSafeSeparation = 0.2f,
            int maxWaypoints = 5
        )
        {
            _verticalOffset = verticalOffset;
            _lateralOffset = lateralOffset;
            _minSafeSeparation = minSafeSeparation;
            _maxWaypoints = maxWaypoints;
        }

        /// <summary>
        /// Plan an alternative path that avoids obstacles.
        /// </summary>
        public List<Vector3> PlanAlternativePath(
            string robotId,
            Vector3 current,
            Vector3 target,
            List<Vector3> obstacles
        )
        {
            if (Vector3.Distance(current, target) < 0.001f)
            {
                return null;
            }

            if (obstacles == null || obstacles.Count == 0)
            {
                return new List<Vector3> { target };
            }

            Debug.Log(
                $"{LOG_PREFIX} Planning alternative path for {robotId} with {obstacles.Count} obstacles"
            );

            var verticalPath = TryVerticalOffset(current, target, obstacles);
            if (verticalPath != null && IsPathClear(verticalPath, obstacles, current))
            {
                Debug.Log($"{LOG_PREFIX} Vertical offset path found for {robotId}");
                return verticalPath;
            }

            var lateralPath = TryLateralOffset(current, target, obstacles);
            if (lateralPath != null && IsPathClear(lateralPath, obstacles, current))
            {
                Debug.Log($"{LOG_PREFIX} Lateral offset path found for {robotId}");
                return lateralPath;
            }

            var combinedPath = TryCombinedOffset(current, target, obstacles);
            if (combinedPath != null && IsPathClear(combinedPath, obstacles, current))
            {
                Debug.Log($"{LOG_PREFIX} Combined offset path found for {robotId}");
                return combinedPath;
            }

            Debug.LogWarning($"{LOG_PREFIX} No alternative path found for {robotId}");
            return new List<Vector3>();
        }

        /// <summary>
        /// Check if replanning is required for a robot's movement.
        /// </summary>
        public bool RequiresReplanning(
            string robotId,
            Vector3 target,
            RobotController[] otherRobots
        )
        {
            if (otherRobots == null || otherRobots.Length == 0)
                return false;

            Vector3? currentPosition = null;

            foreach (var robot in otherRobots)
            {
                if (robot != null && robot.robotId == robotId)
                {
                    currentPosition = robot.GetCurrentEndEffectorPosition();
                    break;
                }
            }

            if (!currentPosition.HasValue)
            {
                Debug.LogWarning(
                    $"{LOG_PREFIX} Could not find robot {robotId} to check replanning"
                );
                return false;
            }

            foreach (var otherRobot in otherRobots)
            {
                if (otherRobot == null || otherRobot.robotId == robotId)
                    continue;

                Vector3 otherPos = otherRobot.GetCurrentEndEffectorPosition();
                if (otherPos == Vector3.zero)
                {
                    otherPos = otherRobot.transform.position;
                }

                float distanceToPath = DistanceToSegment(otherPos, currentPosition.Value, target);

                if (distanceToPath < _minSafeSeparation)
                {
                    Debug.Log(
                        $"{LOG_PREFIX} {robotId} path blocked by {otherRobot.robotId} (distance: {distanceToPath:F3}m)"
                    );
                    return true;
                }

                if (Vector3.Distance(target, otherPos) < _minSafeSeparation)
                {
                    Debug.Log(
                        $"{LOG_PREFIX} {robotId} target conflicts with {otherRobot.robotId} position"
                    );
                    return true;
                }

                if (otherRobot.HasTarget)
                {
                    var otherTarget = otherRobot.GetCurrentTarget();
                    if (
                        otherTarget.HasValue
                        && Vector3.Distance(target, otherTarget.Value) < _minSafeSeparation
                    )
                    {
                        Debug.Log(
                            $"{LOG_PREFIX} {robotId} target conflicts with {otherRobot.robotId} target"
                        );
                        return true;
                    }
                }
            }

            return false;
        }

        /// <summary>
        /// Try vertical offset strategy using buffer pattern.
        /// </summary>
        private List<Vector3> TryVerticalOffset(
            Vector3 current,
            Vector3 target,
            List<Vector3> obstacles
        )
        {
            _pathBuffer.Clear();

            float safeVerticalOffset = Mathf.Max(_verticalOffset, _minSafeSeparation * 1.1f);

            Vector3 liftPoint = current + Vector3.up * safeVerticalOffset;
            _pathBuffer.Add(liftPoint);

            Vector3 overPoint = target + Vector3.up * safeVerticalOffset;
            _pathBuffer.Add(overPoint);

            _pathBuffer.Add(target);

            if (_pathBuffer.Count > _maxWaypoints)
                return null;

            return new List<Vector3>(_pathBuffer);
        }

        /// <summary>
        /// Try lateral offset strategy: move around obstacle (try both sides).
        /// </summary>
        private List<Vector3> TryLateralOffset(
            Vector3 current,
            Vector3 target,
            List<Vector3> obstacles
        )
        {
            // Calculate direction perpendicular to movement
            Vector3 moveDirection = (target - current).normalized;
            Vector3 perpendicular = Vector3.Cross(moveDirection, Vector3.up).normalized;

            // Try right side first
            var rightPath = TryLateralSide(current, target, perpendicular, obstacles);
            if (rightPath != null)
            {
                return rightPath;
            }

            // Try left side
            var leftPath = TryLateralSide(current, target, -perpendicular, obstacles);
            return leftPath;
        }

        /// <summary>
        /// Try lateral offset on a specific side using buffer pattern.
        /// </summary>
        private List<Vector3> TryLateralSide(
            Vector3 current,
            Vector3 target,
            Vector3 offsetDirection,
            List<Vector3> obstacles
        )
        {
            _pathBuffer.Clear();

            Vector3 sidePoint = current + offsetDirection * _lateralOffset;
            _pathBuffer.Add(sidePoint);

            Vector3 targetSidePoint = target + offsetDirection * _lateralOffset;
            _pathBuffer.Add(targetSidePoint);

            _pathBuffer.Add(target);

            if (_pathBuffer.Count > _maxWaypoints)
                return null;

            return new List<Vector3>(_pathBuffer);
        }

        /// <summary>
        /// Try combined offset strategy: vertical + lateral.
        /// </summary>
        private List<Vector3> TryCombinedOffset(
            Vector3 current,
            Vector3 target,
            List<Vector3> obstacles
        )
        {
            // Calculate direction perpendicular to movement
            Vector3 moveDirection = (target - current).normalized;
            Vector3 perpendicular = Vector3.Cross(moveDirection, Vector3.up).normalized;

            // Try right side with vertical offset
            var rightPath = TryCombinedSide(current, target, perpendicular, obstacles);
            if (rightPath != null)
            {
                return rightPath;
            }

            // Try left side with vertical offset
            var leftPath = TryCombinedSide(current, target, -perpendicular, obstacles);
            return leftPath;
        }

        /// <summary>
        /// Try combined offset on a specific side using buffer pattern.
        /// </summary>
        private List<Vector3> TryCombinedSide(
            Vector3 current,
            Vector3 target,
            Vector3 offsetDirection,
            List<Vector3> obstacles
        )
        {
            _pathBuffer.Clear();

            Vector3 liftSidePoint =
                current + offsetDirection * _lateralOffset + Vector3.up * _verticalOffset;
            _pathBuffer.Add(liftSidePoint);

            Vector3 targetLiftSidePoint =
                target + offsetDirection * _lateralOffset + Vector3.up * _verticalOffset;
            _pathBuffer.Add(targetLiftSidePoint);

            Vector3 targetLiftPoint = target + Vector3.up * _verticalOffset;
            _pathBuffer.Add(targetLiftPoint);

            _pathBuffer.Add(target);

            if (_pathBuffer.Count > _maxWaypoints)
                return null;

            return new List<Vector3>(_pathBuffer);
        }

        /// <summary>
        /// Check if a path is clear of obstacles.
        /// </summary>
        private bool IsPathClear(
            List<Vector3> path,
            List<Vector3> obstacles,
            Vector3 currentPosition
        )
        {
            if (obstacles == null || obstacles.Count == 0)
                return true;

            if (path == null || path.Count == 0)
                return true;

            Vector3 segmentStart = currentPosition;
            for (int i = 0; i < path.Count; i++)
            {
                Vector3 segmentEnd = path[i];

                foreach (var obstacle in obstacles)
                {
                    float distToSegment = DistanceToSegment(obstacle, segmentStart, segmentEnd);
                    if (distToSegment < _minSafeSeparation)
                    {
                        return false;
                    }
                }

                segmentStart = segmentEnd;
            }

            return true;
        }

        /// <summary>
        /// Calculate distance from a point to a line segment.
        /// </summary>
        private float DistanceToSegment(Vector3 point, Vector3 segmentStart, Vector3 segmentEnd)
        {
            Vector3 segmentVector = segmentEnd - segmentStart;
            Vector3 pointVector = point - segmentStart;

            float segmentLength = segmentVector.magnitude;
            if (segmentLength < 0.0001f)
            {
                return Vector3.Distance(point, segmentStart);
            }

            Vector3 segmentDirection = segmentVector / segmentLength;
            float projection = Vector3.Dot(pointVector, segmentDirection);

            projection = Mathf.Clamp(projection, 0f, segmentLength);

            Vector3 closestPoint = segmentStart + segmentDirection * projection;

            return Vector3.Distance(point, closestPoint);
        }
    }
}
