using System.Collections.Generic;
using System.Linq;
using Robotics;
using UnityEngine;

namespace Simulation.CoordinationStrategies
{
    /// <summary>
    /// Waypoint-based collision avoidance planner.
    /// Generates intermediate waypoints to avoid obstacles using three strategies:
    /// 1. Vertical offset - lift up, move over, descend
    /// 2. Lateral offset - move around obstacle (try both sides)
    /// 3. Combined offset - vertical + lateral if needed
    /// </summary>
    public class WaypointCollisionAvoidancePlanner : ICollisionAvoidancePlanner
    {
        private float _verticalOffset;
        private float _lateralOffset;
        private float _minSafeSeparation;
        private int _maxWaypoints;

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
        /// Tries three strategies in order: vertical, lateral, combined.
        /// </summary>
        public List<Vector3> PlanAlternativePath(
            string robotId,
            Vector3 current,
            Vector3 target,
            List<Vector3> obstacles
        )
        {
            if (obstacles == null || obstacles.Count == 0)
            {
                return new List<Vector3> { target };
            }

            Debug.Log($"{LOG_PREFIX} Planning alternative path for {robotId} with {obstacles.Count} obstacles");

            // Strategy 1: Vertical offset (lift up, move over, descend)
            var verticalPath = TryVerticalOffset(current, target, obstacles);
            if (verticalPath != null && IsPathClear(verticalPath, obstacles))
            {
                Debug.Log($"{LOG_PREFIX} Vertical offset path found for {robotId}");
                return verticalPath;
            }

            // Strategy 2: Lateral offset (move around obstacle - try both sides)
            var lateralPath = TryLateralOffset(current, target, obstacles);
            if (lateralPath != null && IsPathClear(lateralPath, obstacles))
            {
                Debug.Log($"{LOG_PREFIX} Lateral offset path found for {robotId}");
                return lateralPath;
            }

            // Strategy 3: Combined offset (vertical + lateral)
            var combinedPath = TryCombinedOffset(current, target, obstacles);
            if (combinedPath != null && IsPathClear(combinedPath, obstacles))
            {
                Debug.Log($"{LOG_PREFIX} Combined offset path found for {robotId}");
                return combinedPath;
            }

            // No valid path found
            Debug.LogWarning($"{LOG_PREFIX} No alternative path found for {robotId}");
            return new List<Vector3>();
        }

        /// <summary>
        /// Check if replanning is required for a robot's movement.
        /// </summary>
        public bool RequiresReplanning(string robotId, Vector3 target, RobotController[] otherRobots)
        {
            if (otherRobots == null || otherRobots.Length == 0)
                return false;

            // Check if target is too close to any other robot's current or target position
            foreach (var otherRobot in otherRobots)
            {
                if (otherRobot.robotId == robotId)
                    continue;

                // Check distance to other robot's current position
                Vector3 otherPos = otherRobot.GetCurrentEndEffectorPosition();
                if (Vector3.Distance(target, otherPos) < _minSafeSeparation)
                {
                    Debug.Log($"{LOG_PREFIX} {robotId} target conflicts with {otherRobot.robotId} position");
                    return true;
                }

                // Check distance to other robot's target (if has target)
                if (otherRobot.HasTarget)
                {
                    var otherTarget = otherRobot.GetCurrentTarget();
                    if (otherTarget.HasValue &&
                        Vector3.Distance(target, otherTarget.Value) < _minSafeSeparation)
                    {
                        Debug.Log($"{LOG_PREFIX} {robotId} target conflicts with {otherRobot.robotId} target");
                        return true;
                    }
                }
            }

            return false;
        }

        /// <summary>
        /// Try vertical offset strategy: lift up, move over, descend.
        /// </summary>
        private List<Vector3> TryVerticalOffset(Vector3 current, Vector3 target, List<Vector3> obstacles)
        {
            var path = new List<Vector3>();

            // Waypoint 1: Lift up
            Vector3 liftPoint = current + Vector3.up * _verticalOffset;
            path.Add(liftPoint);

            // Waypoint 2: Move over (at elevated height)
            Vector3 overPoint = target + Vector3.up * _verticalOffset;
            path.Add(overPoint);

            // Waypoint 3: Descend to target
            path.Add(target);

            return path.Count <= _maxWaypoints ? path : null;
        }

        /// <summary>
        /// Try lateral offset strategy: move around obstacle (try both sides).
        /// </summary>
        private List<Vector3> TryLateralOffset(Vector3 current, Vector3 target, List<Vector3> obstacles)
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
        /// Try lateral offset on a specific side.
        /// </summary>
        private List<Vector3> TryLateralSide(
            Vector3 current,
            Vector3 target,
            Vector3 offsetDirection,
            List<Vector3> obstacles
        )
        {
            var path = new List<Vector3>();

            // Waypoint 1: Move to side
            Vector3 sidePoint = current + offsetDirection * _lateralOffset;
            path.Add(sidePoint);

            // Waypoint 2: Move along side
            Vector3 targetSidePoint = target + offsetDirection * _lateralOffset;
            path.Add(targetSidePoint);

            // Waypoint 3: Move to target
            path.Add(target);

            return path.Count <= _maxWaypoints ? path : null;
        }

        /// <summary>
        /// Try combined offset strategy: vertical + lateral.
        /// </summary>
        private List<Vector3> TryCombinedOffset(Vector3 current, Vector3 target, List<Vector3> obstacles)
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
        /// Try combined offset on a specific side.
        /// </summary>
        private List<Vector3> TryCombinedSide(
            Vector3 current,
            Vector3 target,
            Vector3 offsetDirection,
            List<Vector3> obstacles
        )
        {
            var path = new List<Vector3>();

            // Waypoint 1: Lift and move to side
            Vector3 liftSidePoint = current + offsetDirection * _lateralOffset + Vector3.up * _verticalOffset;
            path.Add(liftSidePoint);

            // Waypoint 2: Move along side (elevated)
            Vector3 targetLiftSidePoint = target + offsetDirection * _lateralOffset + Vector3.up * _verticalOffset;
            path.Add(targetLiftSidePoint);

            // Waypoint 3: Move back to target line (still elevated)
            Vector3 targetLiftPoint = target + Vector3.up * _verticalOffset;
            path.Add(targetLiftPoint);

            // Waypoint 4: Descend to target
            path.Add(target);

            return path.Count <= _maxWaypoints ? path : null;
        }

        /// <summary>
        /// Check if a path is clear of obstacles.
        /// </summary>
        private bool IsPathClear(List<Vector3> path, List<Vector3> obstacles)
        {
            if (obstacles == null || obstacles.Count == 0)
                return true;

            // Check each segment of the path
            for (int i = 0; i < path.Count - 1; i++)
            {
                Vector3 segmentStart = path[i];
                Vector3 segmentEnd = path[i + 1];

                // Check if segment passes too close to any obstacle
                foreach (var obstacle in obstacles)
                {
                    float distToSegment = DistanceToSegment(obstacle, segmentStart, segmentEnd);
                    if (distToSegment < _minSafeSeparation)
                    {
                        return false;
                    }
                }
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
                // Degenerate segment
                return Vector3.Distance(point, segmentStart);
            }

            Vector3 segmentDirection = segmentVector / segmentLength;
            float projection = Vector3.Dot(pointVector, segmentDirection);

            // Clamp projection to segment bounds
            projection = Mathf.Clamp(projection, 0f, segmentLength);

            // Find closest point on segment
            Vector3 closestPoint = segmentStart + segmentDirection * projection;

            return Vector3.Distance(point, closestPoint);
        }
    }
}
