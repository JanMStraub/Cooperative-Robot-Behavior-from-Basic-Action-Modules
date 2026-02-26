using System.Collections.Generic;
using Configuration;
using Robotics;
using UnityEngine;

namespace Simulation.CoordinationStrategies
{
    /// <summary>
    /// Collaborative coordination strategy using Python verification for safe multi-robot operations.
    ///
    /// This strategy integrates with Python's CoordinationVerifier to ensure safe collaborative
    /// robot movements with collision avoidance, workspace management, and conflict detection.
    ///
    /// Features:
    /// - Workspace-based coordination using WorkspaceManager
    /// - Collision detection and avoidance
    /// - Synchronized multi-robot movements
    /// - Python CoordinationVerifier integration (optional)
    ///
    /// Usage:
    /// Set SimulationConfig.coordinationMode = RobotCoordinationMode.Collaborative
    /// </summary>
    public class CollaborativeStrategy : ICoordinationStrategy
    {
        private WorkspaceManager _workspaceManager;
        private Dictionary<string, Vector3> _plannedTargets = new Dictionary<string, Vector3>();
        private HashSet<string> _activeRobots = new HashSet<string>();
        private HashSet<string> _blockedRobots = new HashSet<string>();
        private float _minSafeSeparation;

        // Pre-allocated GC-free buffers for hot-path operations
        private readonly List<string> _staleRobotBuffer = new List<string>();
        private readonly List<Vector3> _obstacleBuffer = new List<Vector3>(2);

        // Path replanning for collision avoidance
        private ICollisionAvoidancePlanner _collisionPlanner;
        private Dictionary<string, Queue<Vector3>> _robotWaypoints =
            new Dictionary<string, Queue<Vector3>>();

        // Coordination state
        private float _lastCoordinationCheckTime = -1f; // Initialize to -1 to force first check
        private const float COORDINATION_CHECK_INTERVAL = 0.5f; // Check coordination every 500ms

        private const string LOG_PREFIX = "[COLLABORATIVE_STRATEGY]";

        /// <summary>
        /// Constructor - creates strategy with default configuration (used in tests)
        /// </summary>
        public CollaborativeStrategy()
            : this(ScriptableObject.CreateInstance<CoordinationConfig>()) { }

        /// <summary>
        /// Constructor with configurable minimum safe separation
        /// </summary>
        /// <param name="minSafeSeparation">Minimum safe separation distance in meters</param>
        public CollaborativeStrategy(CoordinationConfig config)
        {
            _minSafeSeparation = Mathf.Max(0.05f, config.minSafeSeparation);
            _workspaceManager = WorkspaceManager.Instance;
            if (_workspaceManager == null)
            {
                Debug.LogWarning(
                    $"{LOG_PREFIX} WorkspaceManager not found. Workspace coordination disabled."
                );
            }

            // Initialize collision avoidance planner
            _collisionPlanner = new WaypointCollisionAvoidancePlanner(
                verticalOffset: config.verticalOffset,
                lateralOffset: config.lateralOffset,
                minSafeSeparation: _minSafeSeparation,
                maxWaypoints: config.maxWaypoints
            );
            Debug.Log($"{LOG_PREFIX} Path replanning enabled with waypoint collision avoidance");
        }

        /// <summary>
        /// Updates the collaborative coordination logic.
        /// Checks for collisions and workspace conflicts before allowing robot movements.
        /// </summary>
        public void Update(
            RobotController[] robotControllers,
            Dictionary<string, bool> robotTargetReached
        )
        {
            if (robotControllers == null || robotControllers.Length == 0)
                return;

            // Cleanup stale entries - remove robots that are no longer in the controllers array
            CleanupStaleEntries(robotControllers);

            // Update planned targets
            foreach (var controller in robotControllers)
            {
                if (controller.HasTarget)
                {
                    var currentTarget = controller.GetCurrentTarget();
                    if (currentTarget.HasValue)
                    {
                        _plannedTargets[controller.robotId] = currentTarget.Value;
                        _activeRobots.Add(controller.robotId);
                    }
                }
                else
                {
                    // Robot has no target, remove from active tracking
                    _activeRobots.Remove(controller.robotId);
                    _plannedTargets.Remove(controller.robotId);
                    _blockedRobots.Remove(controller.robotId);
                }
            }

            // Periodic coordination check
            if (Time.time - _lastCoordinationCheckTime > COORDINATION_CHECK_INTERVAL)
            {
                PerformCoordinationCheck(robotControllers);
                _lastCoordinationCheckTime = Time.time;
            }

            // Update workspace allocations based on current robot positions
            UpdateWorkspaceAllocations(robotControllers);
        }

        /// <summary>
        /// Cleanup stale entries for robots that are no longer in the scene.
        /// Uses pre-allocated buffer to avoid per-frame GC allocations.
        /// </summary>
        private void CleanupStaleEntries(RobotController[] robotControllers)
        {
            _staleRobotBuffer.Clear();

            // Identify stale IDs by scanning active set against the controllers array directly
            foreach (string id in _activeRobots)
            {
                bool found = false;
                for (int i = 0; i < robotControllers.Length; i++)
                {
                    if (robotControllers[i].robotId == id)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    _staleRobotBuffer.Add(id);
            }

            // Remove stale entries
            foreach (string robotId in _staleRobotBuffer)
            {
                _activeRobots.Remove(robotId);
                _plannedTargets.Remove(robotId);
                _blockedRobots.Remove(robotId);
            }
        }

        /// <summary>
        /// Perform coordination safety checks for all active robots
        /// </summary>
        private void PerformCoordinationCheck(RobotController[] robotControllers)
        {
            if (robotControllers.Length < 2)
                return; // No coordination needed for single robot

            // Clear blocked robots list - will be rebuilt during this check
            _blockedRobots.Clear();

            // Check all pairs of robots for potential conflicts
            for (int i = 0; i < robotControllers.Length; i++)
            {
                for (int j = i + 1; j < robotControllers.Length; j++)
                {
                    var robot1 = robotControllers[i];
                    var robot2 = robotControllers[j];

                    if (!robot1.HasTarget || !robot2.HasTarget)
                        continue;

                    CheckRobotPairConflict(robot1, robot2);
                }
            }
        }

        /// <summary>
        /// Check for conflicts between two robots and attempt path replanning if needed.
        /// Uses priority system to prevent deadlocks: lower alphabetical ID yields to higher ID.
        /// </summary>
        private void CheckRobotPairConflict(RobotController robot1, RobotController robot2)
        {
            var target1Nullable = robot1.GetCurrentTarget();
            var target2Nullable = robot2.GetCurrentTarget();

            if (!target1Nullable.HasValue || !target2Nullable.HasValue)
            {
                return;
            }

            Vector3 target1 = target1Nullable.Value;
            Vector3 target2 = target2Nullable.Value;
            Vector3 current1 = robot1.GetCurrentEndEffectorPosition();
            Vector3 current2 = robot2.GetCurrentEndEffectorPosition();

            int priority = string.Compare(
                robot1.robotId,
                robot2.robotId,
                System.StringComparison.Ordinal
            );

            // When priority < 0, robot1 comes first alphabetically, so it gets HIGH priority
            // When priority > 0, robot2 comes first alphabetically, so it gets HIGH priority
            RobotController lowPriorityRobot = priority < 0 ? robot2 : robot1;
            RobotController highPriorityRobot = priority < 0 ? robot1 : robot2;
            Vector3 lowPriorityCurrent = priority < 0 ? current2 : current1;
            Vector3 lowPriorityTarget = priority < 0 ? target2 : target1;
            Vector3 highPriorityCurrent = priority < 0 ? current1 : current2;
            Vector3 highPriorityTarget = priority < 0 ? target1 : target2;

            float targetDistance = Vector3.Distance(target1, target2);
            if (targetDistance < _minSafeSeparation)
            {
                Debug.LogWarning(
                    $"{LOG_PREFIX} Collision detected: {robot1.robotId} and {robot2.robotId} targeting same location (distance: {targetDistance:F3}m)"
                );

                _obstacleBuffer.Clear();
                _obstacleBuffer.Add(highPriorityCurrent);
                _obstacleBuffer.Add(highPriorityTarget);
                if (
                    AttemptReplanning(
                        lowPriorityRobot,
                        lowPriorityCurrent,
                        lowPriorityTarget,
                        _obstacleBuffer
                    )
                )
                {
                    Debug.Log(
                        $"{LOG_PREFIX} Replanned path for {lowPriorityRobot.robotId} to avoid {highPriorityRobot.robotId}"
                    );
                    _blockedRobots.Remove(lowPriorityRobot.robotId);
                }
                else
                {
                    Debug.LogWarning(
                        $"{LOG_PREFIX} No alternative path found for {lowPriorityRobot.robotId}, blocking robot"
                    );
                    _blockedRobots.Add(lowPriorityRobot.robotId);
                }
                return;
            }

            // Skip path collision check if either robot isn't actually moving
            bool robot1Moving = Vector3.Distance(current1, target1) > 0.001f;
            bool robot2Moving = Vector3.Distance(current2, target2) > 0.001f;

            if (
                robot1Moving
                && robot2Moving
                && WillPathsCollide(current1, target1, current2, target2)
            )
            {
                Debug.LogWarning(
                    $"{LOG_PREFIX} Path collision detected between {robot1.robotId} and {robot2.robotId}"
                );

                _obstacleBuffer.Clear();
                _obstacleBuffer.Add(highPriorityCurrent);
                _obstacleBuffer.Add(highPriorityTarget);
                if (
                    AttemptReplanning(
                        lowPriorityRobot,
                        lowPriorityCurrent,
                        lowPriorityTarget,
                        _obstacleBuffer
                    )
                )
                {
                    Debug.Log(
                        $"{LOG_PREFIX} Replanned path for {lowPriorityRobot.robotId} to avoid {highPriorityRobot.robotId}"
                    );
                    _blockedRobots.Remove(lowPriorityRobot.robotId);
                }
                else
                {
                    Debug.LogWarning(
                        $"{LOG_PREFIX} No alternative path found for {lowPriorityRobot.robotId}, blocking robot"
                    );
                    _blockedRobots.Add(lowPriorityRobot.robotId);
                }
                return;
            }

            // Check 3: Workspace conflict (entering other robot's workspace)
            if (_workspaceManager != null)
            {
                var region1 = _workspaceManager.GetRegionAtPosition(target1);
                var region2 = _workspaceManager.GetRegionAtPosition(target2);

                if (region1 != null && region2 != null && region1 == region2)
                {
                    // Both robots targeting same region
                    if (region1.regionName != "shared_zone")
                    {
                        Debug.LogWarning(
                            $"{LOG_PREFIX} Workspace conflict: Both robots targeting '{region1.regionName}', warning only"
                        );
                        // Do not block robot - workspace conflicts are advisory only
                    }
                }
            }

            // No conflicts detected between this pair - no action needed
            // Note: We don't unconditionally unblock robots here because they might be
            // blocked due to conflicts with OTHER robots checked in previous iterations
        }

        /// <summary>
        /// Attempt to replan a robot's path to avoid obstacles.
        /// </summary>
        private bool AttemptReplanning(
            RobotController robot,
            Vector3 current,
            Vector3 target,
            List<Vector3> obstacles
        )
        {
            if (_collisionPlanner == null)
                return false;

            // Generate alternative path
            var waypoints = _collisionPlanner.PlanAlternativePath(
                robot.robotId,
                current,
                target,
                obstacles
            );

            if (waypoints == null || waypoints.Count == 0)
            {
                return false;
            }

            // Store waypoints for this robot
            _robotWaypoints[robot.robotId] = new Queue<Vector3>(waypoints);
            Debug.Log($"{LOG_PREFIX} Generated {waypoints.Count} waypoints for {robot.robotId}");

            return true;
        }

        /// <summary>
        /// Check if two robot paths will collide
        /// Uses simple swept sphere collision detection
        /// </summary>
        private bool WillPathsCollide(Vector3 start1, Vector3 end1, Vector3 start2, Vector3 end2)
        {
            // Check if robots share the same starting position
            bool sameStart = Vector3.Distance(start1, start2) < 0.001f;

            if (sameStart)
            {
                // If starting from same position, check if moving in diverging directions
                Vector3 dir1 = (end1 - start1).normalized;
                Vector3 dir2 = (end2 - start2).normalized;

                // If directions are significantly different (angle > 30 degrees), no collision
                float dotProduct = Vector3.Dot(dir1, dir2);
                if (dotProduct < 0.866f) // cos(30°) ≈ 0.866
                {
                    return false; // Diverging paths from same start
                }

                // Moving in similar directions from same start - check if one target is much farther
                // This handles the case where robots move along the same axis but to different distances
                float dist1 = Vector3.Distance(start1, end1);
                float dist2 = Vector3.Distance(start2, end2);

                // If one robot is moving significantly farther (>2x distance), consider them non-colliding
                // The farther robot will naturally pass by the closer one
                if (dist1 > dist2 * 2f || dist2 > dist1 * 2f)
                {
                    return false; // One robot goes much farther, they won't interfere
                }
            }

            // Find closest points on two line segments
            Vector3 closestPoint1,
                closestPoint2;
            ClosestPointsOnTwoLines(
                start1,
                end1,
                start2,
                end2,
                out closestPoint1,
                out closestPoint2
            );

            float minDistance = Vector3.Distance(closestPoint1, closestPoint2);
            return minDistance < _minSafeSeparation;
        }

        /// <summary>
        /// Find closest points on two line segments
        /// </summary>
        private void ClosestPointsOnTwoLines(
            Vector3 start1,
            Vector3 end1,
            Vector3 start2,
            Vector3 end2,
            out Vector3 closestPoint1,
            out Vector3 closestPoint2
        )
        {
            Vector3 dir1 = end1 - start1;
            Vector3 dir2 = end2 - start2;
            Vector3 diff = start1 - start2;

            float a = Vector3.Dot(dir1, dir1);
            float b = Vector3.Dot(dir1, dir2);
            float c = Vector3.Dot(dir2, dir2);
            float d = Vector3.Dot(dir1, diff);
            float e = Vector3.Dot(dir2, diff);

            float denom = a * c - b * b;
            float s,
                t;

            const float epsilon = 1e-6f;

            if (Mathf.Abs(denom) < epsilon)
            {
                s = 0f;
                t = Mathf.Clamp01(-e / c);
            }
            else
            {
                s = (b * e - c * d) / denom;

                if (s < 0f)
                {
                    s = 0f;
                    t = Mathf.Clamp01(-e / c);
                }
                else if (s > 1f)
                {
                    s = 1f;
                    t = Mathf.Clamp01((b - e) / c);
                }
                else
                {
                    t = (a * e - b * d) / denom;

                    if (t < 0f)
                    {
                        t = 0f;
                        s = Mathf.Clamp01(-d / a);
                    }
                    else if (t > 1f)
                    {
                        t = 1f;
                        s = Mathf.Clamp01((b - d) / a);
                    }
                }
            }

            closestPoint1 = start1 + dir1 * s;
            closestPoint2 = start2 + dir2 * t;
        }

        /// <summary>
        /// Update workspace allocations based on current robot positions
        /// </summary>
        private void UpdateWorkspaceAllocations(RobotController[] robotControllers)
        {
            if (_workspaceManager == null)
                return;

            foreach (var controller in robotControllers)
            {
                Vector3 currentPos = controller.GetCurrentEndEffectorPosition();
                var region = _workspaceManager.GetRegionAtPosition(currentPos);

                if (region != null && !region.IsAllocated())
                {
                    // Robot entered unallocated region, allocate it
                    _workspaceManager.AllocateRegion(controller.robotId, region.regionName);
                }
            }
        }

        /// <summary>
        /// In collaborative mode, robots coordinate based on workspace allocation and collision checks.
        /// A robot is active if:
        /// - It has a target
        /// - It's not blocked by collision detection
        /// - Its target doesn't conflict with other robots
        /// - Its workspace region is allocated or available
        /// </summary>
        public bool IsRobotActive(string robotId)
        {
            // Check if robot is blocked due to collision/conflict
            if (_blockedRobots.Contains(robotId))
            {
                return false;
            }

            // Check if robot is actively moving
            if (!_activeRobots.Contains(robotId))
                return true;

            // Check workspace allocation (if WorkspaceManager is available)
            if (_workspaceManager != null && _plannedTargets.ContainsKey(robotId))
            {
                Vector3 target = _plannedTargets[robotId];
                var targetRegion = _workspaceManager.GetRegionAtPosition(target);

                if (targetRegion != null)
                {
                    // Check if region is available for this robot
                    return _workspaceManager.IsRegionAvailable(targetRegion.regionName, robotId);
                }
            }

            // Default: robot is active (collision detection already handled via _blockedRobots)
            return true;
        }

        /// <summary>
        /// In collaborative mode, all robots can be active simultaneously if coordination checks pass.
        /// </summary>
        public string GetActiveRobotId()
        {
            if (_activeRobots.Count == 0)
                return "None";

            if (_activeRobots.Count == 1)
            {
                foreach (string id in _activeRobots)
                    return id;
            }

            return string.Join(", ", _activeRobots);
        }

        /// <summary>
        /// Get the next waypoint for a robot following a replanned path.
        /// </summary>
        /// <param name="robotId">Robot identifier</param>
        /// <returns>Next waypoint or null if no waypoints queued</returns>
        public Vector3? GetNextWaypoint(string robotId)
        {
            if (!_robotWaypoints.ContainsKey(robotId))
                return null;

            var waypointQueue = _robotWaypoints[robotId];
            if (waypointQueue.Count == 0)
            {
                // All waypoints reached, remove queue
                _robotWaypoints.Remove(robotId);
                return null;
            }

            return waypointQueue.Peek();
        }

        /// <summary>
        /// Mark a waypoint as reached and advance to the next waypoint.
        /// </summary>
        /// <param name="robotId">Robot identifier</param>
        public void WaypointReached(string robotId)
        {
            if (!_robotWaypoints.ContainsKey(robotId))
                return;

            var waypointQueue = _robotWaypoints[robotId];
            if (waypointQueue.Count > 0)
            {
                Vector3 reached = waypointQueue.Dequeue();
                Debug.Log(
                    $"{LOG_PREFIX} {robotId} reached waypoint {reached}, {waypointQueue.Count} remaining"
                );
            }
        }

        /// <summary>
        /// Check if a robot has pending waypoints in its path.
        /// </summary>
        /// <param name="robotId">Robot identifier</param>
        /// <returns>True if robot has waypoints to follow</returns>
        public bool HasWaypoints(string robotId)
        {
            return _robotWaypoints.ContainsKey(robotId) && _robotWaypoints[robotId].Count > 0;
        }

        /// <summary>
        /// Resets the collaborative strategy state.
        /// </summary>
        public void Reset()
        {
            _plannedTargets.Clear();
            _activeRobots.Clear();
            _blockedRobots.Clear();
            _robotWaypoints.Clear();
            _lastCoordinationCheckTime = -1f; // Force check on next update

            // Reset workspace allocations
            if (_workspaceManager != null)
            {
                _workspaceManager.ResetAllocations();
            }

            Debug.Log($"{LOG_PREFIX} Reset collaborative coordination state");
        }

        /// <summary>
        /// Check if two robots' movements should be coordinated
        /// </summary>
        public bool RequiresCoordination(string robot1Id, string robot2Id)
        {
            if (!_plannedTargets.ContainsKey(robot1Id) || !_plannedTargets.ContainsKey(robot2Id))
                return false;

            Vector3 target1 = _plannedTargets[robot1Id];
            Vector3 target2 = _plannedTargets[robot2Id];

            // Require coordination if targets are close
            return Vector3.Distance(target1, target2) < _minSafeSeparation * 2f;
        }

        /// <summary>
        /// Set minimum safe separation distance
        /// </summary>
        public void SetMinSafeSeparation(float distance)
        {
            _minSafeSeparation = Mathf.Max(0.05f, distance);
            Debug.Log($"{LOG_PREFIX} Set minimum safe separation to {_minSafeSeparation:F3}m");
        }
    }
}
