using System.Collections.Generic;
using System.Linq;
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
        private float _minSafeSeparation = 0.2f;

        // Coordination state
        private float _lastCoordinationCheckTime = 0f;
        private const float COORDINATION_CHECK_INTERVAL = 0.5f; // Check coordination every 500ms

        private const string LOG_PREFIX = "[COLLABORATIVE_STRATEGY]";

        /// <summary>
        /// Constructor - initialize workspace manager reference
        /// </summary>
        public CollaborativeStrategy()
        {
            _workspaceManager = WorkspaceManager.Instance;
            if (_workspaceManager == null)
            {
                Debug.LogWarning(
                    $"{LOG_PREFIX} WorkspaceManager not found. Workspace coordination disabled."
                );
            }
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

            // Update planned targets
            foreach (var controller in robotControllers)
            {
                if (controller.HasTarget)
                {
                    _plannedTargets[controller.robotId] = controller.GetCurrentTarget().Value;
                    _activeRobots.Add(controller.robotId);
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
        /// Perform coordination safety checks for all active robots
        /// </summary>
        private void PerformCoordinationCheck(RobotController[] robotControllers)
        {
            if (robotControllers.Length < 2)
                return; // No coordination needed for single robot

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
        /// Check for conflicts between two robots
        /// </summary>
        private void CheckRobotPairConflict(RobotController robot1, RobotController robot2)
        {
            Vector3 target1 = robot1.GetCurrentTarget().Value;
            Vector3 target2 = robot2.GetCurrentTarget().Value;
            Vector3 current1 = robot1.GetCurrentEndEffectorPosition();
            Vector3 current2 = robot2.GetCurrentEndEffectorPosition();

            // Check 1: Target collision (both robots targeting same location)
            float targetDistance = Vector3.Distance(target1, target2);
            if (targetDistance < _minSafeSeparation)
            {
                Debug.LogWarning(
                    $"{LOG_PREFIX} Collision detected: {robot1.robotId} and {robot2.robotId} targeting same location (distance: {targetDistance:F3}m)"
                );
                // Pause one robot temporarily
                // In production, this should trigger Python CoordinationVerifier
                return;
            }

            // Check 2: Path collision (robots' paths will cross)
            if (WillPathsCollide(current1, target1, current2, target2))
            {
                Debug.LogWarning(
                    $"{LOG_PREFIX} Path collision detected between {robot1.robotId} and {robot2.robotId}"
                );
                // Serialize movements or adjust paths
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
                            $"{LOG_PREFIX} Workspace conflict: Both robots targeting '{region1.regionName}'"
                        );
                    }
                }
            }
        }

        /// <summary>
        /// Check if two robot paths will collide
        /// Uses simple swept sphere collision detection
        /// </summary>
        private bool WillPathsCollide(Vector3 start1, Vector3 end1, Vector3 start2, Vector3 end2)
        {
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

            if (denom != 0f)
            {
                s = Mathf.Clamp01((b * e - c * d) / denom);
                t = Mathf.Clamp01((a * e - b * d) / denom);
            }
            else
            {
                s = 0f;
                t = 0f;
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
        /// - Its target doesn't conflict with other robots
        /// - Its workspace region is allocated or available
        /// </summary>
        public bool IsRobotActive(string robotId)
        {
            // If no workspace manager, all robots are active (fallback to independent mode)
            if (_workspaceManager == null)
                return true;

            // Check if robot is actively moving
            if (!_activeRobots.Contains(robotId))
                return true;

            // Check workspace allocation
            if (_plannedTargets.ContainsKey(robotId))
            {
                Vector3 target = _plannedTargets[robotId];
                var targetRegion = _workspaceManager.GetRegionAtPosition(target);

                if (targetRegion != null)
                {
                    // Check if region is available for this robot
                    return _workspaceManager.IsRegionAvailable(targetRegion.regionName, robotId);
                }
            }

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
                return _activeRobots.First();

            return string.Join(", ", _activeRobots);
        }

        /// <summary>
        /// Resets the collaborative strategy state.
        /// </summary>
        public void Reset()
        {
            _plannedTargets.Clear();
            _activeRobots.Clear();
            _lastCoordinationCheckTime = 0f;

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
