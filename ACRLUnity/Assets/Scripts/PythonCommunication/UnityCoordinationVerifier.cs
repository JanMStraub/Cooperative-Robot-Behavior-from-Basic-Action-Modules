using System.Collections.Generic;
using System.Linq;
using Configuration;
using Robotics;
using Simulation;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Fast Unity-only coordination verifier.
    /// Performs local collision checks without Python backend communication.
    /// Best for performance-critical scenarios.
    /// </summary>
    public class UnityCoordinationVerifier : ICoordinationVerifier
    {
        private float _minSafeSeparation;
        private RobotManager _robotManager;
        private SimulationManager _simulationManager;

        private const string LOG_PREFIX = "[UNITY_VERIFIER]";

        public string VerifierName => "Unity";
        public bool IsAvailable => true; // Always available

        /// <summary>
        /// Constructor with configurable minimum safe separation.
        /// </summary>
        /// <param name="minSafeSeparation">Minimum safe separation in meters</param>
        public UnityCoordinationVerifier(float minSafeSeparation = 0.2f)
        {
            _minSafeSeparation = Mathf.Max(0.05f, minSafeSeparation);
            _robotManager = RobotManager.Instance;
            _simulationManager = SimulationManager.Instance;

            if (_robotManager == null)
            {
                Debug.LogWarning($"{LOG_PREFIX} RobotManager not found");
            }
        }

        /// <summary>
        /// Verify if a robot movement is safe using Unity-only checks.
        /// </summary>
        public VerificationResult VerifyMovement(
            string robotId,
            Vector3 targetPosition,
            Vector3 currentPosition
        )
        {
            var warnings = new List<string>();

            if (_robotManager == null)
            {
                return new VerificationResult(
                    true,
                    "RobotManager not available, skipping verification",
                    warnings
                );
            }

            // Get all active robots
            var allRobots = _robotManager.RobotInstances;
            if (allRobots == null || allRobots.Count == 0)
            {
                return new VerificationResult(true, "No other robots to check", warnings);
            }

            // Check for collisions with other robots
            foreach (var otherRobotEntry in allRobots)
            {
                string otherRobotId = otherRobotEntry.Key;
                var otherInstance = otherRobotEntry.Value;

                if (otherRobotId == robotId)
                    continue;

                var otherController = otherInstance.controller;
                if (otherController == null)
                    continue;

                // Check 1: Distance to other robot's current position
                Vector3 otherCurrentPos = otherController.GetCurrentEndEffectorPosition();
                float distanceToCurrent = Vector3.Distance(targetPosition, otherCurrentPos);

                if (distanceToCurrent < _minSafeSeparation)
                {
                    return new VerificationResult(
                        false,
                        $"Target too close to {otherRobotId} current position ({distanceToCurrent:F3}m < {_minSafeSeparation:F3}m)",
                        warnings
                    );
                }

                // Check 2: Distance to other robot's target (if has target)
                if (otherController.HasTarget)
                {
                    var otherTarget = otherController.GetCurrentTarget();
                    if (otherTarget.HasValue)
                    {
                        float distanceToTarget = Vector3.Distance(
                            targetPosition,
                            otherTarget.Value
                        );

                        if (distanceToTarget < _minSafeSeparation)
                        {
                            return new VerificationResult(
                                false,
                                $"Target conflicts with {otherRobotId} target ({distanceToTarget:F3}m < {_minSafeSeparation:F3}m)",
                                warnings
                            );
                        }

                        // Warning if close to other robot's target
                        if (distanceToTarget < _minSafeSeparation * 1.5f)
                        {
                            warnings.Add(
                                $"Target close to {otherRobotId} target ({distanceToTarget:F3}m)"
                            );
                        }
                    }
                }

                // Check 3: Path collision (will paths cross?)
                if (otherController.HasTarget)
                {
                    var otherTarget = otherController.GetCurrentTarget();
                    if (otherTarget.HasValue && WillPathsCollide(
                        currentPosition,
                        targetPosition,
                        otherCurrentPos,
                        otherTarget.Value
                    ))
                    {
                        return new VerificationResult(
                            false,
                            $"Path collision with {otherRobotId}",
                            warnings
                        );
                    }
                }
            }

            // All checks passed
            string resultMsg = warnings.Count > 0
                ? $"Safe with {warnings.Count} warning(s)"
                : "Safe";

            Debug.Log($"{LOG_PREFIX} {robotId} movement verified: {resultMsg}");
            return new VerificationResult(true, resultMsg, warnings);
        }

        /// <summary>
        /// Check if two robot paths will collide using swept sphere collision detection.
        /// Optimized with early-exit AABB check before expensive line segment math.
        /// </summary>
        private bool WillPathsCollide(Vector3 start1, Vector3 end1, Vector3 start2, Vector3 end2)
        {
            // OPTIMIZATION: Early exit with AABB (Axis-Aligned Bounding Box) check
            // This is much faster than line segment closest points and filters out distant robots
            float radius = _minSafeSeparation;

            // Calculate AABB for path 1
            Vector3 min1 = Vector3.Min(start1, end1) - Vector3.one * radius;
            Vector3 max1 = Vector3.Max(start1, end1) + Vector3.one * radius;

            // Calculate AABB for path 2
            Vector3 min2 = Vector3.Min(start2, end2) - Vector3.one * radius;
            Vector3 max2 = Vector3.Max(start2, end2) + Vector3.one * radius;

            // Check if AABBs overlap
            bool xOverlap = min1.x <= max2.x && max1.x >= min2.x;
            bool yOverlap = min1.y <= max2.y && max1.y >= min2.y;
            bool zOverlap = min1.z <= max2.z && max1.z >= min2.z;

            if (!(xOverlap && yOverlap && zOverlap))
            {
                // AABBs don't overlap - robots are far apart, no collision possible
                return false;
            }

            // AABBs overlap - perform accurate line segment closest points test
            Vector3 closestPoint1, closestPoint2;
            ClosestPointsOnTwoLines(start1, end1, start2, end2, out closestPoint1, out closestPoint2);

            float minDistance = Vector3.Distance(closestPoint1, closestPoint2);
            return minDistance < _minSafeSeparation;
        }

        /// <summary>
        /// Find closest points on two line segments.
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
            float s, t;

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
    }
}
