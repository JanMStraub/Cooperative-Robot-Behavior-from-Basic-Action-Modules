using UnityEngine;
using Configuration;
using System.Collections.Generic;

namespace Robotics.Grasp
{
    /// <summary>
    /// Filters grasp candidates based on collision-free approach paths.
    /// Uses SphereCast along approach trajectory to detect obstacles.
    /// </summary>
    public class GraspCollisionFilter
    {
        private readonly GraspConfig _config;

        /// <summary>
        /// Initialize collision filter with configuration.
        /// </summary>
        /// <param name="config">Grasp planning configuration</param>
        public GraspCollisionFilter(GraspConfig config)
        {
            _config = config;
        }

        /// <summary>
        /// Filter candidates by collision-free approach paths.
        /// Removes candidates with obstacles along approach trajectory.
        /// </summary>
        /// <param name="candidates">Candidates to filter</param>
        /// <param name="targetObject">Target object (excluded from collision checks)</param>
        /// <returns>List of collision-free candidates</returns>
        public List<GraspCandidate> FilterCandidates(
            List<GraspCandidate> candidates,
            GameObject targetObject = null
        )
        {
            if (!_config.enableCollisionChecking)
            {
                // Mark all as collision-validated if checking disabled
                for (int i = 0; i < candidates.Count; i++)
                {
                    var candidate = candidates[i];
                    candidate.collisionValidated = true;
                    candidates[i] = candidate;
                }
                return candidates;
            }

            var validCandidates = new List<GraspCandidate>();

            foreach (var candidate in candidates)
            {
                bool collisionFree = CheckApproachPath(candidate, targetObject);

                if (collisionFree)
                {
                    var validated = candidate;
                    validated.collisionValidated = true;
                    validCandidates.Add(validated);
                }
            }

            return validCandidates;
        }

        /// <summary>
        /// Check if approach path is collision-free.
        /// Uses SphereCast along waypoints from pre-grasp to grasp position.
        /// </summary>
        /// <param name="candidate">Candidate to check</param>
        /// <param name="targetObject">Target object to exclude from collision checks</param>
        /// <returns>True if path is collision-free</returns>
        private bool CheckApproachPath(GraspCandidate candidate, GameObject targetObject)
        {
            // Generate waypoints along approach path
            Vector3[] waypoints = GenerateWaypoints(
                candidate.preGraspPosition,
                candidate.graspPosition,
                _config.collisionCheckWaypoints
            );

            // Check each segment between waypoints
            for (int i = 0; i < waypoints.Length - 1; i++)
            {
                Vector3 start = waypoints[i];
                Vector3 end = waypoints[i + 1];
                Vector3 direction = end - start;
                float distance = direction.magnitude;

                // SphereCast along segment
                if (Physics.SphereCast(
                    start,
                    _config.collisionCheckRadius,
                    direction.normalized,
                    out RaycastHit hit,
                    distance,
                    _config.collisionLayerMask
                ))
                {
                    // Check if hit object is the target (allowed)
                    if (targetObject != null && hit.collider.gameObject == targetObject)
                    {
                        continue; // Hitting target is acceptable
                    }

                    // Hit an obstacle - path blocked
                    return false;
                }
            }

            // Additionally check retreat path if enabled
            if (_config.enableRetreat)
            {
                bool retreatClear = CheckRetreatPath(candidate, targetObject);
                if (!retreatClear)
                    return false;
            }

            // All checks passed
            return true;
        }

        /// <summary>
        /// Check if retreat path is collision-free.
        /// </summary>
        /// <param name="candidate">Candidate with retreat position</param>
        /// <param name="targetObject">Target object to exclude</param>
        /// <returns>True if retreat path is clear</returns>
        private bool CheckRetreatPath(GraspCandidate candidate, GameObject targetObject)
        {
            Vector3 start = candidate.graspPosition;
            Vector3 end = candidate.retreatPosition;
            Vector3 direction = end - start;
            float distance = direction.magnitude;

            // Single SphereCast for retreat (typically straight up)
            if (Physics.SphereCast(
                start,
                _config.collisionCheckRadius,
                direction.normalized,
                out RaycastHit hit,
                distance,
                _config.collisionLayerMask
            ))
            {
                // Check if hit is target object (allowed initially)
                if (targetObject != null && hit.collider.gameObject == targetObject)
                {
                    // Target object in retreat path is OK if hit early (object being lifted)
                    if (hit.distance < distance * 0.3f) // Within first 30% of retreat
                    {
                        return true;
                    }
                }

                // Obstacle in retreat path
                return false;
            }

            return true;
        }

        /// <summary>
        /// Generate waypoints along a path.
        /// </summary>
        /// <param name="start">Start position</param>
        /// <param name="end">End position</param>
        /// <param name="count">Number of waypoints (including start and end)</param>
        /// <returns>Array of waypoint positions</returns>
        private Vector3[] GenerateWaypoints(Vector3 start, Vector3 end, int count)
        {
            count = Mathf.Max(2, count); // Minimum 2 waypoints (start and end)
            Vector3[] waypoints = new Vector3[count];

            for (int i = 0; i < count; i++)
            {
                float t = i / (float)(count - 1);
                waypoints[i] = Vector3.Lerp(start, end, t);
            }

            return waypoints;
        }

        /// <summary>
        /// Check if a single candidate has collision-free approach.
        /// </summary>
        /// <param name="candidate">Candidate to validate</param>
        /// <param name="targetObject">Target object to exclude</param>
        /// <returns>True if collision-free</returns>
        public bool IsCollisionFree(GraspCandidate candidate, GameObject targetObject = null)
        {
            if (!_config.enableCollisionChecking)
                return true;

            return CheckApproachPath(candidate, targetObject);
        }

        /// <summary>
        /// Visualize collision check path for debugging (call from OnDrawGizmos).
        /// </summary>
        /// <param name="candidate">Candidate to visualize</param>
        /// <param name="isCollisionFree">Whether path is collision-free</param>
        public void DebugDrawPath(GraspCandidate candidate, bool isCollisionFree)
        {
            Color pathColor = isCollisionFree ? Color.green : Color.red;

            // Draw approach path
            Vector3[] waypoints = GenerateWaypoints(
                candidate.preGraspPosition,
                candidate.graspPosition,
                _config.collisionCheckWaypoints
            );

            for (int i = 0; i < waypoints.Length - 1; i++)
            {
                Gizmos.color = pathColor;
                Gizmos.DrawLine(waypoints[i], waypoints[i + 1]);
                Gizmos.DrawWireSphere(waypoints[i], _config.collisionCheckRadius);
            }

            // Draw retreat path if enabled
            if (_config.enableRetreat)
            {
                Gizmos.color = pathColor * 0.7f; // Slightly dimmer
                Gizmos.DrawLine(candidate.graspPosition, candidate.retreatPosition);
                Gizmos.DrawWireSphere(candidate.retreatPosition, _config.collisionCheckRadius);
            }

            // Draw key poses
            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(candidate.preGraspPosition, _config.collisionCheckRadius * 1.5f);
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(candidate.graspPosition, _config.collisionCheckRadius * 1.5f);
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(candidate.retreatPosition, _config.collisionCheckRadius * 1.5f);
        }

        /// <summary>
        /// Check multiple candidates in batch and return collision status for each.
        /// Useful for analysis and debugging.
        /// </summary>
        /// <param name="candidates">Candidates to check</param>
        /// <param name="targetObject">Target object to exclude</param>
        /// <returns>Array of collision-free flags (parallel to input)</returns>
        public bool[] BatchCheck(List<GraspCandidate> candidates, GameObject targetObject = null)
        {
            bool[] results = new bool[candidates.Count];

            for (int i = 0; i < candidates.Count; i++)
            {
                results[i] = CheckApproachPath(candidates[i], targetObject);
            }

            return results;
        }
    }
}
