using System.Collections.Generic;
using Configuration;
using UnityEngine;

namespace Robotics.Grasp
{
    /// <summary>
    /// Filters grasp candidates based on collision-free approach paths.
    /// Uses SphereCast along approach trajectory to detect obstacles.
    /// </summary>
    public class GraspCollisionFilter
    {
        private readonly GraspConfig _config;

        private readonly string _logPrefix = "[GRASP_COLLISION_FILTER]";

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
                foreach (var candidate in candidates)
                {
                    candidate.collisionValidated = true;
                }
                Debug.Log(
                    $"{_logPrefix} Collision checking disabled, accepting all {candidates.Count} candidates"
                );
                return candidates;
            }

            var validCandidates = new List<GraspCandidate>();
            int rejectedCount = 0;
            int rejectedTopCount = 0;
            int rejectedSideCount = 0;
            int rejectedFrontCount = 0;

            foreach (var candidate in candidates)
            {
                bool collisionFree = CheckApproachPath(candidate, targetObject);

                if (collisionFree)
                {
                    candidate.collisionValidated = true;
                    validCandidates.Add(candidate);
                }
                else
                {
                    rejectedCount++;
                    switch (candidate.approachType)
                    {
                        case GraspApproach.Top:
                            rejectedTopCount++;
                            break;
                        case GraspApproach.Side:
                            rejectedSideCount++;
                            break;
                        case GraspApproach.Front:
                            rejectedFrontCount++;
                            break;
                    }
                }
            }

            Debug.Log(
                $"{_logPrefix} Validated {validCandidates.Count}/{candidates.Count} candidates (rejected {rejectedCount} due to collisions: Top={rejectedTopCount}, Side={rejectedSideCount}, Front={rejectedFrontCount})"
            );

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
            Vector3[] waypoints = GenerateWaypoints(
                candidate.preGraspPosition,
                candidate.graspPosition,
                _config.collisionCheckWaypoints
            );

            for (int i = 0; i < waypoints.Length - 1; i++)
            {
                Vector3 start = waypoints[i];
                Vector3 end = waypoints[i + 1];
                Vector3 direction = end - start;
                float distance = direction.magnitude;

                if (
                    Physics.CheckSphere(
                        start,
                        _config.collisionCheckRadius,
                        _config.collisionLayerMask
                    )
                )
                {
                    var colliders = Physics.OverlapSphere(
                        start,
                        _config.collisionCheckRadius,
                        _config.collisionLayerMask
                    );

                    foreach (var col in colliders)
                    {
                        if (targetObject == null || col.gameObject != targetObject)
                        {
                            return false;
                        }
                    }
                }

                if (
                    Physics.SphereCast(
                        start,
                        _config.collisionCheckRadius,
                        direction.normalized,
                        out RaycastHit hit,
                        distance,
                        _config.collisionLayerMask
                    )
                )
                {
                    if (targetObject != null && hit.collider.gameObject == targetObject)
                    {
                        continue;
                    }

                    return false;
                }
            }

            if (_config.enableRetreat)
            {
                bool retreatClear = CheckRetreatPath(candidate, targetObject);
                if (!retreatClear)
                    return false;
            }

            return true;
        }

        private bool CheckRetreatPath(GraspCandidate candidate, GameObject targetObject)
        {
            Vector3 start = candidate.graspPosition;
            Vector3 end = candidate.retreatPosition;
            Vector3 direction = end - start;
            float distance = direction.magnitude;

            if (
                Physics.SphereCast(
                    start,
                    _config.collisionCheckRadius,
                    direction.normalized,
                    out RaycastHit hit,
                    distance,
                    _config.collisionLayerMask
                )
            )
            {
                if (targetObject != null && hit.collider.gameObject == targetObject)
                {
                    if (hit.distance < distance * 0.3f)
                    {
                        return true;
                    }
                }

                Debug.Log(
                    $"{_logPrefix} Retreat collision detected: hit '{hit.collider.gameObject.name}' (layer: {LayerMask.LayerToName(hit.collider.gameObject.layer)}) at distance {hit.distance:F3}m"
                );
                return false;
            }

            return true;
        }

        private Vector3[] GenerateWaypoints(Vector3 start, Vector3 end, int count)
        {
            count = Mathf.Max(2, count);
            Vector3[] waypoints = new Vector3[count];

            for (int i = 0; i < count; i++)
            {
                float t = i / (float)(count - 1);
                waypoints[i] = Vector3.Lerp(start, end, t);
            }

            return waypoints;
        }

        /// <summary>
        /// Visualize collision check path for debugging (call from OnDrawGizmos).
        /// </summary>
        /// <param name="candidate">Candidate to visualize</param>
        /// <param name="isCollisionFree">Whether path is collision-free</param>
        public void DebugDrawPath(GraspCandidate candidate, bool isCollisionFree)
        {
            Color pathColor = isCollisionFree ? Color.green : Color.red;

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

            if (_config.enableRetreat)
            {
                Gizmos.color = pathColor * 0.7f;
                Gizmos.DrawLine(candidate.graspPosition, candidate.retreatPosition);
                Gizmos.DrawWireSphere(candidate.retreatPosition, _config.collisionCheckRadius);
            }

            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(candidate.preGraspPosition, _config.collisionCheckRadius * 1.5f);
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(candidate.graspPosition, _config.collisionCheckRadius * 1.5f);
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(candidate.retreatPosition, _config.collisionCheckRadius * 1.5f);
        }
    }
}
