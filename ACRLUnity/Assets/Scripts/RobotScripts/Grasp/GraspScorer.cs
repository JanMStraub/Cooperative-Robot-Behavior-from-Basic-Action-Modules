using UnityEngine;
using Configuration;
using System.Collections.Generic;
using System.Linq;

namespace Robotics.Grasp
{
    /// <summary>
    /// Scores and ranks grasp candidates using weighted multi-criteria evaluation.
    /// Criteria: IK quality, approach preference, grasp depth, stability.
    /// </summary>
    public class GraspScorer
    {
        private readonly GraspConfig _config;

        private readonly string _logPrefix = "[GRASP_SCORER]";

        /// <summary>
        /// Initialize scorer with configuration.
        /// </summary>
        /// <param name="config">Grasp planning configuration</param>
        public GraspScorer(GraspConfig config)
        {
            _config = config;
        }

        /// <summary>
        /// Score all candidates and return sorted list (best first).
        /// </summary>
        /// <param name="candidates">List of candidates to score</param>
        /// <param name="objectSize">Size of target object</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <param name="gripperRotation">Current gripper rotation (optional - for orientation consistency)</param>
        /// <returns>Sorted list of candidates (highest score first)</returns>
        public List<GraspCandidate> ScoreAndRank(
            List<GraspCandidate> candidates,
            Vector3 objectSize,
            Vector3 gripperPosition,
            Quaternion? gripperRotation = null
        )
        {
            // Score each candidate
            for (int i = 0; i < candidates.Count; i++)
            {
                var candidate = candidates[i];
                ScoreCandidate(ref candidate, objectSize, gripperPosition, gripperRotation);
                candidates[i] = candidate;
            }

            // Sort by total score (descending)
            return candidates.OrderByDescending(c => c.totalScore).ToList();
        }

        /// <summary>
        /// Compute scores for a single candidate.
        /// Updates candidate's score fields in-place.
        /// </summary>
        /// <param name="candidate">Candidate to score (passed by reference)</param>
        /// <param name="objectSize">Size of target object</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <param name="gripperRotation">Current gripper rotation (optional - for orientation consistency)</param>
        private void ScoreCandidate(
            ref GraspCandidate candidate,
            Vector3 objectSize,
            Vector3 gripperPosition,
            Quaternion? gripperRotation = null
        )
        {
            // Compute individual scores (normalized 0-1)
            candidate.ikScore = ComputeIKScore(ref candidate, gripperPosition);
            candidate.approachScore = ComputeApproachScore(ref candidate);
            candidate.depthScore = ComputeDepthScore(ref candidate, objectSize);
            candidate.stabilityScore = ComputeStabilityScore(ref candidate, objectSize);

            float orientationConsistency = gripperRotation.HasValue
                ? ComputeOrientationConsistencyScore(ref candidate, gripperRotation.Value)
                : 1.0f; // No penalty if current rotation not provided

            // Weighted combination (including antipodal and orientation consistency)
            candidate.totalScore =
                candidate.ikScore * _config.ikScoreWeight +
                candidate.approachScore * _config.approachScoreWeight +
                candidate.depthScore * _config.depthScoreWeight +
                candidate.stabilityScore * _config.stabilityScoreWeight +
                candidate.antipodalScore * _config.antipodalScoreWeight;

            // Apply orientation consistency as a multiplicative penalty (not additive weight)
            // This ensures large rotations always reduce the score significantly
            candidate.totalScore *= orientationConsistency;
        }

        /// <summary>
        /// Compute IK quality score.
        /// Higher score for positions closer to current gripper (easier to reach).
        /// If IK validation was performed, use validation quality instead.
        /// </summary>
        /// <param name="candidate">Candidate to evaluate</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <returns>Normalized score (0-1)</returns>
        private float ComputeIKScore(ref GraspCandidate candidate, Vector3 gripperPosition)
        {
            // If IK validation was performed, use that score
            if (candidate.ikValidated)
            {
                // IK score was already computed during validation
                // Just ensure it's normalized
                return Mathf.Clamp01(candidate.ikScore);
            }

            // Distance-based heuristic
            float distance = Vector3.Distance(candidate.preGraspPosition, gripperPosition);
            float maxReach = _config.maxReachDistance;

            // Closer is better (exponential falloff)
            float normalizedDistance = Mathf.Clamp01(distance / maxReach);
            return 1f - normalizedDistance;
        }

        /// <summary>
        /// Compute approach preference score.
        /// Uses configured preference weights for each approach type.
        /// </summary>
        /// <param name="candidate">Candidate to evaluate</param>
        /// <returns>Normalized score (0-1)</returns>
        private float ComputeApproachScore(ref GraspCandidate candidate)
        {
            float weight = _config.GetApproachWeight(candidate.approachType);

            // Debug: Log approach weights to diagnose scoring issues
            #if UNITY_EDITOR
            if (UnityEngine.Random.value < 0.01f) // Log 1% of the time to avoid spam
            {
                UnityEngine.Debug.Log(
                    $"{_logPrefix} Approach {candidate.approachType} has preference weight {weight:F2}"
                );
            }
            #endif

            // Normalize to 0-1 range (assuming max weight is 2.0)
            return Mathf.Clamp01(weight / 2.0f);
        }

        /// <summary>
        /// Compute grasp depth score.
        /// Penalizes deviations from target depth (now object-size-aware).
        /// </summary>
        /// <param name="candidate">Candidate to evaluate</param>
        /// <param name="objectSize">Size of target object</param>
        /// <returns>Normalized score (0-1)</returns>
        private float ComputeDepthScore(ref GraspCandidate candidate, Vector3 objectSize)
        {
            // Calculate object-size-aware target depth
            float avgObjectSize = (objectSize.x + objectSize.y + objectSize.z) / 3f;
            float targetDepth = _config.targetGraspDepth * avgObjectSize;
            float actualDepth = candidate.graspDepth;

            // Gaussian-like falloff around target (sigma scales with object size)
            float deviation = Mathf.Abs(actualDepth - targetDepth);
            float sigma = avgObjectSize * 0.15f; // 15% of object size as tolerance

            return Mathf.Exp(-deviation * deviation / (2f * sigma * sigma));
        }

        /// <summary>
        /// Compute stability score based on grasp geometry.
        /// Higher for grasps aligned with object center and gravity.
        /// Enhanced with center-of-mass alignment, contact area, and edge avoidance.
        /// </summary>
        /// <param name="candidate">Candidate to evaluate</param>
        /// <param name="objectSize">Size of target object</param>
        /// <returns>Normalized score (0-1)</returns>
        private float ComputeStabilityScore(ref GraspCandidate candidate, Vector3 objectSize)
        {
            float score = 1.0f;

            // Factor 1: Approach alignment with gravity
            // Further reduced penalty for horizontal approaches to reduce asymmetry
            // for robots with asymmetric joint limits that can't easily reach Top approaches
            // Side approach (dot=0): score *= 0.65, Top approach (dot=1): score *= 1.0
            float gravityAlignment = Mathf.Abs(Vector3.Dot(candidate.approachDirection, Vector3.up));
            score *= 0.65f + 0.35f * gravityAlignment;

            // Factor 2: Gripper can physically grasp object
            if (!_config.gripperGeometry.CanGrasp(objectSize))
            {
                score *= 0.5f; // Penalize if object might not fit
            }

            // Factor 3: Approach distance is reasonable (not too far or too close)
            float idealDistance = _config.CalculatePreGraspDistance(objectSize);
            float distanceRatio = Mathf.Abs(candidate.approachDistance - idealDistance) / idealDistance;
            score *= Mathf.Clamp01(1f - distanceRatio); // Penalize distance deviation

            // Factor 4: Center-of-mass alignment (grasp near object center)
            Vector3 graspOffset = candidate.graspPosition - candidate.contactPointEstimate;
            float centerDeviation = graspOffset.magnitude / objectSize.magnitude;
            float centerScore = Mathf.Exp(-centerDeviation * 2f); // Exponential falloff
            score *= 0.7f + 0.3f * centerScore; // Weight center alignment

            // Factor 5: Contact area estimation (wider approach = more contact)
            float contactAreaScore = EstimateContactArea(candidate, objectSize);
            score *= 0.8f + 0.2f * contactAreaScore;

            // Factor 6: Edge avoidance (penalize grasps near object edges)
            float edgeScore = ComputeEdgeAvoidanceScore(candidate, objectSize);
            score *= edgeScore;

            return Mathf.Clamp01(score);
        }

        /// <summary>
        /// Estimate contact area between gripper and object.
        /// Larger contact areas provide more stable grasps.
        /// </summary>
        /// <param name="candidate">Candidate to evaluate</param>
        /// <param name="objectSize">Size of target object</param>
        /// <returns>Normalized contact area score (0-1)</returns>
        private float EstimateContactArea(GraspCandidate candidate, Vector3 objectSize)
        {
            // Determine which object dimension is perpendicular to approach
            Vector3 absApproach = new Vector3(
                Mathf.Abs(candidate.approachDirection.x),
                Mathf.Abs(candidate.approachDirection.y),
                Mathf.Abs(candidate.approachDirection.z)
            );

            // Calculate projected contact area
            float contactArea;
            if (absApproach.x > 0.9f) // Side approach
                contactArea = objectSize.y * objectSize.z;
            else if (absApproach.y > 0.9f) // Top/bottom approach
                contactArea = objectSize.x * objectSize.z;
            else // Front/back approach
                contactArea = objectSize.x * objectSize.y;

            // Compare to gripper contact area
            float gripperArea = _config.gripperGeometry.fingerLength * _config.gripperGeometry.fingerWidth;
            float areaRatio = Mathf.Min(contactArea, gripperArea) / Mathf.Max(contactArea, gripperArea);

            return areaRatio;
        }

        /// <summary>
        /// Compute edge avoidance score.
        /// Penalizes grasps near object edges for better stability.
        /// </summary>
        /// <param name="candidate">Candidate to evaluate</param>
        /// <param name="objectSize">Size of target object</param>
        /// <returns>Edge avoidance score (0-1, higher = farther from edges)</returns>
        private float ComputeEdgeAvoidanceScore(GraspCandidate candidate, Vector3 objectSize)
        {
            // Calculate distance to nearest edge in each dimension
            Vector3 relativePos = candidate.graspPosition - candidate.contactPointEstimate;

            // Normalize to object half-extents
            Vector3 normalizedPos = new Vector3(
                Mathf.Abs(relativePos.x) / (objectSize.x * 0.5f),
                Mathf.Abs(relativePos.y) / (objectSize.y * 0.5f),
                Mathf.Abs(relativePos.z) / (objectSize.z * 0.5f)
            );

            // Find minimum distance to edge (lower = closer to edge)
            float minDistToEdge = Mathf.Min(normalizedPos.x, Mathf.Min(normalizedPos.y, normalizedPos.z));

            // Penalize if too close to edge (within 20% of half-extent)
            if (minDistToEdge > 0.8f)
            {
                // Near edge - apply penalty
                float edgePenalty = (minDistToEdge - 0.8f) / 0.2f;
                return 1.0f - edgePenalty * 0.3f; // Max 30% penalty
            }

            return 1.0f; // Not near edge
        }

        /// <summary>
        /// FIX B: Compute orientation consistency score to prevent "180-degree flip" scenarios.
        /// Penalizes grasp candidates that require large rotations from current gripper orientation.
        /// This prevents timeout issues where robots spend all their time trying to flip 180 degrees.
        /// </summary>
        /// <param name="candidate">Candidate to evaluate</param>
        /// <param name="currentGripperRotation">Current gripper rotation</param>
        /// <returns>Consistency score (0-1, higher = smaller rotation needed)</returns>
        private float ComputeOrientationConsistencyScore(ref GraspCandidate candidate, Quaternion currentGripperRotation)
        {
            // Calculate angular difference between current rotation and candidate grasp rotation
            // Quaternion.Angle returns the minimum angle in degrees (0-180)
            float deltaAngle = Quaternion.Angle(currentGripperRotation, candidate.graspRotation);

            // Apply aggressive penalty for large rotations
            // Thresholds:
            // - 0-45°: No penalty (score = 1.0)
            // - 45-90°: Linear penalty (score = 1.0 -> 0.5)
            // - 90-180°: Heavy penalty (score = 0.5 -> 0.1)

            if (deltaAngle <= 45f)
            {
                // Small rotation - no penalty
                return 1.0f;
            }
            else if (deltaAngle <= 90f)
            {
                // Medium rotation - linear penalty
                float t = (deltaAngle - 45f) / 45f; // 0 at 45°, 1 at 90°
                return Mathf.Lerp(1.0f, 0.5f, t);
            }
            else
            {
                // Large rotation (>90°) - heavy exponential penalty
                // At 175° (near flip), score approaches 0.1 (90% penalty)
                float t = (deltaAngle - 90f) / 90f; // 0 at 90°, 1 at 180°
                return Mathf.Lerp(0.5f, 0.1f, t * t); // Quadratic falloff for heavy penalty
            }
        }

        /// <summary>
        /// Filter candidates below a minimum score threshold.
        /// </summary>
        /// <param name="candidates">Scored candidates</param>
        /// <param name="minScore">Minimum total score threshold</param>
        /// <returns>Filtered list of candidates</returns>
        public List<GraspCandidate> FilterByMinScore(List<GraspCandidate> candidates, float minScore)
        {
            return candidates.Where(c => c.totalScore >= minScore).ToList();
        }

        /// <summary>
        /// Get top N candidates from scored list.
        /// </summary>
        /// <param name="candidates">Scored and sorted candidates</param>
        /// <param name="count">Number of candidates to return</param>
        /// <returns>Top N candidates</returns>
        public List<GraspCandidate> GetTopN(List<GraspCandidate> candidates, int count)
        {
            return candidates.Take(count).ToList();
        }

        /// <summary>
        /// Normalize all candidate scores to 0-1 range within the list.
        /// Useful for comparing candidates across different scenarios.
        /// </summary>
        /// <param name="candidates">Candidates to normalize</param>
        public void NormalizeScores(List<GraspCandidate> candidates)
        {
            if (candidates.Count == 0)
                return;

            float minScore = candidates.Min(c => c.totalScore);
            float maxScore = candidates.Max(c => c.totalScore);
            float range = maxScore - minScore;

            if (range < 0.001f)
                return; // All scores identical

            for (int i = 0; i < candidates.Count; i++)
            {
                var candidate = candidates[i];
                candidate.totalScore = (candidate.totalScore - minScore) / range;
                candidates[i] = candidate;
            }
        }
    }
}
