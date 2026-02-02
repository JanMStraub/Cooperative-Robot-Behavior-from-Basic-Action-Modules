using System.Collections.Generic;
using System.Linq;
using Configuration;
using UnityEngine;

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
            for (int i = 0; i < candidates.Count; i++)
            {
                var candidate = candidates[i];
                ScoreCandidate(ref candidate, objectSize, gripperPosition, gripperRotation);
                candidates[i] = candidate;
            }

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
            candidate.ikScore = ComputeIKScore(ref candidate, gripperPosition);
            candidate.approachScore = ComputeApproachScore(ref candidate);
            candidate.depthScore = ComputeDepthScore(ref candidate, objectSize);
            candidate.stabilityScore = ComputeStabilityScore(ref candidate, objectSize);

            float orientationConsistency = gripperRotation.HasValue
                ? ComputeOrientationConsistencyScore(ref candidate, gripperRotation.Value)
                : 1.0f;

            candidate.totalScore =
                candidate.ikScore * _config.ikScoreWeight
                + candidate.approachScore * _config.approachScoreWeight
                + candidate.depthScore * _config.depthScoreWeight
                + candidate.stabilityScore * _config.stabilityScoreWeight
                + candidate.antipodalScore * _config.antipodalScoreWeight;

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
            if (candidate.ikValidated)
            {
                return Mathf.Clamp01(candidate.ikScore);
            }

            float distance = Vector3.Distance(candidate.preGraspPosition, gripperPosition);
            float maxReach = _config.maxReachDistance;

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

#if UNITY_EDITOR
            if (UnityEngine.Random.value < 0.01f)
            {
                UnityEngine.Debug.Log(
                    $"{_logPrefix} Approach {candidate.approachType} has preference weight {weight:F2}"
                );
            }
#endif

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
            float avgObjectSize = (objectSize.x + objectSize.y + objectSize.z) / 3f;
            float targetDepth = _config.targetGraspDepth * avgObjectSize;
            float actualDepth = candidate.graspDepth;

            float deviation = Mathf.Abs(actualDepth - targetDepth);
            float sigma = avgObjectSize * 0.15f;

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

            float gravityAlignment = Mathf.Abs(
                Vector3.Dot(candidate.approachDirection, Vector3.up)
            );
            score *= 0.65f + 0.35f * gravityAlignment;

            if (!_config.gripperGeometry.CanGrasp(objectSize))
            {
                score *= 0.5f;
            }

            float idealDistance = _config.CalculatePreGraspDistance(objectSize);
            float distanceRatio =
                Mathf.Abs(candidate.approachDistance - idealDistance) / idealDistance;
            score *= Mathf.Clamp01(1f - distanceRatio);

            Vector3 graspOffset = candidate.graspPosition - candidate.contactPointEstimate;
            float centerDeviation = graspOffset.magnitude / objectSize.magnitude;
            float centerScore = Mathf.Exp(-centerDeviation * 2f);
            score *= 0.7f + 0.3f * centerScore;

            float contactAreaScore = EstimateContactArea(candidate, objectSize);
            score *= 0.8f + 0.2f * contactAreaScore;

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
            Vector3 absApproach = new Vector3(
                Mathf.Abs(candidate.approachDirection.x),
                Mathf.Abs(candidate.approachDirection.y),
                Mathf.Abs(candidate.approachDirection.z)
            );

            float contactArea;
            if (absApproach.x > 0.9f)
                contactArea = objectSize.y * objectSize.z;
            else if (absApproach.y > 0.9f)
                contactArea = objectSize.x * objectSize.z;
            else
                contactArea = objectSize.x * objectSize.y;

            float gripperArea =
                _config.gripperGeometry.fingerLength * _config.gripperGeometry.fingerWidth;
            float areaRatio =
                Mathf.Min(contactArea, gripperArea) / Mathf.Max(contactArea, gripperArea);

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
            Vector3 relativePos = candidate.graspPosition - candidate.contactPointEstimate;

            Vector3 normalizedPos = new Vector3(
                Mathf.Abs(relativePos.x) / (objectSize.x * 0.5f),
                Mathf.Abs(relativePos.y) / (objectSize.y * 0.5f),
                Mathf.Abs(relativePos.z) / (objectSize.z * 0.5f)
            );

            float minDistToEdge = Mathf.Min(
                normalizedPos.x,
                Mathf.Min(normalizedPos.y, normalizedPos.z)
            );

            if (minDistToEdge > 0.8f)
            {
                float edgePenalty = (minDistToEdge - 0.8f) / 0.2f;
                return 1.0f - edgePenalty * 0.3f;
            }

            return 1.0f;
        }

        /// <summary>
        /// Compute orientation consistency score to prevent "180-degree flip" scenarios.
        /// Penalizes grasp candidates that require large rotations from current gripper orientation.
        /// </summary>
        /// <param name="candidate">Candidate to evaluate</param>
        /// <param name="currentGripperRotation">Current gripper rotation</param>
        /// <returns>Consistency score (0-1, higher = smaller rotation needed)</returns>
        private float ComputeOrientationConsistencyScore(
            ref GraspCandidate candidate,
            Quaternion currentGripperRotation
        )
        {
            float deltaAngle = Quaternion.Angle(currentGripperRotation, candidate.graspRotation);

            if (deltaAngle <= 45f)
            {
                return 1.0f;
            }
            else if (deltaAngle <= 90f)
            {
                float t = (deltaAngle - 45f) / 45f;
                return Mathf.Lerp(1.0f, 0.5f, t);
            }
            else
            {
                float t = (deltaAngle - 90f) / 90f;
                return Mathf.Lerp(0.5f, 0.1f, t * t);
            }
        }

        /// <summary>
        /// Filter candidates below a minimum score threshold.
        /// </summary>
        /// <param name="candidates">Scored candidates</param>
        /// <param name="minScore">Minimum total score threshold</param>
        /// <returns>Filtered list of candidates</returns>
        public List<GraspCandidate> FilterByMinScore(
            List<GraspCandidate> candidates,
            float minScore
        )
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
                return;

            for (int i = 0; i < candidates.Count; i++)
            {
                var candidate = candidates[i];
                candidate.totalScore = (candidate.totalScore - minScore) / range;
                candidates[i] = candidate;
            }
        }
    }
}
