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
        /// <returns>Sorted list of candidates (highest score first)</returns>
        public List<GraspCandidate> ScoreAndRank(
            List<GraspCandidate> candidates,
            Vector3 objectSize,
            Vector3 gripperPosition
        )
        {
            // Score each candidate
            for (int i = 0; i < candidates.Count; i++)
            {
                var candidate = candidates[i];
                ScoreCandidate(ref candidate, objectSize, gripperPosition);
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
        private void ScoreCandidate(
            ref GraspCandidate candidate,
            Vector3 objectSize,
            Vector3 gripperPosition
        )
        {
            // Compute individual scores (normalized 0-1)
            candidate.ikScore = ComputeIKScore(ref candidate, gripperPosition);
            candidate.approachScore = ComputeApproachScore(ref candidate);
            candidate.depthScore = ComputeDepthScore(ref candidate, objectSize);
            candidate.stabilityScore = ComputeStabilityScore(ref candidate, objectSize);

            // Weighted combination
            candidate.totalScore =
                candidate.ikScore * _config.ikScoreWeight +
                candidate.approachScore * _config.approachScoreWeight +
                candidate.depthScore * _config.depthScoreWeight +
                candidate.stabilityScore * _config.stabilityScoreWeight;
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

            // Normalize to 0-1 range (assuming max weight is 2.0)
            return Mathf.Clamp01(weight / 2.0f);
        }

        /// <summary>
        /// Compute grasp depth score.
        /// Penalizes deviations from target depth.
        /// </summary>
        /// <param name="candidate">Candidate to evaluate</param>
        /// <param name="objectSize">Size of target object</param>
        /// <returns>Normalized score (0-1)</returns>
        private float ComputeDepthScore(ref GraspCandidate candidate, Vector3 objectSize)
        {
            // Compare to target depth
            float targetDepth = _config.targetGraspDepth * 0.05f; // Reference depth
            float actualDepth = candidate.graspDepth;

            // Gaussian-like falloff around target
            float deviation = Mathf.Abs(actualDepth - targetDepth);
            float sigma = 0.02f; // 2cm standard deviation

            return Mathf.Exp(-deviation * deviation / (2f * sigma * sigma));
        }

        /// <summary>
        /// Compute stability score based on grasp geometry.
        /// Higher for grasps aligned with object center and gravity.
        /// </summary>
        /// <param name="candidate">Candidate to evaluate</param>
        /// <param name="objectSize">Size of target object</param>
        /// <returns>Normalized score (0-1)</returns>
        private float ComputeStabilityScore(ref GraspCandidate candidate, Vector3 objectSize)
        {
            float score = 1.0f;

            // Factor 1: Approach alignment with gravity (top approaches more stable)
            float gravityAlignment = Mathf.Abs(Vector3.Dot(candidate.approachDirection, Vector3.up));
            score *= 0.3f + 0.7f * gravityAlignment; // Bias toward top approaches

            // Factor 2: Gripper can physically grasp object
            if (!_config.gripperGeometry.CanGrasp(objectSize))
            {
                score *= 0.5f; // Penalize if object might not fit
            }

            // Factor 3: Approach distance is reasonable (not too far or too close)
            float idealDistance = _config.CalculatePreGraspDistance(objectSize);
            float distanceRatio = Mathf.Abs(candidate.approachDistance - idealDistance) / idealDistance;
            score *= Mathf.Clamp01(1f - distanceRatio); // Penalize distance deviation

            return Mathf.Clamp01(score);
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
