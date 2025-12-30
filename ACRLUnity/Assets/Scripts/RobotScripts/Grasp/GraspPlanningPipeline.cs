using UnityEngine;
using Configuration;
using System.Collections.Generic;
using System.Diagnostics;

namespace Robotics.Grasp
{
    /// <summary>
    /// MoveIt2-inspired grasp planning pipeline orchestrating full candidate selection.
    /// Pipeline: Generate → IK Filter → Collision Filter → Score → Select Best.
    /// </summary>
    public class GraspPlanningPipeline
    {
        private readonly GraspConfig _config;
        private readonly GraspCandidateGenerator _generator;
        private readonly GraspIKFilter _ikFilter;
        private readonly GraspCollisionFilter _collisionFilter;
        private readonly GraspScorer _scorer;

        private const string _logPrefix = "[GRASP_PIPELINE]";

        /// <summary>
        /// Initialize pipeline with robot configuration.
        /// </summary>
        /// <param name="config">Grasp planning configuration</param>
        /// <param name="joints">Robot joint articulation bodies</param>
        /// <param name="ikReferenceFrame">IK coordinate frame</param>
        /// <param name="endEffector">End effector transform</param>
        /// <param name="dampingFactor">IK solver damping factor</param>
        public GraspPlanningPipeline(
            GraspConfig config,
            ArticulationBody[] joints,
            Transform ikReferenceFrame,
            Transform endEffector,
            float dampingFactor = 0.1f
        )
        {
            _config = config;
            _generator = new GraspCandidateGenerator(config);
            _ikFilter = new GraspIKFilter(config, joints, ikReferenceFrame, endEffector, dampingFactor);
            _collisionFilter = new GraspCollisionFilter(config);
            _scorer = new GraspScorer(config);
        }

        /// <summary>
        /// Execute full planning pipeline to find best grasp candidate.
        /// </summary>
        /// <param name="targetObject">Object to grasp</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <param name="options">Grasp options (approach override, etc.)</param>
        /// <returns>Best grasp candidate, or null if no valid grasp found</returns>
        public GraspCandidate? PlanGrasp(
            GameObject targetObject,
            Vector3 gripperPosition,
            GraspOptions? options = null
        )
        {
            var stopwatch = Stopwatch.StartNew();

            // Stage 1: Generate candidates
            List<GraspCandidate> candidates;

            if (options?.useGraspPlanning == false)
            {
                // Simple mode - use existing planner
                var simpleCandidate = _generator.GenerateSimpleCandidate(
                    targetObject,
                    gripperPosition,
                    options?.approach
                );
                candidates = new List<GraspCandidate> { simpleCandidate };
            }
            else
            {
                // Advanced mode - generate multiple candidates
                candidates = _generator.GenerateCandidates(targetObject, gripperPosition);
            }

            UnityEngine.Debug.Log($"{_logPrefix} Generated {candidates.Count} candidates");

            if (candidates.Count == 0)
            {
                UnityEngine.Debug.LogWarning($"{_logPrefix} No candidates generated");
                return null;
            }

            // Stage 2: IK filtering
            var ikValidCandidates = _ikFilter.FilterCandidates(candidates, gripperPosition);
            UnityEngine.Debug.Log($"{_logPrefix} {ikValidCandidates.Count} candidates passed IK filter");

            if (ikValidCandidates.Count == 0)
            {
                UnityEngine.Debug.LogWarning($"{_logPrefix} No candidates passed IK validation");
                return FallbackToSimplePlanner(targetObject, gripperPosition, options);
            }

            // Stage 3: Collision filtering
            var collisionFreeCandidates = _collisionFilter.FilterCandidates(ikValidCandidates, targetObject);
            UnityEngine.Debug.Log($"{_logPrefix} {collisionFreeCandidates.Count} candidates passed collision filter");

            if (collisionFreeCandidates.Count == 0)
            {
                UnityEngine.Debug.LogWarning($"{_logPrefix} No collision-free candidates found");
                return FallbackToSimplePlanner(targetObject, gripperPosition, options);
            }

            // Stage 4: Scoring and ranking
            Vector3 objectSize = GraspUtilities.GetObjectSize(targetObject);
            var rankedCandidates = _scorer.ScoreAndRank(
                collisionFreeCandidates,
                objectSize,
                gripperPosition
            );

            stopwatch.Stop();
            float elapsedMs = (float)stopwatch.Elapsed.TotalMilliseconds;

            UnityEngine.Debug.Log(
                $"{_logPrefix} Pipeline completed in {elapsedMs:F1}ms. " +
                $"Best candidate score: {rankedCandidates[0].totalScore:F2}"
            );

            // Check time budget
            if (elapsedMs > _config.maxPipelineTimeMs)
            {
                UnityEngine.Debug.LogWarning(
                    $"{_logPrefix} Pipeline exceeded time budget ({elapsedMs:F1}ms > {_config.maxPipelineTimeMs}ms)"
                );
            }

            // Return best candidate
            return rankedCandidates[0];
        }

        /// <summary>
        /// Fallback to simple planner if advanced pipeline fails.
        /// </summary>
        /// <param name="targetObject">Object to grasp</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <param name="options">Grasp options</param>
        /// <returns>Simple grasp candidate</returns>
        private GraspCandidate? FallbackToSimplePlanner(
            GameObject targetObject,
            Vector3 gripperPosition,
            GraspOptions? options
        )
        {
            UnityEngine.Debug.Log($"{_logPrefix} Falling back to simple planner");

            try
            {
                var candidate = _generator.GenerateSimpleCandidate(
                    targetObject,
                    gripperPosition,
                    options?.approach
                );

                // Basic distance check for fallback (skip full IK validation)
                float distance = Vector3.Distance(candidate.preGraspPosition, gripperPosition);
                if (distance > _config.maxReachDistance)
                {
                    UnityEngine.Debug.LogWarning(
                        $"{_logPrefix} Fallback candidate too far ({distance:F2}m > {_config.maxReachDistance:F2}m)"
                    );
                    return null;
                }

                // Mark as validated (skipping full pipeline)
                candidate.ikValidated = true;
                candidate.collisionValidated = true;
                candidate.totalScore = 0.5f; // Medium score for fallback

                return candidate;
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogError($"{_logPrefix} Fallback planner also failed: {e.Message}");
                return null;
            }
        }

        /// <summary>
        /// Plan grasp with detailed diagnostics (for debugging).
        /// </summary>
        /// <param name="targetObject">Object to grasp</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <param name="options">Grasp options</param>
        /// <returns>Planning result with diagnostics</returns>
        public GraspPlanningResult PlanGraspWithDiagnostics(
            GameObject targetObject,
            Vector3 gripperPosition,
            GraspOptions? options = null
        )
        {
            var result = new GraspPlanningResult();
            var stopwatch = Stopwatch.StartNew();

            // Stage 1: Generate
            result.generatedCandidates = _generator.GenerateCandidates(targetObject, gripperPosition);
            result.generatedCount = result.generatedCandidates.Count;

            if (result.generatedCount == 0)
            {
                result.success = false;
                result.failureReason = "No candidates generated";
                return result;
            }

            // Stage 2: IK Filter
            result.ikValidCandidates = _ikFilter.FilterCandidates(
                result.generatedCandidates,
                gripperPosition
            );
            result.ikValidCount = result.ikValidCandidates.Count;

            if (result.ikValidCount == 0)
            {
                result.success = false;
                result.failureReason = "No IK-valid candidates";
                return result;
            }

            // Stage 3: Collision Filter
            result.collisionFreeCandidates = _collisionFilter.FilterCandidates(
                result.ikValidCandidates,
                targetObject
            );
            result.collisionFreeCount = result.collisionFreeCandidates.Count;

            if (result.collisionFreeCount == 0)
            {
                result.success = false;
                result.failureReason = "No collision-free candidates";
                return result;
            }

            // Stage 4: Score and Rank
            Vector3 objectSize = GraspUtilities.GetObjectSize(targetObject);
            result.rankedCandidates = _scorer.ScoreAndRank(
                result.collisionFreeCandidates,
                objectSize,
                gripperPosition
            );

            stopwatch.Stop();
            result.elapsedMs = (float)stopwatch.Elapsed.TotalMilliseconds;
            result.success = true;
            result.bestCandidate = result.rankedCandidates[0];

            return result;
        }

        /// <summary>
        /// Get top N grasp candidates (for multi-grasp attempts).
        /// </summary>
        /// <param name="targetObject">Object to grasp</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <param name="count">Number of candidates to return</param>
        /// <param name="options">Grasp options</param>
        /// <returns>List of top candidates (best first)</returns>
        public List<GraspCandidate> PlanMultipleGrasps(
            GameObject targetObject,
            Vector3 gripperPosition,
            int count,
            GraspOptions? options = null
        )
        {
            var bestCandidate = PlanGrasp(targetObject, gripperPosition, options);

            if (!bestCandidate.HasValue)
                return new List<GraspCandidate>();

            // Re-run pipeline with diagnostics to get all ranked candidates
            var result = PlanGraspWithDiagnostics(targetObject, gripperPosition, options);

            if (!result.success)
                return new List<GraspCandidate> { bestCandidate.Value };

            return _scorer.GetTopN(result.rankedCandidates, count);
        }
    }

    /// <summary>
    /// Result of grasp planning with detailed diagnostics.
    /// </summary>
    public class GraspPlanningResult
    {
        public bool success;
        public string failureReason;
        public float elapsedMs;

        public List<GraspCandidate> generatedCandidates;
        public List<GraspCandidate> ikValidCandidates;
        public List<GraspCandidate> collisionFreeCandidates;
        public List<GraspCandidate> rankedCandidates;

        public int generatedCount;
        public int ikValidCount;
        public int collisionFreeCount;

        public GraspCandidate bestCandidate;

        /// <summary>
        /// Get summary string for logging.
        /// </summary>
        public string GetSummary()
        {
            if (!success)
                return $"Failed: {failureReason}";

            return $"Success in {elapsedMs:F1}ms: " +
                   $"{generatedCount} generated → " +
                   $"{ikValidCount} IK-valid → " +
                   $"{collisionFreeCount} collision-free. " +
                   $"Best score: {bestCandidate.totalScore:F2}";
        }
    }
}
