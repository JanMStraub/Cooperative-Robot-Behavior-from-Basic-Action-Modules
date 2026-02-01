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
        private readonly Transform _endEffector; // Store for orientation consistency scoring

        private const string _logPrefix = "[GRASP_PLANNING_PIPELINE]";

        /// <summary>
        /// Initialize pipeline with robot configuration.
        /// </summary>
        /// <param name="config">Grasp planning configuration</param>
        /// <param name="joints">Robot joint articulation bodies</param>
        /// <param name="ikReferenceFrame">IK coordinate frame</param>
        /// <param name="endEffector">End effector transform</param>
        /// <param name="ikConfig">IK configuration (contains damping factor and other IK parameters)</param>
        public GraspPlanningPipeline(
            GraspConfig config,
            ArticulationBody[] joints,
            Transform ikReferenceFrame,
            Transform endEffector,
            IKConfig ikConfig
        )
        {
            _config = config;
            _endEffector = endEffector; // Cache for orientation scoring
            _generator = new GraspCandidateGenerator(config);
            _ikFilter = new GraspIKFilter(
                config,
                joints,
                ikReferenceFrame,
                endEffector,
                ikConfig,
                enableDebugLogging: false,
                enableInitializationValidation: true
            );
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
                UnityEngine.Debug.Log($"{_logPrefix} Advanced grasp planning with adaptive candidate count");

                // Adaptive candidate count based on time budget
                int adaptiveCandidateCount = ComputeAdaptiveCandidateCount(targetObject, gripperPosition);

                // Temporarily adjust config for this planning cycle
                int originalCount = _config.candidatesPerApproach;
                _config.candidatesPerApproach = adaptiveCandidateCount;

                // Advanced mode - generate multiple candidates
                candidates = _generator.GenerateCandidates(targetObject, gripperPosition);

                // Restore original config
                _config.candidatesPerApproach = originalCount;
            }

            UnityEngine.Debug.Log($"{_logPrefix} Generated {candidates.Count} candidates");
            LogApproachDistribution("Generated", candidates);

            if (candidates.Count == 0)
            {
                UnityEngine.Debug.LogWarning($"{_logPrefix} No candidates generated");
                return null;
            }

            // Stage 2: IK filtering
            var ikValidCandidates = _ikFilter.FilterCandidates(candidates, gripperPosition);
            UnityEngine.Debug.Log($"{_logPrefix} {ikValidCandidates.Count} candidates passed IK filter");
            LogApproachDistribution("IK-filtered", ikValidCandidates);

            if (ikValidCandidates.Count == 0)
            {
                UnityEngine.Debug.LogWarning($"{_logPrefix} No candidates passed IK validation");
                return FallbackToSimplePlanner(targetObject, gripperPosition, options);
            }

            // Stage 3: Collision filtering
            var collisionFreeCandidates = _collisionFilter.FilterCandidates(ikValidCandidates, targetObject);
            UnityEngine.Debug.Log($"{_logPrefix} {collisionFreeCandidates.Count} candidates passed collision filter");
            LogApproachDistribution("Collision-free", collisionFreeCandidates);

            if (collisionFreeCandidates.Count == 0)
            {
                UnityEngine.Debug.LogWarning($"{_logPrefix} No collision-free candidates found");
                return FallbackToSimplePlanner(targetObject, gripperPosition, options);
            }

            // Stage 4: Scoring and ranking (with orientation consistency check)
            Vector3 objectSize = GraspUtilities.GetObjectSize(targetObject);
            Quaternion currentGripperRotation = _endEffector != null ? _endEffector.rotation : Quaternion.identity;
            var rankedCandidates = _scorer.ScoreAndRank(
                collisionFreeCandidates,
                objectSize,
                gripperPosition,
                currentGripperRotation  // Pass current rotation for orientation consistency scoring
            );

            stopwatch.Stop();
            float elapsedMs = (float)stopwatch.Elapsed.TotalMilliseconds;

            var bestCandidate = rankedCandidates[0];
            UnityEngine.Debug.Log(
                $"{_logPrefix} Pipeline completed in {elapsedMs:F1}ms. " +
                $"Best candidate score: {bestCandidate.totalScore:F2}, " +
                $"Approach: {bestCandidate.approachType}, " +
                $"GraspPos: {bestCandidate.graspPosition}, " +
                $"PreGraspPos: {bestCandidate.preGraspPosition}"
            );
            LogTopCandidatesScores(rankedCandidates);

            // Check time budget
            if (elapsedMs > _config.maxPipelineTimeMs)
            {
                UnityEngine.Debug.LogWarning(
                    $"{_logPrefix} Pipeline exceeded time budget ({elapsedMs:F1}ms > {_config.maxPipelineTimeMs}ms)"
                );
            }

            // Return best candidate
            return bestCandidate;
        }

        /// <summary>
        /// Fallback to simple planner if advanced pipeline fails.
        /// Uses SimpleRobotController for more robust execution.
        /// </summary>
        /// <param name="targetObject">Object to grasp</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <param name="options">Grasp options</param>
        /// <returns>Simple grasp candidate with simplified execution flag</returns>
        private GraspCandidate? FallbackToSimplePlanner(
            GameObject targetObject,
            Vector3 gripperPosition,
            GraspOptions? options
        )
        {
            UnityEngine.Debug.Log($"{_logPrefix} Falling back to simple planner with SimpleRobotController");

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

                // Mark this candidate for simplified execution using SimpleRobotController
                // This flag will be checked by RobotController to use simpler motion control
                candidate.useSimplifiedExecution = true;

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

            // Stage 4: Score and Rank (with orientation consistency)
            Vector3 objectSize = GraspUtilities.GetObjectSize(targetObject);
            Quaternion currentGripperRotation = _endEffector != null ? _endEffector.rotation : Quaternion.identity;
            result.rankedCandidates = _scorer.ScoreAndRank(
                result.collisionFreeCandidates,
                objectSize,
                gripperPosition,
                currentGripperRotation
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

        /// <summary>
        /// Compute adaptive candidate count based on time budget and task complexity.
        /// Generates more candidates when we have time and the grasp is challenging.
        /// </summary>
        /// <param name="targetObject">Object to grasp</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <returns>Adaptive candidate count per approach</returns>
        private int ComputeAdaptiveCandidateCount(GameObject targetObject, Vector3 gripperPosition)
        {
            int baseCandidates = _config.candidatesPerApproach;

            // Factor 1: Time budget utilization (assume 50% budget available at start)
            float timeBudgetFactor = 1.5f; // Can generate more candidates if we have time

            // Factor 2: Object complexity (smaller objects = more candidates for precision)
            Vector3 objectSize = GraspUtilities.GetObjectSize(targetObject);
            float avgSize = (objectSize.x + objectSize.y + objectSize.z) / 3f;
            float complexityFactor = avgSize < 0.05f ? 1.5f : 1.0f; // Small objects need more attempts

            // Factor 3: Distance to object (farther = more candidates for IK diversity)
            float distance = Vector3.Distance(targetObject.transform.position, gripperPosition);
            float distanceFactor = distance > 0.5f ? 1.3f : 1.0f;

            // Compute adaptive count (clamp to reasonable range)
            int adaptiveCount = Mathf.RoundToInt(baseCandidates * timeBudgetFactor * complexityFactor * distanceFactor);
            adaptiveCount = Mathf.Clamp(adaptiveCount, baseCandidates, baseCandidates * 3);

            return adaptiveCount;
        }

        /// <summary>
        /// Log approach type distribution for debugging.
        /// </summary>
        /// <param name="stage">Pipeline stage name</param>
        /// <param name="candidates">Candidates to analyze</param>
        private void LogApproachDistribution(string stage, List<GraspCandidate> candidates)
        {
            int topCount = 0;
            int sideCount = 0;
            int frontCount = 0;

            foreach (var candidate in candidates)
            {
                switch (candidate.approachType)
                {
                    case GraspApproach.Top:
                        topCount++;
                        break;
                    case GraspApproach.Side:
                        sideCount++;
                        break;
                    case GraspApproach.Front:
                        frontCount++;
                        break;
                }
            }

            UnityEngine.Debug.Log(
                $"{_logPrefix} [{stage}] Approach distribution: " +
                $"Top={topCount}, Side={sideCount}, Front={frontCount}"
            );
        }

        /// <summary>
        /// Log top 3 candidate scores for debugging.
        /// </summary>
        /// <param name="rankedCandidates">Sorted candidates</param>
        private void LogTopCandidatesScores(List<GraspCandidate> rankedCandidates)
        {
            int showCount = Mathf.Min(3, rankedCandidates.Count);
            UnityEngine.Debug.Log($"{_logPrefix} Top {showCount} candidates (showing weighted contributions):");

            for (int i = 0; i < showCount; i++)
            {
                var c = rankedCandidates[i];

                // Calculate weighted contributions
                float ikWeighted = c.ikScore * _config.ikScoreWeight;
                float approachWeighted = c.approachScore * _config.approachScoreWeight;
                float depthWeighted = c.depthScore * _config.depthScoreWeight;
                float stabilityWeighted = c.stabilityScore * _config.stabilityScoreWeight;
                float antipodalWeighted = c.antipodalScore * _config.antipodalScoreWeight;

                UnityEngine.Debug.Log(
                    $"{_logPrefix}   #{i + 1}: {c.approachType} - Total={c.totalScore:F2}\n" +
                    $"{_logPrefix}      Raw scores: IK={c.ikScore:F3}, Approach={c.approachScore:F3}, " +
                    $"Depth={c.depthScore:F3}, Stability={c.stabilityScore:F3}, Antipodal={c.antipodalScore:F3}\n" +
                    $"{_logPrefix}      Weighted:   IK={ikWeighted:F3}({_config.ikScoreWeight:F1}x), " +
                    $"Approach={approachWeighted:F3}({_config.approachScoreWeight:F1}x), " +
                    $"Depth={depthWeighted:F3}({_config.depthScoreWeight:F1}x), " +
                    $"Stability={stabilityWeighted:F3}({_config.stabilityScoreWeight:F1}x), " +
                    $"Antipodal={antipodalWeighted:F3}({_config.antipodalScoreWeight:F1}x)"
                );
            }
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
