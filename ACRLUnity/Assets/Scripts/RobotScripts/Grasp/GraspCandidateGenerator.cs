using UnityEngine;
using Configuration;
using System.Collections.Generic;

namespace Robotics.Grasp
{
    /// <summary>
    /// Generates multiple grasp candidates per approach type with adaptive positioning.
    /// Samples different angles, depths, and distances to create diverse candidate set.
    /// </summary>
    public class GraspCandidateGenerator
    {
        private readonly GraspConfig _config;
        private readonly System.Random _random;

        /// <summary>
        /// Initialize generator with configuration.
        /// </summary>
        /// <param name="config">Grasp planning configuration</param>
        /// <param name="seed">Random seed for reproducibility (0 = use system time)</param>
        public GraspCandidateGenerator(GraspConfig config, int seed = 0)
        {
            _config = config;
            _random = seed == 0 ? new System.Random() : new System.Random(seed);
        }

        /// <summary>
        /// Generate all grasp candidates for a target object.
        /// Creates multiple candidates per enabled approach type.
        /// </summary>
        /// <param name="targetObject">Object to grasp</param>
        /// <param name="gripperPosition">Current gripper position (for approach planning)</param>
        /// <returns>List of grasp candidates (unvalidated)</returns>
        public List<GraspCandidate> GenerateCandidates(GameObject targetObject, Vector3 gripperPosition)
        {
            var candidates = new List<GraspCandidate>();
            Vector3 objectPosition = targetObject.transform.position;
            Vector3 objectSize = GraspUtilities.GetObjectSize(targetObject);

            // Generate candidates for each enabled approach type
            foreach (var approachSetting in _config.enabledApproaches)
            {
                if (!approachSetting.enabled)
                    continue;

                var approachCandidates = GenerateCandidatesForApproach(
                    approachSetting.approachType,
                    objectPosition,
                    objectSize,
                    gripperPosition
                );

                candidates.AddRange(approachCandidates);
            }

            return candidates;
        }

        /// <summary>
        /// Generate candidates for a specific approach type.
        /// Samples variations in distance, angle, and depth.
        /// </summary>
        /// <param name="approach">Approach type</param>
        /// <param name="objectPosition">Object center position</param>
        /// <param name="objectSize">Object dimensions</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <returns>List of candidates for this approach</returns>
        private List<GraspCandidate> GenerateCandidatesForApproach(
            GraspApproach approach,
            Vector3 objectPosition,
            Vector3 objectSize,
            Vector3 gripperPosition
        )
        {
            var candidates = new List<GraspCandidate>();
            float basePreGraspDistance = _config.CalculatePreGraspDistance(objectSize);

            for (int i = 0; i < _config.candidatesPerApproach; i++)
            {
                // Sample variation parameters (now object-size-aware)
                float distanceVariation = SampleDistanceVariation(basePreGraspDistance);
                float angleVariation = SampleAngleVariation();
                float depthVariation = SampleDepthVariation(objectSize);

                // Generate candidate with variations
                var candidate = GenerateSingleCandidate(
                    approach,
                    objectPosition,
                    objectSize,
                    distanceVariation,
                    angleVariation,
                    depthVariation
                );

                candidates.Add(candidate);
            }

            return candidates;
        }

        /// <summary>
        /// Generate a single grasp candidate with specified variations.
        /// Enhanced with approach direction perturbations and antipodal grasp consideration.
        /// </summary>
        /// <param name="approach">Approach type</param>
        /// <param name="objectPosition">Object center position</param>
        /// <param name="objectSize">Object dimensions</param>
        /// <param name="preGraspDistance">Distance for pre-grasp waypoint</param>
        /// <param name="angleOffset">Angular offset in degrees</param>
        /// <param name="depthOffset">Depth offset as fraction of object size</param>
        /// <returns>Generated grasp candidate</returns>
        private GraspCandidate GenerateSingleCandidate(
            GraspApproach approach,
            Vector3 objectPosition,
            Vector3 objectSize,
            float preGraspDistance,
            float angleOffset,
            float depthOffset
        )
        {
            Vector3 graspPosition;
            Quaternion graspRotation;
            Vector3 approachDirection;

            // Sample small perturbations for approach direction diversity
            float lateralPerturbation = (float)(_random.NextDouble() * 10.0 - 5.0); // ±5 degrees

            switch (approach)
            {
                case GraspApproach.Top:
                    // Top approach: gripper approaches from above (approach direction is upward from object)
                    approachDirection = Vector3.up; // Direction gripper comes from
                    // Apply lateral perturbation to approach direction
                    Vector3 perturbedTopDir = Quaternion.Euler(lateralPerturbation, lateralPerturbation, 0f) * approachDirection;
                    // Grasp position is on top of object
                    graspPosition = objectPosition + Vector3.up * (objectSize.y * 0.5f + depthOffset);
                    // Gripper pointing down: The URDF gripper has a 90° Z-rotation baked in
                    // So we need: 180° X-rotation (point down) + 90° Z-rotation (compensate for gripper orientation)
                    graspRotation = Quaternion.Euler(180f + angleOffset, 0f, 90f + lateralPerturbation);
                    approachDirection = perturbedTopDir.normalized;
                    break;

                case GraspApproach.Side:
                    float sideSign = Mathf.Sign(Random.Range(-1f, 1f));
                    float sideOffset = objectSize.x * 0.5f + depthOffset;
                    approachDirection = Vector3.right * sideSign;
                    // Apply lateral perturbation
                    Vector3 perturbedSideDir = Quaternion.Euler(lateralPerturbation, 0f, lateralPerturbation) * approachDirection;
                    graspPosition = objectPosition + perturbedSideDir.normalized * sideOffset;
                    float sideAngle = sideSign > 0 ? -90f : 90f;
                    graspRotation = Quaternion.Euler(angleOffset + lateralPerturbation, sideAngle, 0f);
                    approachDirection = perturbedSideDir.normalized;
                    break;

                case GraspApproach.Front:
                    float frontSign = Mathf.Sign(Random.Range(-1f, 1f));
                    float frontOffset = objectSize.z * 0.5f + depthOffset;
                    approachDirection = Vector3.forward * frontSign;
                    // Apply lateral perturbation
                    Vector3 perturbedFrontDir = Quaternion.Euler(lateralPerturbation, lateralPerturbation, 0f) * approachDirection;
                    graspPosition = objectPosition + perturbedFrontDir.normalized * frontOffset;
                    float frontAngle = frontSign > 0 ? 180f : 0f;
                    graspRotation = Quaternion.Euler(angleOffset + lateralPerturbation, frontAngle, 0f);
                    approachDirection = perturbedFrontDir.normalized;
                    break;

                default:
                    graspPosition = objectPosition;
                    graspRotation = Quaternion.identity;
                    approachDirection = Vector3.up;
                    break;
            }

            // Calculate pre-grasp position
            Vector3 preGraspPosition = graspPosition + approachDirection * preGraspDistance;

            // Calculate retreat position
            Vector3 retreatPosition = graspPosition;
            if (_config.enableRetreat)
            {
                float retreatDistance = _config.CalculateRetreatDistance(objectSize);
                retreatPosition = graspPosition + _config.retreatDirection.normalized * retreatDistance;
            }

            // Create candidate
            var candidate = GraspCandidate.Create(
                preGraspPosition,
                graspRotation,
                graspPosition,
                graspRotation,
                approach
            );

            // Update additional fields
            candidate.retreatPosition = retreatPosition;
            candidate.retreatRotation = graspRotation;
            candidate.approachDistance = preGraspDistance;
            candidate.graspDepth = depthOffset;
            candidate.contactPointEstimate = graspPosition;
            candidate.approachDirection = approachDirection;

            // Compute antipodal grasp quality
            candidate.antipodalScore = ComputeAntipodalScore(graspPosition, objectPosition, objectSize, approachDirection);

            return candidate;
        }

        /// <summary>
        /// Compute antipodal grasp quality score.
        /// Measures how well gripper fingers will oppose each other across object center.
        /// </summary>
        /// <param name="graspPosition">Grasp position</param>
        /// <param name="objectCenter">Object center position</param>
        /// <param name="objectSize">Object dimensions</param>
        /// <param name="approachDirection">Approach direction (gripper closing axis)</param>
        /// <returns>Antipodal score (0-1, higher = better force closure)</returns>
        private float ComputeAntipodalScore(Vector3 graspPosition, Vector3 objectCenter, Vector3 objectSize, Vector3 approachDirection)
        {
            // Calculate expected finger positions (assuming gripper opens along approach direction)
            float gripperWidth = _config.gripperGeometry.maxWidth;
            Vector3 perpendicular = Vector3.Cross(approachDirection, Vector3.up);
            if (perpendicular.magnitude < 0.1f)
                perpendicular = Vector3.Cross(approachDirection, Vector3.right);
            perpendicular = perpendicular.normalized;

            // Finger positions
            Vector3 finger1Pos = graspPosition + perpendicular * (gripperWidth * 0.5f);
            Vector3 finger2Pos = graspPosition - perpendicular * (gripperWidth * 0.5f);

            // Check if object center lies between fingers (good antipodal grasp)
            Vector3 toCenter = objectCenter - graspPosition;
            float centerProjection = Vector3.Dot(toCenter, perpendicular);

            // Ideal: center is on the grasp line (projection near zero)
            float centerAlignment = 1.0f - Mathf.Clamp01(Mathf.Abs(centerProjection) / (objectSize.magnitude * 0.5f));

            // Check if fingers are symmetric relative to object center
            float dist1 = Vector3.Distance(finger1Pos, objectCenter);
            float dist2 = Vector3.Distance(finger2Pos, objectCenter);
            float symmetry = 1.0f - Mathf.Abs(dist1 - dist2) / (dist1 + dist2 + 0.001f);

            // Combine factors
            return (centerAlignment * 0.6f + symmetry * 0.4f);
        }

        /// <summary>
        /// Sample distance variation around base pre-grasp distance.
        /// </summary>
        /// <param name="baseDistance">Base pre-grasp distance</param>
        /// <returns>Varied distance within config bounds</returns>
        private float SampleDistanceVariation(float baseDistance)
        {
            // Vary ±30% around base distance
            float variation = (float)(_random.NextDouble() * 0.6 - 0.3);
            float varied = baseDistance * (1f + variation);
            return Mathf.Clamp(varied, _config.minPreGraspDistance, _config.maxPreGraspDistance);
        }

        /// <summary>
        /// Sample angular variation for grasp orientation.
        /// </summary>
        /// <returns>Angle offset in degrees (±15°)</returns>
        private float SampleAngleVariation()
        {
            return (float)(_random.NextDouble() * 30.0 - 15.0);
        }

        /// <summary>
        /// Sample depth variation for grasp penetration (now object-size-aware).
        /// </summary>
        /// <param name="objectSize">Size of target object</param>
        /// <returns>Depth offset in meters</returns>
        private float SampleDepthVariation(Vector3 objectSize)
        {
            // Calculate object-size-aware base depth
            float avgSize = (objectSize.x + objectSize.y + objectSize.z) / 3f;
            float baseDepth = _config.targetGraspDepth * avgSize;

            // Variation scales with object size (±20% of average size)
            float variation = avgSize * 0.2f;
            return baseDepth + (float)(_random.NextDouble() * variation * 2f - variation);
        }

        /// <summary>
        /// Generate a single optimized candidate using simple grasp calculation.
        /// Useful for fallback when advanced planning is disabled.
        /// </summary>
        /// <param name="targetObject">Object to grasp</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <param name="approach">Preferred approach type (null for auto-detect)</param>
        /// <returns>Single grasp candidate</returns>
        public GraspCandidate GenerateSimpleCandidate(
            GameObject targetObject,
            Vector3 gripperPosition,
            GraspApproach? approach = null
        )
        {
            Vector3 objectPosition = targetObject.transform.position;
            Vector3 objectSize = GraspUtilities.GetObjectSize(targetObject);

            // Determine approach if not specified
            GraspApproach selectedApproach = approach ?? GraspUtilities.DetermineOptimalApproach(
                objectPosition,
                gripperPosition,
                objectSize
            );

            // Calculate basic grasp pose
            var (graspPosition, graspRotation) = GraspUtilities.CalculateBasicGraspPose(
                objectPosition,
                objectSize,
                gripperPosition,
                selectedApproach
            );

            // Calculate approach direction
            Vector3 approachDirection = selectedApproach switch
            {
                GraspApproach.Top => Vector3.up,
                GraspApproach.Side => Vector3.right * (gripperPosition.x > objectPosition.x ? 1f : -1f),
                GraspApproach.Front => Vector3.forward * (gripperPosition.z > objectPosition.z ? 1f : -1f),
                _ => Vector3.up
            };

            // Calculate pre-grasp position
            float preGraspDistance = _config.CalculatePreGraspDistance(objectSize);
            Vector3 preGraspPosition = graspPosition + approachDirection * preGraspDistance;

            // Create candidate
            var candidate = GraspCandidate.Create(
                preGraspPosition,
                graspRotation,
                graspPosition,
                graspRotation,
                selectedApproach
            );

            // Add retreat position
            if (_config.enableRetreat)
            {
                float retreatDistance = _config.CalculateRetreatDistance(objectSize);
                candidate.retreatPosition = graspPosition + _config.retreatDirection.normalized * retreatDistance;
                candidate.retreatRotation = graspRotation;
            }

            candidate.preGraspGripperWidth = 1.0f; // Fully open for approach
            candidate.graspGripperWidth = 0.0f;    // Fully closed for grasp
            candidate.approachDistance = preGraspDistance;
            candidate.approachDirection = approachDirection;

            return candidate;
        }
    }
}
