using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Configuration;
using UnityEngine;

namespace Robotics.Grasp
{
    /// <summary>
    /// Generates multiple grasp candidates per approach type with adaptive positioning.
    /// Optimized for memory allocation and local-space object orientation.
    /// </summary>
    public class GraspCandidateGenerator
    {
        private readonly GraspConfig _config;
        private readonly System.Random _random;

        private const string _logPrefix = "[GRASP_CANDIDATE_GENERATOR]";

        // Cached basis vectors for local space calculations to avoid new struct allocs
        private static readonly Vector3[] ApproachAxes =
        {
            Vector3.up,
            Vector3.right,
            Vector3.forward,
        };

        public GraspCandidateGenerator(GraspConfig config, int seed = 0)
        {
            _config = config;
            _random = seed == 0 ? new System.Random() : new System.Random(seed);
        }

        /// <summary>
        /// Generate all grasp candidates for a target object.
        /// </summary>
        public List<GraspCandidate> GenerateCandidates(
            GameObject targetObject,
            Vector3 gripperPosition
        )
        {
            // 1. Calculate total expected capacity to avoid List resizing
            int totalCandidates = 0;
            foreach (var approach in _config.enabledApproaches)
            {
                if (approach.enabled)
                    totalCandidates += _config.candidatesPerApproach;
            }

            var candidates = new List<GraspCandidate>(totalCandidates);

            // Cache object transform data once
            Transform objTransform = targetObject.transform;
            Vector3 objectPosition = objTransform.position;
            Quaternion objectRotation = objTransform.rotation;

            // Get size (assuming OBB or AABB logic, ideally this comes from a Collider)
            Vector3 objectSize = GraspUtilities.GetObjectSize(targetObject);
            float basePreGraspDist = _config.CalculatePreGraspDistance(objectSize);

            // 2. Single loop generation
            foreach (var approachSetting in _config.enabledApproaches)
            {
                if (!approachSetting.enabled)
                    continue;

                GenerateCandidatesForApproach(
                    candidates, // Pass list by reference to avoid alloc
                    approachSetting.approachType,
                    objectPosition,
                    objectRotation,
                    objectSize,
                    basePreGraspDist
                );
            }

            return candidates;
        }

        private void GenerateCandidatesForApproach(
            List<GraspCandidate> results,
            GraspApproach approach,
            Vector3 objPos,
            Quaternion objRot,
            Vector3 objSize,
            float basePreGraspDist
        )
        {
            // Determine the local axis for this approach (Top=Y, Side=X, Front=Z)
            GetApproachBasis(
                approach,
                objRot,
                out Vector3 approachAxisWorld,
                out Vector3 approachTangentWorld
            );

            // Determine the dimension magnitude along that axis
            float dimensionOnAxis = GetDimensionOnAxis(approach, objSize);

            for (int i = 0; i < _config.candidatesPerApproach; i++)
            {
                // Sample variations
                float distVar = SampleDistanceVariation(basePreGraspDist);
                float angleVar = SampleAngleVariation(); // Degrees
                float depthVar = SampleDepthVariation(objSize);

                // Perturb the approach direction vector (Cone sampling)
                Vector3 perturbedApproachDir = PerturbDirection(
                    approachAxisWorld,
                    approachTangentWorld
                );

                // Calculate Grasp Position (Surface + Depth)
                // We move from center -> along axis -> minus depth
                Vector3 graspPoint =
                    objPos + (perturbedApproachDir * ((dimensionOnAxis * 0.5f) + depthVar));

                // Debug first candidate only
                if (i == 0 && approach == GraspApproach.Top)
                {
                    UnityEngine.Debug.Log($"{_logPrefix} {approach} approach: objPos={objPos}, dimensionOnAxis={dimensionOnAxis}, " +
                        $"depthVar={depthVar}, perturbedApproachDir={perturbedApproachDir}, graspPoint={graspPoint}");
                }

                // Calculate Rotation
                // Use gripper-specific rotations (URDF has 90° Z-rotation baked in)
                Quaternion graspRotation = CalculateGripperRotation(
                    approach,
                    perturbedApproachDir,
                    approachTangentWorld,
                    objRot,
                    angleVar
                );

                // Pre-grasp Position
                Vector3 preGraspPos = graspPoint + (perturbedApproachDir * distVar);

                // Retreat Position
                Vector3 retreatPos = graspPoint;
                if (_config.enableRetreat)
                {
                    // Assuming retreat is simply backing out the way we came
                    retreatPos =
                        graspPoint
                        + (perturbedApproachDir * _config.CalculateRetreatDistance(objSize));
                }

                // Construct Candidate
                var candidate = GraspCandidate.Create(
                    preGraspPos,
                    graspRotation,
                    graspPoint,
                    graspRotation,
                    approach
                );

                candidate.retreatPosition = retreatPos;
                candidate.retreatRotation = graspRotation;
                candidate.approachDistance = distVar;
                candidate.graspDepth = depthVar;
                candidate.contactPointEstimate = graspPoint;
                candidate.approachDirection = perturbedApproachDir;

                // Compute Score (approach-aware antipodal scoring)
                candidate.antipodalScore = ComputeAntipodalScore(
                    graspPoint,
                    graspRotation,
                    objPos,
                    objSize,
                    approach
                );

                results.Add(candidate);
            }
        }

        /// <summary>
        /// Calculates the primary approach axis and a reference tangent in World Space,
        /// respecting the object's rotation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void GetApproachBasis(
            GraspApproach approach,
            Quaternion objRot,
            out Vector3 axis,
            out Vector3 tangent
        )
        {
            switch (approach)
            {
                case GraspApproach.Top: // Local Y
                    axis = objRot * Vector3.up;
                    tangent = objRot * Vector3.right; // Arbitrary tangent for roll reference
                    break;
                case GraspApproach.Side: // Local X
                    // Determine which side (left or right) randomly
                    float sideSign = _random.NextDouble() > 0.5 ? 1f : -1f;
                    axis = objRot * (Vector3.right * sideSign);
                    tangent = objRot * Vector3.up;
                    break;
                case GraspApproach.Front: // Local Z
                    float frontSign = _random.NextDouble() > 0.5 ? 1f : -1f;
                    axis = objRot * (Vector3.forward * frontSign);
                    tangent = objRot * Vector3.up;
                    break;
                default:
                    axis = objRot * Vector3.up;
                    tangent = objRot * Vector3.right;
                    break;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float GetDimensionOnAxis(GraspApproach approach, Vector3 size)
        {
            return approach switch
            {
                GraspApproach.Top => size.y,
                GraspApproach.Side => size.x,
                GraspApproach.Front => size.z,
                _ => size.y,
            };
        }

        /// <summary>
        /// Perturbs a direction vector within a small cone.
        /// </summary>
        private Vector3 PerturbDirection(Vector3 mainAxis, Vector3 tangent)
        {
            float perturbationAngle = (float)((_random.NextDouble() * 2.0 - 1.0) * _config.angleVariationRange);
            float perturbationRoll = (float)(_random.NextDouble() * 360.0);

            Quaternion rot = Quaternion.AngleAxis(perturbationAngle, tangent);
            Quaternion roll = Quaternion.AngleAxis(perturbationRoll, mainAxis);
            return (roll * rot * mainAxis).normalized;
        }

        /// <summary>
        /// Calculate gripper rotation for approach type.
        /// Accounts for URDF gripper coordinate frame (90° Z-rotation baked in).
        /// </summary>
        /// <param name="approach">Approach type</param>
        /// <param name="approachDir">World-space approach direction</param>
        /// <param name="tangent">World-space tangent for roll reference</param>
        /// <param name="objRot">Object rotation</param>
        /// <param name="angleVar">Angle variation in degrees</param>
        /// <returns>Gripper rotation quaternion</returns>
        private Quaternion CalculateGripperRotation(
            GraspApproach approach,
            Vector3 approachDir,
            Vector3 tangent,
            Quaternion objRot,
            float angleVar
        )
        {
            // Use approach-specific Euler angles to match expected gripper orientation
            // The gripper's coordinate frame expects specific rotations for each approach

            Quaternion baseRotation;

            switch (approach)
            {
                case GraspApproach.Top:
                    // Top approach: gripper points down
                    // X-rotation: 180° (point down) + variation
                    // Z-rotation: 90° (URDF compensation)
                    baseRotation = Quaternion.Euler(180f + angleVar, 0f, 90f);
                    break;

                case GraspApproach.Side:
                    // Side approach: gripper points horizontally
                    // Transform approach direction to object's local space to determine side
                    Vector3 localApproachDir = Quaternion.Inverse(objRot) * approachDir;
                    float sideAngle = localApproachDir.x > 0 ? -90f : 90f;
                    baseRotation = Quaternion.Euler(angleVar, sideAngle, 0f);
                    break;

                case GraspApproach.Front:
                    // Front approach: gripper points horizontally
                    // Transform approach direction to object's local space to determine front/back
                    Vector3 localFrontDir = Quaternion.Inverse(objRot) * approachDir;
                    float frontAngle = localFrontDir.z > 0 ? 180f : 0f;
                    baseRotation = Quaternion.Euler(angleVar, frontAngle, 0f);
                    break;

                default:
                    baseRotation = Quaternion.identity;
                    break;
            }

            // Apply object rotation to align with object's local frame
            return objRot * baseRotation;
        }

        /// <summary>
        /// Compute antipodal grasp quality score.
        /// Approach-aware: Top approaches use centering-only scoring since traditional
        /// antipodal (opposing contact points) concept doesn't apply to top-down grasps.
        /// </summary>
        private float ComputeAntipodalScore(
            Vector3 graspPos,
            Quaternion graspRot,
            Vector3 objCenter,
            Vector3 objSize,
            GraspApproach approach
        )
        {
            Vector3 toCenter = objCenter - graspPos;

            // For Top approaches, use centering-based scoring instead of antipodal
            // Top-down grasps on flat surfaces don't have traditional antipodal contact points
            if (approach == GraspApproach.Top)
            {
                // Score based on horizontal centering (XZ plane)
                // A good top grasp is centered over the object
                Vector2 horizontalOffset = new Vector2(toCenter.x, toCenter.z);
                float maxHorizontalExtent = Mathf.Max(objSize.x, objSize.z) * 0.5f;
                float centeringScore = 1.0f - Mathf.Clamp01(horizontalOffset.magnitude / maxHorizontalExtent);

                // Bonus for being directly above (vertical alignment)
                float verticalAlignment = Mathf.Abs(Vector3.Dot(toCenter.normalized, Vector3.down));
                float verticalScore = Mathf.Clamp01(verticalAlignment);

                // Top approaches get a baseline score since they're inherently good for table-top objects
                return 0.3f + (centeringScore * 0.4f) + (verticalScore * 0.3f);
            }

            // For Side/Front approaches, use traditional antipodal scoring
            Vector3 closingAxis = graspRot * Vector3.right;
            Vector3 approachAxis = graspRot * Vector3.forward;

            // Check alignment: gripper should point at object center
            float alignmentDot = Vector3.Dot(toCenter.normalized, approachAxis);
            float pointingScore = Mathf.Clamp01(alignmentDot);

            // Centering: grasp should be centered on object
            float distFromCenterLine = Vector3.Cross(toCenter, approachAxis).magnitude;
            float sideGraspCenteringScore =
                1.0f - Mathf.Clamp01(distFromCenterLine / (Mathf.Max(objSize.x, objSize.z) * 0.5f));

            return (pointingScore * 0.6f) + (sideGraspCenteringScore * 0.4f);
        }

        // --- Helpers ---

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float SampleDistanceVariation(float baseDist)
        {
            float variation = (float)((_random.NextDouble() * 2.0 - 1.0) * _config.distanceVariationRange);
            return Mathf.Clamp(
                baseDist * (1f + variation),
                _config.minPreGraspDistance,
                _config.maxPreGraspDistance
            );
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float SampleAngleVariation()
        {
            return (float)((_random.NextDouble() * 2.0 - 1.0) * _config.angleVariationRange);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float SampleDepthVariation(Vector3 objSize)
        {
            float avgSize = (objSize.x + objSize.y + objSize.z) / 3f;
            float baseDepth = _config.targetGraspDepth * avgSize;
            float variation = avgSize * _config.depthVariationRange;
            return baseDepth + (float)(_random.NextDouble() * variation * 2f - variation);
        }

        /// <summary>
        /// Legacy fallback for simple candidate generation.
        /// </summary>
        public GraspCandidate GenerateSimpleCandidate(
            GameObject targetObject,
            Vector3 gripperPosition,
            GraspApproach? approach = null
        )
        {
            Vector3 objPos = targetObject.transform.position;
            Vector3 objSize = GraspUtilities.GetObjectSize(targetObject);

            GraspApproach selectedApproach =
                approach
                ?? GraspUtilities.DetermineOptimalApproach(objPos, gripperPosition, objSize);

            // Use the improved helper to get clean rotation logic even for simple candidates
            GetApproachBasis(
                selectedApproach,
                targetObject.transform.rotation,
                out Vector3 approachDir,
                out Vector3 tangent
            );

            // Simple center grasp
            float dim = GetDimensionOnAxis(selectedApproach, objSize);
            Vector3 graspPos = objPos + (approachDir * (dim * 0.5f)); // Surface grasp

            // Orient gripper using same calculation as main generator
            Quaternion graspRot = CalculateGripperRotation(
                selectedApproach,
                approachDir,
                tangent,
                targetObject.transform.rotation,
                0f // No angle variation for simple candidate
            );

            float preGraspDist = _config.CalculatePreGraspDistance(objSize);
            Vector3 preGraspPos = graspPos + (approachDir * preGraspDist);

            var candidate = GraspCandidate.Create(
                preGraspPos,
                graspRot,
                graspPos,
                graspRot,
                selectedApproach
            );

            if (_config.enableRetreat)
            {
                candidate.retreatPosition =
                    graspPos + (approachDir * _config.CalculateRetreatDistance(objSize));
                candidate.retreatRotation = graspRot;
            }

            candidate.approachDistance = preGraspDist;
            candidate.approachDirection = approachDir;

            return candidate;
        }
    }
}
