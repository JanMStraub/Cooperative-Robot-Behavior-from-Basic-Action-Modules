using System.Collections.Generic;
using Configuration;
using UnityEngine;

namespace Robotics.Grasp
{
    /// <summary>
    /// Filters grasp candidates based on IK reachability and quality.
    /// Optimized for high-performance runtime evaluation with zero GC allocations.
    /// Combines mathematical kinematic derivation with optional validation for safety.
    /// </summary>
    public class GraspIKFilter
    {
        private readonly GraspConfig _config;
        private readonly IKSolver _ikSolver;
        private readonly ArticulationBody[] _joints;
        private readonly Transform _ikReferenceFrame;
        private readonly Transform _endEffector;
        private readonly Transform _robotBase;

        // --- Cached Kinematic Structure (Static Geometry) ---
        private readonly Vector3[] _localJointPositions; // Pos relative to parent
        private readonly Quaternion[] _localJointBindRotations; // Rot relative to parent at Angle=0
        private readonly Vector3[] _localJointAxes; // Axis relative to parent
        private readonly float[] _jointLowerLimits;
        private readonly float[] _jointUpperLimits;
        private Vector3 _endEffectorLocalOffset; // Offset from last joint (not readonly - set in CacheKinematicStructure)
        private readonly int _jointCount;

        // --- Runtime Buffers (To avoid GC allocation) ---
        private readonly float[] _bufferJointAngles;
        private readonly JointInfo[] _bufferJointInfos;

        // --- Debug and Validation Settings ---
        private readonly bool _enableDebugLogging;
        private readonly bool _enableInitializationValidation;
        private readonly string _logPrefix = "[ROBOT_CONTROLLER]";

        /// <summary>
        /// Initialize IK filter with robot configuration.
        /// </summary>
        /// <param name="config">Grasp planning configuration</param>
        /// <param name="joints">Robot joint articulation bodies</param>
        /// <param name="ikReferenceFrame">IK coordinate frame</param>
        /// <param name="endEffector">End effector transform</param>
        /// <param name="dampingFactor">IK solver damping factor</param>
        /// <param name="enableDebugLogging">Enable debug logging (disable in production)</param>
        /// <param name="enableInitializationValidation">Enable FK validation during initialization (recommended for first-time setup)</param>
        public GraspIKFilter(
            GraspConfig config,
            ArticulationBody[] joints,
            Transform ikReferenceFrame,
            Transform endEffector,
            float dampingFactor = 0.1f,
            bool enableDebugLogging = false,
            bool enableInitializationValidation = true
        )
        {
            _config = config ?? throw new System.ArgumentNullException(nameof(config));
            _joints = joints ?? throw new System.ArgumentNullException(nameof(joints));
            _ikReferenceFrame =
                ikReferenceFrame
                ?? throw new System.ArgumentNullException(nameof(ikReferenceFrame));
            _endEffector =
                endEffector ?? throw new System.ArgumentNullException(nameof(endEffector));

            _enableDebugLogging = enableDebugLogging;
            _enableInitializationValidation = enableInitializationValidation;

            _jointCount = joints.Length;
            _ikSolver = new IKSolver(_jointCount, dampingFactor);

            // 1. Identify Robot Base
            _robotBase = _joints[0].transform.parent;
            if (_robotBase == null)
                _robotBase = _joints[0].transform;

            // 2. Allocate Caches
            _localJointPositions = new Vector3[_jointCount];
            _localJointBindRotations = new Quaternion[_jointCount];
            _localJointAxes = new Vector3[_jointCount];
            _jointLowerLimits = new float[_jointCount];
            _jointUpperLimits = new float[_jointCount];

            // 3. Allocate Runtime Buffers (Zero GC during runtime)
            _bufferJointAngles = new float[_jointCount];
            _bufferJointInfos = new JointInfo[_jointCount];

            // 4. Cache Geometry
            CacheKinematicStructure();
        }

        /// <summary>
        /// Cache kinematic structure by temporarily resetting robot to zero configuration.
        /// This ensures accurate geometric offsets for FK computation.
        /// </summary>
        private void CacheKinematicStructure()
        {
            // Backup current angles
            float[] originalAngles = new float[_jointCount];
            for (int i = 0; i < _jointCount; i++)
            {
                if (_joints[i].jointPosition.dofCount > 0)
                    originalAngles[i] = _joints[i].jointPosition[0];
            }

            // Temporarily set all joints to zero configuration
            foreach (var joint in _joints)
            {
                if (joint.jointPosition.dofCount > 0)
                    joint.jointPosition = new ArticulationReducedSpace(0f);
            }
            UnityEngine.Physics.SyncTransforms();

            // Cache geometric parameters at zero configuration
            for (int i = 0; i < _jointCount; i++)
            {
                ArticulationBody joint = _joints[i];
                Transform parent = (i == 0) ? _robotBase : _joints[i - 1].transform;

                // A. Cache joint position relative to parent
                _localJointPositions[i] = parent.InverseTransformPoint(joint.transform.position);

                // B. Cache initial rotation in parent's frame
                _localJointBindRotations[i] =
                    Quaternion.Inverse(parent.rotation) * joint.transform.rotation;

                // C. Cache joint axis relative to joint's local frame (for proper FK)
                Vector3 worldAxis = joint.transform.rotation * joint.anchorRotation * Vector3.right;
                _localJointAxes[i] = joint
                    .transform.InverseTransformDirection(worldAxis)
                    .normalized;

                // D. Cache joint limits
                var drive = joint.xDrive;
                if (drive.lowerLimit < drive.upperLimit)
                {
                    _jointLowerLimits[i] = drive.lowerLimit * Mathf.Deg2Rad;
                    _jointUpperLimits[i] = drive.upperLimit * Mathf.Deg2Rad;
                }
                else
                {
                    _jointLowerLimits[i] = -float.MaxValue;
                    _jointUpperLimits[i] = float.MaxValue;
                }
            }

            // E. Cache end-effector offset from last joint
            _endEffectorLocalOffset = _joints[_jointCount - 1]
                .transform.InverseTransformPoint(_endEffector.position);

            // Restore original joint angles
            for (int i = 0; i < _jointCount; i++)
            {
                if (_joints[i].jointPosition.dofCount > 0)
                    _joints[i].jointPosition = new ArticulationReducedSpace(originalAngles[i]);
            }
            UnityEngine.Physics.SyncTransforms();

            // F. Validation (Optional - can be disabled in production)
            if (_enableInitializationValidation)
            {
                ValidateEndEffectorOffset();
                ValidateForwardKinematics(originalAngles);
            }
        }

        /// <summary>
        /// Validate end-effector offset to catch common configuration errors.
        /// </summary>
        private void ValidateEndEffectorOffset()
        {
            float eeOffsetMagnitude = _endEffectorLocalOffset.magnitude;
            if (eeOffsetMagnitude < 0.001f)
            {
                Debug.LogError(
                    $"{_logPrefix} End-effector offset is near-zero ({eeOffsetMagnitude:F6}m)! "
                        + $"This usually means endEffectorBase is assigned to Joint {_jointCount - 1} instead of the gripper link. "
                        + "IK validation will be INACCURATE."
                );
            }
            else if (eeOffsetMagnitude < 0.02f)
            {
                Debug.LogWarning(
                    $"{_logPrefix} End-effector offset is very small ({eeOffsetMagnitude:F4}m). "
                        + "Verify that endEffectorBase points to the gripper center."
                );
            }
            else if (_enableDebugLogging)
            {
                Debug.Log(
                    $"{_logPrefix} End-effector offset validated: {eeOffsetMagnitude:F4}m"
                );
            }
        }

        /// <summary>
        /// Validate FK computation by comparing with actual Unity transforms.
        /// This ensures the mathematical derivation matches physical reality.
        /// </summary>
        private void ValidateForwardKinematics(float[] testAngles)
        {
            // Compute FK using our cached method
            IKState computedState = ComputeForwardKinematicsInIKFrame(
                testAngles,
                _bufferJointInfos
            );

            // Get actual end-effector position from Unity
            Vector3 actualWorldPos = _endEffector.position;
            Vector3 actualIKPos = _ikReferenceFrame.InverseTransformPoint(actualWorldPos);

            // Compare
            float error = Vector3.Distance(computedState.Position, actualIKPos);

            if (error > 0.01f)
            {
                Debug.LogWarning(
                    $"{_logPrefix} FK validation error: {error:F4}m - "
                        + $"Computed: {computedState.Position}, Actual: {actualIKPos}. "
                        + "Kinematic caching may have issues."
                );
            }
            else if (_enableDebugLogging)
            {
                Debug.Log($"{_logPrefix} FK validation passed with error: {error:F6}m");
            }

            // Validate individual joints if debug logging enabled
            if (_enableDebugLogging)
            {
                for (int i = 0; i < _jointCount; i++)
                {
                    Vector3 actualJointWorld = _joints[i].transform.position;
                    Vector3 actualJointIK = _ikReferenceFrame.InverseTransformPoint(
                        actualJointWorld
                    );
                    float jointError = Vector3.Distance(
                        _bufferJointInfos[i].WorldPosition,
                        actualJointIK
                    );

                    if (jointError > 0.01f)
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} Joint {i} FK error: {jointError:F4}m - "
                                + $"Computed: {_bufferJointInfos[i].WorldPosition}, Actual: {actualJointIK}"
                        );
                    }
                }
            }
        }

        /// <summary>
        /// Filter candidates by IK reachability.
        /// Returns only candidates that pass both distance check and IK validation.
        /// Zero GC allocations during filtering (uses pre-allocated buffers).
        /// </summary>
        /// <param name="candidates">Candidates to filter</param>
        /// <param name="currentGripperPosition">Current gripper position (world space)</param>
        /// <returns>List of validated candidates with IK scores</returns>
        public List<GraspCandidate> FilterCandidates(
            List<GraspCandidate> candidates,
            Vector3 currentGripperPosition
        )
        {
            var validCandidates = new List<GraspCandidate>(candidates.Count);
            float maxReachSqr = _config.maxReachDistance * _config.maxReachDistance;
            int rejectedByDistance = 0;
            int rejectedByIK = 0;

            foreach (var candidate in candidates)
            {
                // Stage 1: Fast Distance Check (Squared magnitude avoids Sqrt)
                if (
                    (candidate.preGraspPosition - currentGripperPosition).sqrMagnitude > maxReachSqr
                )
                {
                    rejectedByDistance++;
                    continue;
                }

                // Stage 2: IK Validation
                if (_config.enableIKValidation)
                {
                    var validated = ValidateWithIK(candidate);
                    if (validated.ikValidated)
                    {
                        validCandidates.Add(validated);
                    }
                    else
                    {
                        rejectedByIK++;
                    }
                }
                else
                {
                    var accepted = candidate;
                    accepted.ikValidated = true;
                    accepted.ikScore = 1.0f;
                    validCandidates.Add(accepted);
                }
            }

            // Optional debug logging
            if (_enableDebugLogging && candidates.Count > 0)
            {
                Debug.Log(
                    $"{_logPrefix} Filtered {candidates.Count} candidates: "
                        + $"{validCandidates.Count} valid, {rejectedByDistance} out of reach, {rejectedByIK} IK failed"
                );
            }

            return validCandidates;
        }

        /// <summary>
        /// Validate candidate with full IK solver.
        /// Uses seeded IK (pre-grasp solution seeds grasp solution) for better performance.
        /// </summary>
        /// <param name="candidate">Candidate to validate</param>
        /// <returns>Candidate with IK validation flags and score</returns>
        private GraspCandidate ValidateWithIK(GraspCandidate candidate)
        {
            // 1. Attempt Pre-Grasp (seed with current robot state)
            CaptureCurrentJointAngles(_bufferJointAngles);
            var preGraspResult = AttemptIK(
                candidate.preGraspPosition,
                candidate.preGraspRotation,
                _bufferJointAngles
            );

            if (!preGraspResult.success)
            {
                if (_enableDebugLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Pre-grasp failed: error={preGraspResult.finalError:F4}m, iterations={preGraspResult.iterations}, threshold={_config.ikValidationThreshold:F4}m"
                    );
                }
                candidate.ikValidated = false;
                candidate.ikScore = 0f;
                return candidate;
            }

            // 2. Attempt Grasp (Seed with Pre-Grasp solution for better convergence)
            var graspResult = AttemptIK(
                candidate.graspPosition,
                candidate.graspRotation,
                preGraspResult.jointAngles
            );

            if (!graspResult.success)
            {
                if (_enableDebugLogging)
                {
                    Debug.Log(
                        $"{_logPrefix} Grasp failed: error={graspResult.finalError:F4}m, iterations={graspResult.iterations}, threshold={_config.ikValidationThreshold:F4}m"
                    );
                }
                candidate.ikValidated = false;
                candidate.ikScore = 0f;
                return candidate;
            }

            // 3. Compute Quality Score
            candidate.ikValidated = true;
            candidate.ikScore = ComputeIKQualityScore(preGraspResult, graspResult);

            if (_enableDebugLogging)
            {
                Debug.Log(
                    $"{_logPrefix} Candidate validated: preGraspError={preGraspResult.finalError:F4}m, graspError={graspResult.finalError:F4}m, score={candidate.ikScore:F3}"
                );
            }

            return candidate;
        }

        /// <summary>
        /// Fills the buffer with current actual robot angles.
        /// </summary>
        private void CaptureCurrentJointAngles(float[] buffer)
        {
            for (int i = 0; i < _jointCount; i++)
            {
                buffer[i] =
                    _joints[i].jointPosition.dofCount > 0 ? _joints[i].jointPosition[0] : 0f;
            }
        }

        /// <summary>
        /// Attempt to solve IK for a target pose.
        /// Uses pre-allocated buffers to avoid GC allocations.
        /// </summary>
        /// <param name="targetPosition">Target position (world space)</param>
        /// <param name="targetRotation">Target rotation (world space)</param>
        /// <param name="seedAngles">Initial joint angles (buffer will be modified)</param>
        /// <returns>IK result with success flag and quality metrics</returns>
        private IKResult AttemptIK(
            Vector3 targetPosition,
            Quaternion targetRotation,
            float[] seedAngles
        )
        {
            // Prepare Target in IK Frame
            Vector3 targetLocalPos = _ikReferenceFrame.InverseTransformPoint(targetPosition);
            Quaternion targetLocalRot =
                Quaternion.Inverse(_ikReferenceFrame.rotation) * targetRotation;
            IKState targetState = new IKState(targetLocalPos, targetLocalRot);

            // Use seed angles directly (will be modified in place)
            float[] currentAngles = seedAngles;

            int iterations = 0;
            float error = float.MaxValue;
            float lastError = float.MaxValue;
            int stallCount = 0;

            // Cache constants for performance
            float threshold = _config.ikValidationThreshold;
            float rotationTolerance = _config.ikRotationTolerance;
            float maxStep = _config.maxJointStepPerIteration;
            int maxIter = _config.maxIKValidationIterations;
            float orientationWeight = 0.05f; // Low weight - primarily position-based

            while (iterations < maxIter)
            {
                // 1. Forward Kinematics
                IKState currentState = ComputeForwardKinematicsInIKFrame(
                    currentAngles,
                    _bufferJointInfos
                );

                // 2. Check Convergence
                error = Vector3.Distance(currentState.Position, targetState.Position);

                if (error < threshold)
                {
                    if (
                        Quaternion.Angle(currentState.Rotation, targetState.Rotation)
                        < rotationTolerance
                    )
                        break; // Converged
                }

                // 3. Stall Detection - if error doesn't change, we're stuck
                if (Mathf.Abs(lastError - error) < 1e-5f)
                {
                    if (++stallCount > 5 && error > 0.05f)
                        break; // Abort if stuck far away
                }
                else
                {
                    stallCount = 0;
                }
                lastError = error;

                // 4. Solve Jacobian / Deltas
                var deltas = _ikSolver.ComputeJointDeltas(
                    currentState,
                    targetState,
                    _bufferJointInfos,
                    threshold,
                    orientationWeight
                );

                if (deltas == null)
                    break;

                // 5. Apply Deltas with limit clamping
                for (int i = 0; i < _jointCount; i++)
                {
                    float delta = (float)deltas[i];
                    if (delta == 0)
                        continue;

                    // Clamp step size
                    delta = Mathf.Clamp(delta, -maxStep, maxStep);

                    float newAngle = currentAngles[i] + delta;
                    bool hitLower = newAngle < _jointLowerLimits[i];
                    bool hitUpper = newAngle > _jointUpperLimits[i];

                    // Apply clamping to joint limits
                    if (hitLower || hitUpper)
                    {
                        currentAngles[i] = Mathf.Clamp(
                            newAngle,
                            _jointLowerLimits[i],
                            _jointUpperLimits[i]
                        );
                        // Future enhancement: Could signal to solver to reduce movement in this direction
                    }
                    else
                    {
                        currentAngles[i] = newAngle;
                    }
                }

                iterations++;
            }

            bool success = error < threshold;

            return new IKResult
            {
                success = success,
                iterations = iterations,
                finalError = error,
                jointAngles = currentAngles, // Returns the modified buffer
            };
        }

        /// <summary>
        /// Optimized Forward Kinematics using Vector3/Quaternion math.
        /// Calculates everything in RobotBase space, transforms to IK Frame only at the end.
        /// Uses pre-allocated buffer to avoid GC allocations.
        /// </summary>
        /// <param name="jointAngles">Joint angles in radians</param>
        /// <param name="jointInfosOut">Pre-allocated buffer for joint positions and axes in IK frame</param>
        /// <returns>End-effector state in IK frame</returns>
        private IKState ComputeForwardKinematicsInIKFrame(
            float[] jointAngles,
            JointInfo[] jointInfosOut
        )
        {
            // Start at Robot Base (Identity in Base Space)
            Vector3 currentPos = Vector3.zero;
            Quaternion currentRot = Quaternion.identity;

            for (int i = 0; i < _jointCount; i++)
            {
                // 1. Apply geometric offset from parent
                currentPos += currentRot * _localJointPositions[i];

                // 2. Calculate local rotation (Bind Rotation + Joint Angle)
                Quaternion jointRotation = _localJointBindRotations[i];
                if (_joints[i].jointPosition.dofCount > 0)
                {
                    jointRotation *= Quaternion.AngleAxis(
                        jointAngles[i] * Mathf.Rad2Deg,
                        _localJointAxes[i]
                    );
                }

                // 3. Accumulate rotation
                currentRot *= jointRotation;

                // 4. Store Joint Info for Jacobian (In Base Space temporarily)
                jointInfosOut[i].WorldPosition = currentPos;
                jointInfosOut[i].WorldAxis = currentRot * _localJointAxes[i];
            }

            // Apply End Effector Offset
            currentPos += currentRot * _endEffectorLocalOffset;

            // --- Transform from Base Space to IK Reference Frame ---
            // Construct cached transformation matrix
            Matrix4x4 baseToWorld = _robotBase.localToWorldMatrix;
            Matrix4x4 worldToIK = _ikReferenceFrame.worldToLocalMatrix;
            Matrix4x4 baseToIK = worldToIK * baseToWorld;

            // Transform End Effector
            Vector3 eePosIK = baseToIK.MultiplyPoint3x4(currentPos);
            Quaternion eeRotIK =
                (Quaternion.Inverse(_ikReferenceFrame.rotation) * _robotBase.rotation) * currentRot;

            // Transform Joint Infos (reuse buffer, no allocation)
            for (int i = 0; i < _jointCount; i++)
            {
                jointInfosOut[i].WorldPosition = baseToIK.MultiplyPoint3x4(
                    jointInfosOut[i].WorldPosition
                );
                jointInfosOut[i].WorldAxis = baseToIK
                    .MultiplyVector(jointInfosOut[i].WorldAxis)
                    .normalized;
            }

            return new IKState(eePosIK, eeRotIK);
        }

        /// <summary>
        /// Compute IK quality score from validation results.
        /// Balanced weights: error (40%), manipulability (30%), iterations (20%), limits (10%).
        /// </summary>
        /// <param name="preGraspResult">Pre-grasp IK result</param>
        /// <param name="graspResult">Grasp IK result</param>
        /// <returns>Quality score (0-1, higher is better)</returns>
        private float ComputeIKQualityScore(IKResult preGraspResult, IKResult graspResult)
        {
            float maxIterations = _config.maxIKValidationIterations;
            float avgIterations = (preGraspResult.iterations + graspResult.iterations) * 0.5f;

            // Factor 1: Iteration efficiency (fewer iterations is better)
            float iterationScore = Mathf.Clamp01(1f - (avgIterations / maxIterations));

            // Factor 2: Position accuracy (heavily penalize error)
            float avgError = (preGraspResult.finalError + graspResult.finalError) * 0.5f;
            float errorScore = Mathf.Clamp01(1.0f - (avgError * 20f));

            // Factor 3: Manipulability (ability to move from this configuration)
            float manipulability = ComputeManipulability(graspResult.jointAngles);

            // Factor 4: Joint limit proximity (penalize configurations near limits)
            float limitScore = ComputeJointLimitScore(graspResult.jointAngles);

            // Weighted combination (favor manipulability and low error for safer, more reliable grasps)
            return (iterationScore * 0.2f)
                + (errorScore * 0.4f)
                + (manipulability * 0.3f)
                + (limitScore * 0.1f);
        }

        /// <summary>
        /// Compute manipulability index using optimized inline calculations.
        /// Uses Frobenius norm as proxy for full Yoshikawa manipulability.
        /// Zero GC allocations (reuses buffer).
        /// </summary>
        /// <param name="jointAngles">Joint configuration</param>
        /// <returns>Manipulability score (0-1, higher is better)</returns>
        private float ComputeManipulability(float[] jointAngles)
        {
            // Re-run FK to get latest joint infos (reusing buffer)
            IKState state = ComputeForwardKinematicsInIKFrame(jointAngles, _bufferJointInfos);
            Vector3 eePos = state.Position;

            float sumSquared = 0f;

            // Simplified Manipulability (Frobenius norm proxy)
            // J = [cross(axis, r)] for each joint
            for (int i = 0; i < _jointCount; i++)
            {
                // Vector from joint to EE
                float rx = eePos.x - _bufferJointInfos[i].WorldPosition.x;
                float ry = eePos.y - _bufferJointInfos[i].WorldPosition.y;
                float rz = eePos.z - _bufferJointInfos[i].WorldPosition.z;

                Vector3 axis = _bufferJointInfos[i].WorldAxis;

                // Inline cross product (faster than Vector3.Cross - no allocation)
                float cx = axis.y * rz - axis.z * ry;
                float cy = axis.z * rx - axis.x * rz;
                float cz = axis.x * ry - axis.y * rx;

                sumSquared += (cx * cx) + (cy * cy) + (cz * cz);
            }

            // Normalize to 0-1 range
            return Mathf.Clamp01(Mathf.Sqrt(sumSquared) / (_jointCount * 0.5f));
        }

        /// <summary>
        /// Compute joint limit proximity score.
        /// Penalizes configurations where joints are near their limits.
        /// </summary>
        /// <param name="jointAngles">Joint configuration</param>
        /// <returns>Joint limit score (0-1, higher = farther from limits)</returns>
        private float ComputeJointLimitScore(float[] jointAngles)
        {
            float totalScore = 0f;
            int validJoints = 0;

            for (int i = 0; i < _jointCount; i++)
            {
                float min = _jointLowerLimits[i];
                float max = _jointUpperLimits[i];
                float range = max - min;

                // Ignore infinite/fixed joints
                if (range > 100f || range <= 0f)
                    continue;

                // Normalized position within range (0..1)
                float t = (jointAngles[i] - min) / range;

                // Distance from center (0.5), range 0..0.5
                float distFromCenter = Mathf.Abs(t - 0.5f);

                // Score: 1 at center, 0 at limits
                // Map 0..0.5 -> 1..0
                totalScore += 1.0f - (distFromCenter * 2.0f);
                validJoints++;
            }

            return validJoints > 0 ? totalScore / validJoints : 1.0f;
        }

        /// <summary>
        /// Quick validation of a single candidate (used for best-candidate selection).
        /// </summary>
        /// <param name="candidate">Candidate to validate</param>
        /// <param name="currentGripperPosition">Current gripper position</param>
        /// <returns>True if candidate is reachable</returns>
        public bool IsReachable(GraspCandidate candidate, Vector3 currentGripperPosition)
        {
            // Optimized squared distance check
            if (
                (candidate.preGraspPosition - currentGripperPosition).sqrMagnitude
                > (_config.maxReachDistance * _config.maxReachDistance)
            )
                return false;

            if (_config.enableIKValidation)
            {
                return ValidateWithIK(candidate).ikValidated;
            }
            return true;
        }
    }

    /// <summary>
    /// Result of IK validation attempt.
    /// </summary>
    internal struct IKResult
    {
        public bool success;
        public int iterations;
        public float finalError;
        public float[] jointAngles;
    }
}
