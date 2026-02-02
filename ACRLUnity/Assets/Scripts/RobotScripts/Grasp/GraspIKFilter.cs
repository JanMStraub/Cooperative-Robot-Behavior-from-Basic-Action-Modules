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

        private readonly Vector3[] _localJointPositions;
        private readonly Quaternion[] _localJointBindRotations;
        private readonly Vector3[] _localJointAxes;
        private readonly float[] _jointLowerLimits;
        private readonly float[] _jointUpperLimits;
        private Vector3 _endEffectorLocalOffset;
        private readonly int _jointCount;

        private readonly float[] _bufferJointAngles;
        private readonly JointInfo[] _bufferJointInfos;

        private readonly bool _enableDebugLogging;
        private readonly bool _enableInitializationValidation;
        private readonly string _logPrefix = "[GRASP_IK_FILTER]";

        /// <summary>
        /// Initialize IK filter with robot configuration.
        /// </summary>
        /// <param name="config">Grasp planning configuration</param>
        /// <param name="joints">Robot joint articulation bodies</param>
        /// <param name="ikReferenceFrame">IK coordinate frame (MUST be stationary relative to joints)</param>
        /// <param name="endEffector">End effector transform</param>
        /// <param name="ikConfig">IK configuration (contains damping factor and other IK parameters)</param>
        /// <param name="enableDebugLogging">Enable debug logging (disable in production)</param>
        /// <param name="enableInitializationValidation">Enable FK validation during initialization (recommended for first-time setup)</param>
        /// <remarks>
        /// IMPORTANT: This implementation assumes the robot base is stationary.
        /// The kinematic structure (joint positions, rotations, axes) is cached relative to ikReferenceFrame
        /// during initialization for performance. If the robot is mounted on a mobile base or linear rail,
        /// call RecalibrateKinematics() whenever the base moves to update the cached geometry.
        /// </remarks>
        public GraspIKFilter(
            GraspConfig config,
            ArticulationBody[] joints,
            Transform ikReferenceFrame,
            Transform endEffector,
            IKConfig ikConfig,
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
            if (ikConfig == null)
                throw new System.ArgumentNullException(nameof(ikConfig));

            _enableDebugLogging = enableDebugLogging;
            _enableInitializationValidation = enableInitializationValidation;

            _jointCount = joints.Length;
            _ikSolver = new IKSolver(_jointCount, ikConfig.dampingFactor);

            _localJointPositions = new Vector3[_jointCount];
            _localJointBindRotations = new Quaternion[_jointCount];
            _localJointAxes = new Vector3[_jointCount];
            _jointLowerLimits = new float[_jointCount];
            _jointUpperLimits = new float[_jointCount];

            _bufferJointAngles = new float[_jointCount];
            _bufferJointInfos = new JointInfo[_jointCount];

            CacheKinematicStructure();
        }

        /// <summary>
        /// Recalibrate kinematic structure after robot base has moved.
        /// Call this if the robot is mounted on a mobile base or linear rail.
        /// This updates the cached joint positions, rotations, and axes relative to ikReferenceFrame.
        /// </summary>
        public void RecalibrateKinematics()
        {
            CacheKinematicStructure();
        }

        /// <summary>
        /// Cache kinematic structure by temporarily resetting robot to zero configuration.
        /// This ensures accurate geometric offsets for FK computation.
        /// </summary>
        private void CacheKinematicStructure()
        {
            float[] originalAngles = new float[_jointCount];
            for (int i = 0; i < _jointCount; i++)
            {
                if (_joints[i].jointPosition.dofCount > 0)
                    originalAngles[i] = _joints[i].jointPosition[0];
            }

            foreach (var joint in _joints)
            {
                if (joint.jointPosition.dofCount > 0)
                    joint.jointPosition = new ArticulationReducedSpace(0f);
            }
            UnityEngine.Physics.SyncTransforms();

            for (int i = 0; i < _jointCount; i++)
            {
                ArticulationBody joint = _joints[i];
                Transform parentTransform = (i == 0) ? _ikReferenceFrame : _joints[i - 1].transform;

                _localJointPositions[i] = parentTransform.InverseTransformPoint(
                    joint.transform.position
                );

                _localJointBindRotations[i] =
                    Quaternion.Inverse(parentTransform.rotation) * joint.transform.rotation;

                Vector3 worldAxis = joint.transform.rotation * joint.anchorRotation * Vector3.right;
                _localJointAxes[i] = joint
                    .transform.InverseTransformDirection(worldAxis)
                    .normalized;

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

            _endEffectorLocalOffset = _joints[_jointCount - 1]
                .transform.InverseTransformPoint(_endEffector.position);

            for (int i = 0; i < _jointCount; i++)
            {
                if (_joints[i].jointPosition.dofCount > 0)
                    _joints[i].jointPosition = new ArticulationReducedSpace(originalAngles[i]);
            }
            UnityEngine.Physics.SyncTransforms();

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
                Debug.Log($"{_logPrefix} End-effector offset validated: {eeOffsetMagnitude:F4}m");
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
                if (
                    (candidate.preGraspPosition - currentGripperPosition).sqrMagnitude > maxReachSqr
                )
                {
                    rejectedByDistance++;
                    continue;
                }

                if (_config.enableIKValidation)
                {
                    ValidateWithIK(candidate);
                    if (candidate.ikValidated)
                    {
                        validCandidates.Add(candidate);
                    }
                    else
                    {
                        rejectedByIK++;
                    }
                }
                else
                {
                    candidate.ikValidated = true;
                    candidate.ikScore = 1.0f;
                    validCandidates.Add(candidate);
                }
            }

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
        /// <param name="candidate">Candidate to validate (modified in-place)</param>
        private void ValidateWithIK(GraspCandidate candidate)
        {
            CaptureCurrentJointAngles(_bufferJointAngles);

            float[] preGraspSeed = (float[])_bufferJointAngles.Clone();

            var preGraspResult = AttemptIK(
                candidate.preGraspPosition,
                candidate.preGraspRotation,
                preGraspSeed
            );

            if (!preGraspResult.success)
            {
                candidate.ikValidated = false;
                candidate.ikScore = 0f;
                return;
            }

            float[] graspSeed = (float[])preGraspResult.jointAngles.Clone();

            var graspResult = AttemptIK(
                candidate.graspPosition,
                candidate.graspRotation,
                graspSeed
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
                return;
            }

            candidate.ikValidated = true;
            candidate.ikScore = ComputeIKQualityScore(preGraspResult, graspResult);
            candidate.preGraspJointPositions = preGraspResult.jointAngles;
            candidate.graspJointPositions = graspResult.jointAngles;

            if (_enableDebugLogging)
            {
                Debug.Log(
                    $"{_logPrefix} Candidate validated: preGraspError={preGraspResult.finalError:F4}m, graspError={graspResult.finalError:F4}m, score={candidate.ikScore:F3}"
                );
            }
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
            Vector3 targetLocalPos = _ikReferenceFrame.InverseTransformPoint(targetPosition);
            Quaternion targetLocalRot =
                Quaternion.Inverse(_ikReferenceFrame.rotation) * targetRotation;
            IKState targetState = new IKState(targetLocalPos, targetLocalRot);

            float[] currentAngles = seedAngles;

            int iterations = 0;
            float error = float.MaxValue;
            float lastError = float.MaxValue;
            int stallCount = 0;

            float threshold = _config.ikValidationThreshold;
            float rotationTolerance = _config.ikRotationTolerance;
            float maxStep = _config.maxJointStepPerIteration;
            int maxIter = _config.maxIKValidationIterations;
            float orientationWeight = 0.05f;

            while (iterations < maxIter)
            {
                IKState currentState = ComputeForwardKinematicsInIKFrame(
                    currentAngles,
                    _bufferJointInfos
                );

                error = Vector3.Distance(currentState.Position, targetState.Position);

                if (error < threshold)
                {
                    if (
                        Quaternion.Angle(currentState.Rotation, targetState.Rotation)
                        < rotationTolerance
                    )
                        break;
                }

                if (Mathf.Abs(lastError - error) < 1e-5f)
                {
                    if (++stallCount > 5 && error > 0.05f)
                        break;
                }
                else
                {
                    stallCount = 0;
                }
                lastError = error;

                var deltas = _ikSolver.ComputeJointDeltas(
                    currentState,
                    targetState,
                    _bufferJointInfos,
                    threshold,
                    orientationWeight
                );

                if (deltas == null)
                    break;

                for (int i = 0; i < _jointCount; i++)
                {
                    float delta = (float)deltas[i];
                    if (delta == 0)
                        continue;

                    delta = Mathf.Clamp(delta, -maxStep, maxStep);

                    float newAngle = currentAngles[i] + delta;
                    bool hitLower = newAngle < _jointLowerLimits[i];
                    bool hitUpper = newAngle > _jointUpperLimits[i];

                    if (hitLower || hitUpper)
                    {
                        currentAngles[i] = Mathf.Clamp(
                            newAngle,
                            _jointLowerLimits[i],
                            _jointUpperLimits[i]
                        );
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
                jointAngles = currentAngles,
            };
        }

        /// <summary>
        /// Optimized Forward Kinematics using Vector3/Quaternion math.
        /// Calculates everything directly in IK Frame space.
        /// </summary>
        /// <param name="jointAngles">Joint angles in radians</param>
        /// <param name="jointInfosOut">Pre-allocated buffer for joint positions and axes in IK frame</param>
        /// <returns>End-effector state in IK frame</returns>
        private IKState ComputeForwardKinematicsInIKFrame(
            float[] jointAngles,
            JointInfo[] jointInfosOut
        )
        {
            Vector3 currentPos = Vector3.zero;
            Quaternion currentRot = Quaternion.identity;

            for (int i = 0; i < _jointCount; i++)
            {
                currentPos += currentRot * _localJointPositions[i];

                Quaternion jointRotation = _localJointBindRotations[i];
                if (_joints[i].jointPosition.dofCount > 0)
                {
                    jointRotation *= Quaternion.AngleAxis(
                        jointAngles[i] * Mathf.Rad2Deg,
                        _localJointAxes[i]
                    );
                }

                currentRot *= jointRotation;

                jointInfosOut[i].WorldPosition = currentPos;
                jointInfosOut[i].WorldAxis = currentRot * _localJointAxes[i];
            }

            currentPos += currentRot * _endEffectorLocalOffset;

            return new IKState(currentPos, currentRot);
        }

        /// <summary>
        /// Compute IK quality score from validation results.
        /// </summary>
        /// <param name="preGraspResult">Pre-grasp IK result</param>
        /// <param name="graspResult">Grasp IK result</param>
        /// <returns>Quality score (0-1, higher is better)</returns>
        private float ComputeIKQualityScore(IKResult preGraspResult, IKResult graspResult)
        {
            float avgIterations = (preGraspResult.iterations + graspResult.iterations) * 0.5f;

            float iterationScore = 1f / (1f + avgIterations / 100f);

            float avgError = (preGraspResult.finalError + graspResult.finalError) * 0.5f;
            float errorScore = Mathf.Clamp01(1.0f - (avgError * 20f));

            float manipulability = ComputeManipulability(graspResult.jointAngles);

            float limitScore = ComputeJointLimitScore(graspResult.jointAngles);

            const float iterationWeight = 0.10f;
            const float errorWeight = 0.50f;
            const float manipWeight = 0.35f;
            const float limitWeight = 0.05f;

            float total =
                (iterationScore * iterationWeight)
                + (errorScore * errorWeight)
                + (manipulability * manipWeight)
                + (limitScore * limitWeight);

            if (UnityEngine.Random.value < 0.02f)
            {
                UnityEngine.Debug.Log(
                    $"{_logPrefix} IK Score Breakdown: "
                        + $"iteration={iterationScore:F3} ({avgIterations:F1} iters), "
                        + $"error={errorScore:F3} ({avgError:F4}m), "
                        + $"manip={manipulability:F3}, "
                        + $"limit={limitScore:F3}, "
                        + $"total={total:F3}"
                );
            }

            return total;
        }

        /// <summary>
        /// Compute manipulability index using optimized inline calculations.
        /// Uses Frobenius norm as proxy for full Yoshikawa manipulability.
        /// </summary>
        /// <param name="jointAngles">Joint configuration</param>
        /// <returns>Manipulability score (0-1, higher is better)</returns>
        private float ComputeManipulability(float[] jointAngles)
        {
            IKState state = ComputeForwardKinematicsInIKFrame(jointAngles, _bufferJointInfos);
            Vector3 eePos = state.Position;

            float sumSquared = 0f;

            for (int i = 0; i < _jointCount; i++)
            {
                float rx = eePos.x - _bufferJointInfos[i].WorldPosition.x;
                float ry = eePos.y - _bufferJointInfos[i].WorldPosition.y;
                float rz = eePos.z - _bufferJointInfos[i].WorldPosition.z;

                Vector3 axis = _bufferJointInfos[i].WorldAxis;

                float cx = axis.y * rz - axis.z * ry;
                float cy = axis.z * rx - axis.x * rz;
                float cz = axis.x * ry - axis.y * rx;

                sumSquared += (cx * cx) + (cy * cy) + (cz * cz);
            }

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

                if (range > 100f || range <= 0f)
                    continue;

                float angle = jointAngles[i];

                float distFromLowerLimit = angle - min;
                float distFromUpperLimit = max - angle;
                float minDistFromEdge = Mathf.Min(distFromLowerLimit, distFromUpperLimit);
                float edgeFraction = minDistFromEdge / range;

                float k = 40f;
                float midpoint = 0.05f;
                float score = 1f / (1f + Mathf.Exp(-k * (edgeFraction - midpoint)));

                totalScore += score;
                validJoints++;
            }

            return validJoints > 0 ? totalScore / validJoints : 1.0f;
        }

        /// <summary>
        /// Quick validation of a single candidate (used for best-candidate selection).
        /// NOTE: Modifies candidate in-place if IK validation is performed.
        /// </summary>
        /// <param name="candidate">Candidate to validate (modified in-place)</param>
        /// <param name="currentGripperPosition">Current gripper position</param>
        /// <returns>True if candidate is reachable</returns>
        public bool IsReachable(GraspCandidate candidate, Vector3 currentGripperPosition)
        {
            if (
                (candidate.preGraspPosition - currentGripperPosition).sqrMagnitude
                > (_config.maxReachDistance * _config.maxReachDistance)
            )
                return false;

            if (_config.enableIKValidation)
            {
                ValidateWithIK(candidate);
                return candidate.ikValidated;
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
