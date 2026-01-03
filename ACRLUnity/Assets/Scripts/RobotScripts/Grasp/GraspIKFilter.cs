using System.Collections.Generic;
using Configuration;
using UnityEngine;

namespace Robotics.Grasp
{
    /// <summary>
    /// Filters grasp candidates based on IK reachability and quality.
    /// Two-stage approach: quick distance rejection, then full IK validation.
    /// </summary>
    public class GraspIKFilter
    {
        private readonly GraspConfig _config;
        private readonly IKSolver _ikSolver;
        private readonly ArticulationBody[] _joints;
        private readonly Transform _ikReferenceFrame;
        private readonly Transform _endEffector;

        // Cached kinematic structure (link offsets and axes)
        private Vector3[] _localJointPositions;
        private Vector3[] _localJointAxes;
        private Vector3 _endEffectorLocalOffset;
        private float[] _jointLowerLimits;
        private float[] _jointUpperLimits;
        private Quaternion[] _localJointInitialRotations;
        private Transform _robotBase; // Robot base link for FK computations

        /// <summary>
        /// Initialize IK filter with robot configuration.
        /// </summary>
        /// <param name="config">Grasp planning configuration</param>
        /// <param name="joints">Robot joint articulation bodies</param>
        /// <param name="ikReferenceFrame">IK coordinate frame</param>
        /// <param name="endEffector">End effector transform</param>
        /// <param name="dampingFactor">IK solver damping factor</param>
        public GraspIKFilter(
            GraspConfig config,
            ArticulationBody[] joints,
            Transform ikReferenceFrame,
            Transform endEffector,
            float dampingFactor = 0.1f
        )
        {
            // Validate parameters
            if (config == null)
                throw new System.ArgumentNullException(nameof(config));
            if (joints == null)
                throw new System.ArgumentNullException(nameof(joints));
            if (ikReferenceFrame == null)
                throw new System.ArgumentNullException(nameof(ikReferenceFrame));
            if (endEffector == null)
                throw new System.ArgumentNullException(nameof(endEffector));

            _config = config;
            _joints = joints;
            _ikReferenceFrame = ikReferenceFrame;
            _endEffector = endEffector;
            _ikSolver = new IKSolver(joints.Length, dampingFactor);

            // Pre-allocate caches
            _localJointPositions = new Vector3[joints.Length];
            _localJointAxes = new Vector3[joints.Length];
            _jointLowerLimits = new float[joints.Length];
            _jointUpperLimits = new float[joints.Length];

            // Cache kinematic structure
            CacheKinematicStructure();
        }

        /// <summary>
        /// Cache the static kinematic structure (link offsets, joint axes, and limits).
        /// Temporarily sets robot to zero configuration to capture clean geometric offsets.
        /// This prevents caching rotated positions that would cause FK errors.
        /// </summary>
        private void CacheKinematicStructure()
        {
            // Store robot base (first joint's parent) for transform computations
            _robotBase = _joints[0].transform.parent;

            // Backup current joint angles
            float[] originalAngles = new float[_joints.Length];
            for (int i = 0; i < _joints.Length; i++)
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
            Physics.SyncTransforms();

            _localJointInitialRotations = new Quaternion[_joints.Length];

            // Cache geometric parameters at zero configuration
            for (int i = 0; i < _joints.Length; i++)
            {
                ArticulationBody joint = _joints[i];
                Transform parent = (i == 0) ? _robotBase : _joints[i - 1].transform;

                // 1. Cache joint position relative to parent
                _localJointPositions[i] = parent.InverseTransformPoint(joint.transform.position);

                // Cache initial rotation in parent's frame
                _localJointInitialRotations[i] = Quaternion.Inverse(parent.rotation) * joint.transform.rotation;

                // 2. Cache joint axis relative to parent (including anchor rotation)
                Vector3 worldAxis = joint.transform.rotation * joint.anchorRotation * Vector3.right;
                _localJointAxes[i] = joint.transform.InverseTransformDirection(worldAxis).normalized;

                // 3. Cache joint limits
                var drive = joint.xDrive;
                if (drive.lowerLimit < drive.upperLimit)
                {
                    _jointLowerLimits[i] = drive.lowerLimit * Mathf.Deg2Rad;
                    _jointUpperLimits[i] = drive.upperLimit * Mathf.Deg2Rad;
                }
                else
                {
                    _jointLowerLimits[i] = -1000f;
                    _jointUpperLimits[i] = 1000f;
                }
            }

            // Cache end-effector offset from last joint
            _endEffectorLocalOffset = _joints[_joints.Length - 1]
                .transform.InverseTransformPoint(_endEffector.position);

            // Restore original joint angles
            for (int i = 0; i < _joints.Length; i++)
            {
                if (_joints[i].jointPosition.dofCount > 0)
                    _joints[i].jointPosition = new ArticulationReducedSpace(originalAngles[i]);
            }
            Physics.SyncTransforms();

            // Validate end-effector assignment
            float eeOffsetMagnitude = _endEffectorLocalOffset.magnitude;
            if (eeOffsetMagnitude < 0.001f)
            {
                UnityEngine.Debug.LogError(
                    $"[GRASP_IK_FILTER] End-effector offset is near-zero ({eeOffsetMagnitude:F6}m)! "
                        + $"This usually means endEffectorBase is assigned to Joint {_joints.Length - 1} instead of the gripper link. "
                        + "IK validation will be INACCURATE."
                );
            }
            else if (eeOffsetMagnitude < 0.02f)
            {
                UnityEngine.Debug.LogWarning(
                    $"[GRASP_IK_FILTER] End-effector offset is very small ({eeOffsetMagnitude:F4}m). "
                        + "Verify that endEffectorBase points to the gripper center."
                );
            }

            // Validate FK by comparing computed position with actual end-effector position
            ValidateForwardKinematics(originalAngles);
        }

        /// <summary>
        /// Validate FK computation by comparing with actual Unity transforms.
        /// </summary>
        private void ValidateForwardKinematics(float[] testAngles)
        {
            // First, verify what angles the joints actually have after our zero-setting
            float[] actualAngles = new float[_joints.Length];
            for (int i = 0; i < _joints.Length; i++)
            {
                if (_joints[i].jointPosition.dofCount > 0)
                    actualAngles[i] = _joints[i].jointPosition[0];
            }

            // Compute FK using our cached method
            JointInfo[] jointInfos;
            IKState computedState = ComputeForwardKinematicsInIKFrame(testAngles, out jointInfos);

            // Get actual end-effector position from Unity
            Vector3 actualWorldPos = _endEffector.position;
            Vector3 actualIKPos = _ikReferenceFrame.InverseTransformPoint(actualWorldPos);

            // Compare
            float error = Vector3.Distance(computedState.Position, actualIKPos);

            // Log individual joint offset errors during caching
            for (int i = 0; i < _joints.Length; i++)
            {
                Transform parent = (i == 0) ? _robotBase : _joints[i - 1].transform;
                Vector3 actualOffset = parent.InverseTransformPoint(_joints[i].transform.position);
            }

            // Log joint positions for debugging
            for (int i = 0; i < _joints.Length; i++)
            {
                Vector3 actualJointWorld = _joints[i].transform.position;
                Vector3 actualJointIK = _ikReferenceFrame.InverseTransformPoint(actualJointWorld);
                float jointError = Vector3.Distance(jointInfos[i].WorldPosition, actualJointIK);

                if (jointError > 0.01f)
                {
                    UnityEngine.Debug.LogWarning(
                        $"[GRASP_IK_FILTER] Joint {i} FK error: {jointError:F4}m - "
                            + $"Computed: {jointInfos[i].WorldPosition}, Actual: {actualJointIK}"
                    );
                }
            }
        }

        /// <summary>
        /// Filter candidates by IK reachability.
        /// Returns only candidates that pass both distance check and IK validation.
        /// </summary>
        /// <param name="candidates">Candidates to filter</param>
        /// <param name="currentGripperPosition">Current gripper position (world space)</param>
        /// <returns>List of validated candidates with IK scores</returns>
        public List<GraspCandidate> FilterCandidates(
            List<GraspCandidate> candidates,
            Vector3 currentGripperPosition
        )
        {
            var validCandidates = new List<GraspCandidate>();
            int rejectedByDistance = 0;
            int rejectedByIK = 0;

            foreach (var candidate in candidates)
            {
                // Stage 1: Quick distance rejection
                if (!IsWithinReach(candidate, currentGripperPosition))
                {
                    rejectedByDistance++;
                    continue;
                }

                // Stage 2: Full IK validation (if enabled)
                if (_config.enableIKValidation)
                {
                    var validatedCandidate = ValidateWithIK(candidate);
                    if (validatedCandidate.ikValidated)
                    {
                        validCandidates.Add(validatedCandidate);
                    }
                    else
                    {
                        rejectedByIK++;
                    }
                }
                else
                {
                    // Skip IK validation - accept all within-reach candidates
                    var accepted = candidate;
                    accepted.ikValidated = true;
                    accepted.ikScore = 1.0f;
                    validCandidates.Add(accepted);
                }
            }

            return validCandidates;
        }

        /// <summary>
        /// Quick distance-based reachability check.
        /// Rejects candidates outside max reach distance.
        /// </summary>
        /// <param name="candidate">Candidate to check</param>
        /// <param name="currentPosition">Current gripper position</param>
        /// <returns>True if within reach</returns>
        private bool IsWithinReach(GraspCandidate candidate, Vector3 currentPosition)
        {
            float distance = Vector3.Distance(candidate.preGraspPosition, currentPosition);
            return distance <= _config.maxReachDistance;
        }

        /// <summary>
        /// Validate candidate with full IK solver.
        /// Attempts to compute IK solution and scores quality.
        /// Uses seeded IK (pre-grasp solution seeds grasp solution) for better performance.
        /// </summary>
        /// <param name="candidate">Candidate to validate</param>
        /// <returns>Candidate with IK validation flags and score</returns>
        private GraspCandidate ValidateWithIK(GraspCandidate candidate)
        {
            // First attempt: Pre-Grasp
            var preGraspResult = AttemptIK(candidate.preGraspPosition, candidate.preGraspRotation);
            if (!preGraspResult.success)
            {
                candidate.ikValidated = false;
                candidate.ikScore = 0f;
                return candidate;
            }

            // Second attempt: Grasp (Seed with pre-grasp result for speed)
            var graspResult = AttemptIK(
                candidate.graspPosition,
                candidate.graspRotation,
                preGraspResult.jointAngles
            );
            if (!graspResult.success)
            {
                candidate.ikValidated = false;
                candidate.ikScore = 0f;
                return candidate;
            }

            // Both positions reachable - compute quality score
            candidate.ikValidated = true;
            candidate.ikScore = ComputeIKQualityScore(preGraspResult, graspResult);

            return candidate;
        }

        /// <summary>
        /// Attempt to solve IK for a target pose using analytical forward kinematics.
        /// </summary>
        /// <param name="targetPosition">Target position (world space)</param>
        /// <param name="targetRotation">Target rotation (world space)</param>
        /// <param name="initialAngles">Optional initial joint angles (for seeded IK)</param>
        /// <returns>IK result with success flag and quality metrics</returns>
        private IKResult AttemptIK(
            Vector3 targetPosition,
            Quaternion targetRotation,
            float[] initialAngles = null
        )
        {
            // Convert target to IK reference frame
            Vector3 targetLocalPos = _ikReferenceFrame.InverseTransformPoint(targetPosition);
            Quaternion targetLocalRot =
                Quaternion.Inverse(_ikReferenceFrame.rotation) * targetRotation;
            IKState targetState = new IKState(targetLocalPos, targetLocalRot);

            // Initialize joint angles (either from seed or current state)
            float[] jointAngles = new float[_joints.Length];
            if (initialAngles != null)
            {
                System.Array.Copy(initialAngles, jointAngles, initialAngles.Length);
            }
            else
            {
                for (int i = 0; i < _joints.Length; i++)
                {
                    jointAngles[i] =
                        _joints[i].jointPosition.dofCount > 0 ? _joints[i].jointPosition[0] : 0f;
                }
            }

            // IK iteration loop
            int iterations = 0;
            float error = float.MaxValue;
            float lastError = float.MaxValue;
            int stallCount = 0;

            // Use low orientation weight - we care primarily about position for grasp validation
            float orientationWeight = 0.05f;

            while (iterations < _config.maxIKValidationIterations)
            {
                // Compute FK analytically from joint angles
                JointInfo[] jointInfos;
                IKState currentState = ComputeForwardKinematicsInIKFrame(
                    jointAngles,
                    out jointInfos
                );

                // Compute position error
                error = Vector3.Distance(currentState.Position, targetState.Position);

                // Success condition: position within threshold
                if (error < _config.ikValidationThreshold)
                {
                    // Check rotation tolerance (configurable)
                    if (
                        Quaternion.Angle(currentState.Rotation, targetState.Rotation)
                        < _config.ikRotationTolerance
                    )
                        break;
                }

                // Stall detection - if error doesn't change, we're stuck at joint limits
                if (Mathf.Abs(lastError - error) < 0.0001f)
                {
                    stallCount++;
                    if (stallCount > 10)
                    {
                        // If stuck far from target, abort early
                        if (error > 0.05f)
                            break;
                    }
                }
                else
                {
                    stallCount = 0;
                }
                lastError = error;

                // Compute joint deltas using IK solver
                var deltas = _ikSolver.ComputeJointDeltas(
                    currentState,
                    targetState,
                    jointInfos,
                    _config.ikValidationThreshold,
                    orientationWeight
                );

                if (deltas == null)
                    break;

                // Apply deltas with configurable step limit
                for (int i = 0; i < _joints.Length; i++)
                {
                    float delta = (float)deltas[i];

                    // Safety clamp using config parameter
                    delta = Mathf.Clamp(
                        delta,
                        -_config.maxJointStepPerIteration,
                        _config.maxJointStepPerIteration
                    );

                    jointAngles[i] += delta;

                    // Hard clamp to cached joint limits
                    jointAngles[i] = Mathf.Clamp(
                        jointAngles[i],
                        _jointLowerLimits[i],
                        _jointUpperLimits[i]
                    );
                }

                iterations++;
            }

            // Log result for debugging
            bool success = error < _config.ikValidationThreshold;

            return new IKResult
            {
                success = success,
                iterations = iterations,
                finalError = error,
                jointAngles = jointAngles,
            };
        }

        /// <summary>
        /// Compute forward kinematics analytically from joint angles in IK reference frame.
        /// Uses cached local joint positions/axes for efficient computation.
        /// CRITICAL: Must use same axes for FK and Jacobian for IK convergence.
        /// </summary>
        /// <param name="jointAngles">Joint angles in radians</param>
        /// <param name="jointInfos">Output: joint positions and axes in IK frame (for Jacobian)</param>
        /// <returns>End-effector state in IK frame</returns>
        private IKState ComputeForwardKinematicsInIKFrame(
            float[] jointAngles,
            out JointInfo[] jointInfos
        )
        {
            jointInfos = new JointInfo[_joints.Length];

            // Build FK by composing transforms from parent to child
            // Start with base transform (identity in base-local space)
            Matrix4x4 accumulatedTransform = Matrix4x4.identity;

            // Chain transformations through each joint
            for (int i = 0; i < _joints.Length; i++)
            {

                Quaternion jointActuation = Quaternion.AngleAxis(jointAngles[i] * Mathf.Rad2Deg, _localJointAxes[i]);
                Quaternion totalLocalRotation = _localJointInitialRotations[i] * jointActuation;

                Matrix4x4 localTransform = Matrix4x4.TRS(
                    _localJointPositions[i], 
                    totalLocalRotation, 
                    Vector3.one
                );

                // 3. Compose: T_i = T_{i-1} * Translate * Rotate
                accumulatedTransform = accumulatedTransform * localTransform;

                // 4. Extract joint position and axis for Jacobian
                Vector3 jointPosInBase = accumulatedTransform.GetColumn(3); // Position column
                Quaternion jointRotInBase = accumulatedTransform.rotation;
                Vector3 axisInBase = jointRotInBase * _localJointAxes[i]; // Default axis after rotation

                // Transform to world then to IK frame
                Vector3 jPosWorld = _robotBase.TransformPoint(jointPosInBase);
                Vector3 jAxisWorld = _robotBase.TransformDirection(axisInBase);

                Vector3 jPosIK = _ikReferenceFrame.InverseTransformPoint(jPosWorld);
                Vector3 jAxisIK = _ikReferenceFrame
                    .InverseTransformDirection(jAxisWorld)
                    .normalized;

                jointInfos[i] = new JointInfo(jPosIK, jAxisIK);
            }

            // Add end-effector offset (in final joint's local frame)
            Matrix4x4 eeTranslation = Matrix4x4.Translate(_endEffectorLocalOffset);
            accumulatedTransform = accumulatedTransform * eeTranslation;

            // Extract final pose in base-local space
            Vector3 eePosInBase = accumulatedTransform.GetColumn(3);
            Quaternion eeRotInBase = accumulatedTransform.rotation;

            // Transform to world then to IK frame
            Vector3 worldPos = _robotBase.TransformPoint(eePosInBase);
            Quaternion worldRot = _robotBase.rotation * eeRotInBase;

            return new IKState(
                _ikReferenceFrame.InverseTransformPoint(worldPos),
                Quaternion.Inverse(_ikReferenceFrame.rotation) * worldRot
            );
        }

        /// <summary>
        /// Compute IK quality score from validation results.
        /// Favors low error over iteration count, and adds manipulability and joint limit proximity.
        /// </summary>
        /// <param name="preGraspResult">Pre-grasp IK result</param>
        /// <param name="graspResult">Grasp IK result</param>
        /// <returns>Quality score (0-1, higher is better)</returns>
        private float ComputeIKQualityScore(IKResult preGraspResult, IKResult graspResult)
        {
            // Factor 1: Convergence speed (fewer iterations is better)
            float maxIterations = _config.maxIKValidationIterations;
            float avgIterations = (preGraspResult.iterations + graspResult.iterations) / 2f;
            float iterationScore = 1f - (avgIterations / maxIterations);

            // Factor 2: Final error (smaller is better) - weighted more heavily
            float avgError = (preGraspResult.finalError + graspResult.finalError) / 2f;
            float errorScore = 1.0f - (avgError * 10f); // Penalize error heavily

            // Factor 3: Manipulability (ability to move from this configuration)
            float manipulability = ComputeManipulability(graspResult.jointAngles);

            // Factor 4: Joint limit proximity (penalize configurations near joint limits)
            float jointLimitScore = ComputeJointLimitScore(graspResult.jointAngles);

            // Weighted combination (favor low error and good manipulability)
            return Mathf.Max(0f,
                iterationScore * 0.2f +
                errorScore * 0.5f +
                manipulability * 0.2f +
                jointLimitScore * 0.1f);
        }

        /// <summary>
        /// Compute manipulability index from joint angles.
        /// Measures how easily the robot can move in different directions from this configuration.
        /// Uses simplified Yoshikawa manipulability (determinant of Jacobian).
        /// </summary>
        /// <param name="jointAngles">Joint configuration</param>
        /// <returns>Manipulability score (0-1, higher is better)</returns>
        private float ComputeManipulability(float[] jointAngles)
        {
            // Compute forward kinematics to get joint info
            JointInfo[] jointInfos;
            IKState currentState = ComputeForwardKinematicsInIKFrame(jointAngles, out jointInfos);

            // Build Jacobian matrix (simplified - position only)
            int numJoints = _joints.Length;
            float[,] jacobian = new float[3, numJoints]; // 3 DOF (position only)

            Vector3 endEffectorPos = currentState.Position;

            for (int i = 0; i < numJoints; i++)
            {
                Vector3 jointPos = jointInfos[i].WorldPosition;
                Vector3 jointAxis = jointInfos[i].WorldAxis;

                // Compute contribution to end-effector velocity
                Vector3 r = endEffectorPos - jointPos;
                Vector3 contribution = Vector3.Cross(jointAxis, r);

                jacobian[0, i] = contribution.x;
                jacobian[1, i] = contribution.y;
                jacobian[2, i] = contribution.z;
            }

            // Compute manipulability measure (simplified: sum of squared singular values approximation)
            // For a proper implementation, we'd compute SVD, but this is computationally expensive
            // Instead, use Frobenius norm as proxy
            float manipulabilityMeasure = 0f;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < numJoints; j++)
                {
                    manipulabilityMeasure += jacobian[i, j] * jacobian[i, j];
                }
            }

            // Normalize (typical values range from 0 to ~0.5 for this robot)
            manipulabilityMeasure = Mathf.Sqrt(manipulabilityMeasure) / (numJoints * 0.3f);
            return Mathf.Clamp01(manipulabilityMeasure);
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
            int validJointCount = 0;

            for (int i = 0; i < jointAngles.Length; i++)
            {
                float lower = _jointLowerLimits[i];
                float upper = _jointUpperLimits[i];

                // Skip joints with effectively unlimited range
                if (upper - lower > 100f)
                    continue;

                // Compute normalized position within range (0 = at lower, 1 = at upper)
                float range = upper - lower;
                float normalizedPos = (jointAngles[i] - lower) / range;

                // Distance to nearest limit (0.5 = centered, 0 or 1 = at limit)
                float distToLimit = Mathf.Abs(normalizedPos - 0.5f);

                // Score: 1.0 at center, 0.0 at limits
                float jointScore = 1.0f - (distToLimit * 2f);

                // Penalty increases near limits (within 10% of range)
                if (distToLimit > 0.4f) // Within 10% of limit
                {
                    float limitProximity = (distToLimit - 0.4f) / 0.1f;
                    jointScore *= (1.0f - limitProximity * 0.5f);
                }

                totalScore += jointScore;
                validJointCount++;
            }

            return validJointCount > 0 ? (totalScore / validJointCount) : 1.0f;
        }

        /// <summary>
        /// Quick validation of a single candidate (used for best-candidate selection).
        /// </summary>
        /// <param name="candidate">Candidate to validate</param>
        /// <param name="currentGripperPosition">Current gripper position</param>
        /// <returns>True if candidate is reachable</returns>
        public bool IsReachable(GraspCandidate candidate, Vector3 currentGripperPosition)
        {
            if (!IsWithinReach(candidate, currentGripperPosition))
                return false;

            if (_config.enableIKValidation)
            {
                var validated = ValidateWithIK(candidate);
                return validated.ikValidated;
            }

            return true; // If validation disabled, assume reachable
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
