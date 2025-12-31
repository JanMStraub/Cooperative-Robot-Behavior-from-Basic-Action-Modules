using UnityEngine;
using Configuration;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

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

        // Cache for original joint states during validation
        private float[] _cachedJointStates;
        private float[] _cachedDriveTargets;

        // Cached kinematic structure (link offsets and axes)
        private Vector3[] _linkOffsets;
        private Vector3[] _jointAxes;
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

            // Pre-allocate cache for joint states
            _cachedJointStates = new float[joints.Length];
            _cachedDriveTargets = new float[joints.Length];

            // Log configuration for debugging
            UnityEngine.Debug.Log($"[GRASP_IK_FILTER] Initialized with config: maxIter={_config.maxIKValidationIterations}, threshold={_config.ikValidationThreshold:F4}m, damping={dampingFactor}");

            // Cache kinematic structure
            CacheKinematicStructure();
        }

        /// <summary>
        /// Cache the static kinematic structure (link offsets and joint axes).
        /// Called once during initialization to avoid repeated transform lookups.
        /// CRITICAL: Robot must be in zero configuration (all joints at 0 radians) at scene start!
        /// </summary>
        private void CacheKinematicStructure()
        {
            _linkOffsets = new Vector3[_joints.Length + 1]; // +1 for end-effector
            _jointAxes = new Vector3[_joints.Length];

            // Verify robot is at zero configuration
            bool atZeroConfig = true;
            for (int i = 0; i < _joints.Length; i++)
            {
                if (_joints[i].jointPosition.dofCount > 0)
                {
                    float jointAngle = _joints[i].jointPosition[0];
                    if (Mathf.Abs(jointAngle) > 0.01f) // Tolerance of ~0.5 degrees
                    {
                        atZeroConfig = false;
                        UnityEngine.Debug.LogWarning($"[GRASP_IK_FILTER] Joint {i} not at zero: {jointAngle:F4} rad");
                    }
                }
            }

            if (!atZeroConfig)
            {
                UnityEngine.Debug.LogError("[GRASP_IK_FILTER] Robot not in zero configuration! FK will be inaccurate. Please ensure all joints start at 0 degrees.");
            }

            // CRITICAL FIX: Use robot base (first joint's parent) as FK reference, not the scene-level IK reference
            // The _ikReferenceFrame is for IK computations in world space, but FK should use the robot's own base
            Transform robotBase = _joints[0].transform.parent;

            // Cache joint axes - transform each joint's local X-axis into its parent's frame for FK chaining
            // This ensures FK and Jacobian use consistent axes
            for (int i = 0; i < _joints.Length; i++)
            {
                // Get joint's local X-axis in world space
                Vector3 worldAxis = _joints[i].transform.TransformDirection(Vector3.right);

                // Transform into parent's local frame (robot base for joint 0, previous joint for others)
                if (i == 0)
                {
                    _jointAxes[i] = robotBase.InverseTransformDirection(worldAxis);
                }
                else
                {
                    _jointAxes[i] = _joints[i - 1].transform.InverseTransformDirection(worldAxis);
                }
            }

            // Cache link offsets from transform hierarchy at zero configuration
            // This captures the actual link lengths and offsets
            for (int i = 0; i < _joints.Length; i++)
            {
                if (i == 0)
                {
                    // First joint: offset from robot base to first joint
                    _linkOffsets[i] = robotBase.InverseTransformPoint(_joints[i].transform.position);
                }
                else
                {
                    // Subsequent joints: offset from parent joint to this joint in parent's local frame
                    _linkOffsets[i] = _joints[i - 1].transform.InverseTransformPoint(_joints[i].transform.position);
                }
            }

            // Last offset: from last joint to end-effector (measured at current config)
            // This needs to be measured because end-effector is not an ArticulationBody
            _linkOffsets[_joints.Length] = _joints[_joints.Length - 1].transform.InverseTransformPoint(_endEffector.position);

            // Store robot base for FK computations
            _robotBase = robotBase;
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
        /// </summary>
        /// <param name="candidate">Candidate to validate</param>
        /// <returns>Candidate with IK validation flags and score</returns>
        private GraspCandidate ValidateWithIK(GraspCandidate candidate)
        {
            // Cache original joint states before any IK attempts
            CacheJointStates();

            try
            {
                // Try to solve IK for pre-grasp position
                var preGraspResult = AttemptIK(candidate.preGraspPosition, candidate.preGraspRotation);

                // If pre-grasp fails, candidate is invalid
                if (!preGraspResult.success)
                {
                    UnityEngine.Debug.Log($"[GRASP_IK_FILTER] Pre-grasp failed: error={preGraspResult.finalError:F4}m, threshold={_config.ikValidationThreshold:F4}m, iterations={preGraspResult.iterations}");
                    candidate.ikValidated = false;
                    candidate.ikScore = 0f;
                    return candidate;
                }

                // Try to solve IK for final grasp position
                var graspResult = AttemptIK(candidate.graspPosition, candidate.graspRotation);

                // If grasp position fails, candidate is invalid
                if (!graspResult.success)
                {
                    UnityEngine.Debug.Log($"[GRASP_IK_FILTER] Grasp failed: error={graspResult.finalError:F4}m, threshold={_config.ikValidationThreshold:F4}m, iterations={graspResult.iterations}");
                    candidate.ikValidated = false;
                    candidate.ikScore = 0f;
                    return candidate;
                }

                // Both positions reachable - compute quality score
                candidate.ikValidated = true;
                candidate.ikScore = ComputeIKQualityScore(preGraspResult, graspResult);

                return candidate;
            }
            finally
            {
                // Always restore joint states, even if validation throws exception
                RestoreJointStates();
            }
        }

        /// <summary>
        /// Attempt to solve IK for a target pose using proper forward kinematics.
        /// </summary>
        /// <param name="targetPosition">Target position (world space)</param>
        /// <param name="targetRotation">Target rotation (world space)</param>
        /// <returns>IK result with success flag and quality metrics</returns>
        private IKResult AttemptIK(Vector3 targetPosition, Quaternion targetRotation)
        {
            // Convert target from world space to robot base local frame (for FK comparison)
            Vector3 targetLocal = _robotBase.InverseTransformPoint(targetPosition);
            Quaternion targetRotLocal = Quaternion.Inverse(_robotBase.rotation) * targetRotation;

            IKState targetState = new IKState(targetLocal, targetRotLocal);

            // Initialize with current joint angles
            float[] tempJointAngles = new float[_joints.Length];
            for (int i = 0; i < _joints.Length; i++)
            {
                // Check if joint has degrees of freedom before accessing
                if (_joints[i].jointPosition.dofCount > 0)
                {
                    tempJointAngles[i] = _joints[i].jointPosition[0];
                }
                else
                {
                    tempJointAngles[i] = 0f; // Default to 0 if no DOF (e.g., fixed joint)
                }
            }

            // Iterative IK solving with proper FK
            int iterations = 0;
            float finalError = float.MaxValue;

            while (iterations < _config.maxIKValidationIterations)
            {
                // Compute current end-effector state using analytical forward kinematics
                // This also computes joint positions and axes for the Jacobian
                IKState currentState = ComputeForwardKinematicsFromTransforms(tempJointAngles, out JointInfo[] jointInfos);

                // Check convergence
                finalError = Vector3.Distance(currentState.Position, targetState.Position);

                if (finalError < _config.ikValidationThreshold)
                {
                    // Converged successfully
                    UnityEngine.Debug.Log($"[GRASP_IK_FILTER] IK converged in {iterations} iterations, final error={finalError:F4}m");
                    break;
                }

                // Compute joint deltas using IK solver
                // Use reduced orientation weight (0.3) to prioritize position accuracy for grasp planning
                Vector<double> deltas = _ikSolver.ComputeJointDeltas(
                    currentState,
                    targetState,
                    jointInfos,
                    _config.ikValidationThreshold,
                    orientationWeight: 0.3f
                );

                // If deltas is null, solver thinks we're converged (but we checked above)
                if (deltas == null)
                {
                    UnityEngine.Debug.Log($"[GRASP_IK_FILTER] IK solver returned null deltas at iteration {iterations}, error={finalError:F4}m");
                    break;
                }

                // Apply deltas to joint angles
                for (int i = 0; i < _joints.Length; i++)
                {
                    tempJointAngles[i] += (float)deltas[i];

                    // Clamp to reasonable joint limits to prevent unrealistic solutions
                    tempJointAngles[i] = Mathf.Clamp(tempJointAngles[i], -Mathf.PI, Mathf.PI);
                }

                iterations++;
            }

            if (iterations >= _config.maxIKValidationIterations)
            {
                UnityEngine.Debug.Log($"[GRASP_IK_FILTER] IK reached max iterations ({iterations}), final error={finalError:F4}m");
            }

            // Check if converged within threshold
            bool success = finalError < _config.ikValidationThreshold;

            return new IKResult
            {
                success = success,
                iterations = iterations,
                finalError = finalError,
                jointAngles = tempJointAngles
            };
        }

        /// <summary>
        /// Compute forward kinematics analytically from joint angles.
        /// Uses cached link offsets and joint axes measured at zero configuration.
        /// CRITICAL: Must use same axes as Jacobian for IK convergence.
        /// </summary>
        /// <param name="jointAngles">Joint angles in radians</param>
        /// <param name="jointInfos">Output: joint positions and axes in robot base frame (for Jacobian)</param>
        /// <returns>End-effector state in robot base frame</returns>
        private IKState ComputeForwardKinematicsFromTransforms(float[] jointAngles, out JointInfo[] jointInfos)
        {
            // Start at robot base
            Vector3 position = Vector3.zero;
            Quaternion rotation = Quaternion.identity;

            // Allocate joint info array
            jointInfos = new JointInfo[jointAngles.Length];

            // Chain transformations for each joint
            for (int i = 0; i < jointAngles.Length; i++)
            {
                // Translate to joint position (in current frame)
                position += rotation * _linkOffsets[i];

                // Store joint position in robot base frame
                Vector3 jointPosition = position;

                // Transform joint axis into robot base frame (current accumulated rotation applied to cached axis)
                Vector3 jointAxisInBaseFrame = rotation * _jointAxes[i];

                // Store joint info for Jacobian computation
                jointInfos[i] = new JointInfo(jointPosition, jointAxisInBaseFrame);

                // Rotate around cached joint axis (in parent's local frame)
                Quaternion jointRotation = Quaternion.AngleAxis(jointAngles[i] * Mathf.Rad2Deg, _jointAxes[i]);
                rotation *= jointRotation;
            }

            // Final translation to end-effector (in current frame)
            position += rotation * _linkOffsets[jointAngles.Length];

            return new IKState(position, rotation);
        }

        /// <summary>
        /// Compute forward kinematics without joint info output (for simple FK queries).
        /// </summary>
        /// <param name="jointAngles">Joint angles in radians</param>
        /// <returns>End-effector state in robot base frame</returns>
        private IKState ComputeForwardKinematicsFromTransforms(float[] jointAngles)
        {
            return ComputeForwardKinematicsFromTransforms(jointAngles, out _);
        }

        /// <summary>
        /// Build joint info array for IK solver.
        /// CRITICAL: Must use same coordinate frame as FK (_robotBase, not _ikReferenceFrame)
        /// </summary>
        /// <returns>Array of joint information in robot base frame</returns>
        private JointInfo[] BuildJointInfoArray()
        {
            JointInfo[] jointInfos = new JointInfo[_joints.Length];

            for (int i = 0; i < _joints.Length; i++)
            {
                ArticulationBody joint = _joints[i];

                // Get joint position in robot base frame (same as FK)
                Vector3 localPos = _robotBase.InverseTransformPoint(joint.transform.position);

                // Get joint axis in robot base frame (same as FK)
                Vector3 worldAxis = joint.transform.TransformDirection(Vector3.right); // Assume X-axis rotation
                Vector3 localAxis = _robotBase.InverseTransformDirection(worldAxis);

                jointInfos[i] = new JointInfo(localPos, localAxis);
            }

            return jointInfos;
        }

        /// <summary>
        /// Cache current joint angles and drive targets before validation.
        /// </summary>
        private void CacheJointStates()
        {
            for (int i = 0; i < _joints.Length; i++)
            {
                if (_joints[i] != null && _joints[i].jointPosition.dofCount > 0)
                {
                    _cachedJointStates[i] = _joints[i].jointPosition[0];
                    _cachedDriveTargets[i] = _joints[i].xDrive.target;
                }
                else
                {
                    _cachedJointStates[i] = 0f;
                    _cachedDriveTargets[i] = 0f;
                }
            }
        }

        /// <summary>
        /// Temporarily apply joint angles to ArticulationBody for FK evaluation.
        /// Used by tests via reflection. Sets both position and drive target to prevent physics interference.
        /// </summary>
        /// <param name="jointAngles">Joint angles in radians</param>
        private void ApplyJointAnglesTemporarily(float[] jointAngles)
        {
            for (int i = 0; i < Mathf.Min(jointAngles.Length, _joints.Length); i++)
            {
                if (_joints[i] == null || _joints[i].jointPosition.dofCount == 0)
                    continue;

                _joints[i].jointPosition = new ArticulationReducedSpace(jointAngles[i]);

                // Also set drive target to prevent physics from fighting the position
                var drive = _joints[i].xDrive;
                drive.target = jointAngles[i] * Mathf.Rad2Deg;
                _joints[i].xDrive = drive;
            }

            // Force Unity to update transform hierarchy immediately
            Physics.SyncTransforms();
        }

        /// <summary>
        /// Restore original joint states and drive targets after validation.
        /// Called in finally block to ensure robot state is restored even if validation throws.
        /// </summary>
        private void RestoreJointStates()
        {
            for (int i = 0; i < _cachedJointStates.Length; i++)
            {
                if (_joints[i] == null || _joints[i].jointPosition.dofCount == 0)
                    continue;

                _joints[i].jointPosition = new ArticulationReducedSpace(_cachedJointStates[i]);

                // Restore drive target as well
                var drive = _joints[i].xDrive;
                drive.target = _cachedDriveTargets[i];
                _joints[i].xDrive = drive;
            }

            Physics.SyncTransforms();
        }

        /// <summary>
        /// Compute IK quality score from validation results.
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

            // Factor 2: Final error (smaller is better)
            float maxError = _config.ikValidationThreshold * 2f; // Allow some tolerance
            float avgError = (preGraspResult.finalError + graspResult.finalError) / 2f;
            float errorScore = 1f - Mathf.Clamp01(avgError / maxError);

            // Weighted combination
            return iterationScore * 0.4f + errorScore * 0.6f;
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
