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

            foreach (var candidate in candidates)
            {
                // Stage 1: Quick distance rejection
                if (!IsWithinReach(candidate, currentGripperPosition))
                {
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
                    candidate.ikValidated = false;
                    candidate.ikScore = 0f;
                    return candidate;
                }

                // Try to solve IK for final grasp position
                var graspResult = AttemptIK(candidate.graspPosition, candidate.graspRotation);

                // If grasp position fails, candidate is invalid
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
            // Convert to IK local frame
            Vector3 targetLocal = _ikReferenceFrame.InverseTransformPoint(targetPosition);
            Quaternion targetRotLocal = Quaternion.Inverse(_ikReferenceFrame.rotation) * targetRotation;

            IKState targetState = new IKState(targetLocal, targetRotLocal);

            // Build joint info array (this contains static joint structure)
            JointInfo[] jointInfos = BuildJointInfoArray();

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
                // Compute current end-effector state using forward kinematics
                IKState currentState = ComputeForwardKinematics(tempJointAngles, jointInfos);

                // Check convergence
                finalError = Vector3.Distance(currentState.Position, targetState.Position);
                if (finalError < _config.ikValidationThreshold)
                {
                    // Converged successfully
                    break;
                }

                // Compute joint deltas using IK solver
                Vector<double> deltas = _ikSolver.ComputeJointDeltas(
                    currentState,
                    targetState,
                    jointInfos,
                    _config.ikValidationThreshold
                );

                // If deltas is null, solver thinks we're converged (but we checked above)
                if (deltas == null)
                {
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
        /// Compute forward kinematics using Unity's ArticulationBody physics system.
        /// Temporarily applies joint angles and reads actual end-effector transform.
        /// Does NOT restore state - caller is responsible for state management.
        /// </summary>
        /// <param name="jointAngles">Current joint angles (radians)</param>
        /// <param name="jointInfos">Joint information array (unused - kept for API compatibility)</param>
        /// <returns>End-effector state in IK frame</returns>
        private IKState ComputeForwardKinematics(float[] jointAngles, JointInfo[] jointInfos)
        {
            // Save current state before applying test angles
            float[] tempCache = new float[_joints.Length];
            for (int i = 0; i < _joints.Length; i++)
            {
                if (_joints[i] != null && _joints[i].jointPosition.dofCount > 0)
                    tempCache[i] = _joints[i].jointPosition[0];
                else
                    tempCache[i] = 0f;
            }

            // Apply test joint angles to ArticulationBody components
            for (int i = 0; i < Mathf.Min(jointAngles.Length, _joints.Length); i++)
            {
                if (_joints[i] == null || _joints[i].jointPosition.dofCount == 0)
                    continue;

                _joints[i].jointPosition = new ArticulationReducedSpace(jointAngles[i]);
            }

            // Force Unity to update transform hierarchy immediately
            Physics.SyncTransforms();

            // Read actual end-effector position from Unity's transform hierarchy
            Vector3 worldPosition = _endEffector.position;
            Quaternion worldRotation = _endEffector.rotation;

            // Restore previous state immediately to prevent accumulation
            for (int i = 0; i < tempCache.Length; i++)
            {
                if (_joints[i] == null || _joints[i].jointPosition.dofCount == 0)
                    continue;

                _joints[i].jointPosition = new ArticulationReducedSpace(tempCache[i]);
            }

            Physics.SyncTransforms();

            // Transform to IK local frame (matching RobotController pattern)
            Vector3 localPosition = _ikReferenceFrame.InverseTransformPoint(worldPosition);
            Quaternion localRotation = Quaternion.Inverse(_ikReferenceFrame.rotation) * worldRotation;

            return new IKState(localPosition, localRotation);
        }

        /// <summary>
        /// Build joint info array for IK solver.
        /// </summary>
        /// <returns>Array of joint information in IK frame</returns>
        private JointInfo[] BuildJointInfoArray()
        {
            JointInfo[] jointInfos = new JointInfo[_joints.Length];

            for (int i = 0; i < _joints.Length; i++)
            {
                ArticulationBody joint = _joints[i];

                // Get joint position in IK frame
                Vector3 localPos = _ikReferenceFrame.InverseTransformPoint(joint.transform.position);

                // Get joint axis in IK frame
                Vector3 worldAxis = joint.transform.TransformDirection(Vector3.right); // Assume X-axis rotation
                Vector3 localAxis = _ikReferenceFrame.InverseTransformDirection(worldAxis);

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
