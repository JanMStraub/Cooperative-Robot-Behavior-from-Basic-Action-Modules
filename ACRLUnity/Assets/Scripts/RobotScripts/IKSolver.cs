using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Pure C# class for inverse kinematics calculations using damped least squares.
    /// No Unity MonoBehaviour dependencies - testable in isolation.
    /// </summary>
    public class IKSolver
    {
        private readonly float _dampingFactor;

        // Pre-allocated matrices for GC-free operation
        private Matrix<double> _jacobianMatrix;
        private Matrix<double> _cachedIdentity;
        private Vector<double> _errorVector;
        private Vector<double> _jointDelta;

        // Iteration tracking
        private int _iterationCount;
        public int IterationCount => _iterationCount;

        /// <summary>
        /// Reset iteration counter (call when starting new target)
        /// </summary>
        public void ResetIterationCount()
        {
            _iterationCount = 0;
        }

        /// <summary>
        /// Set iteration count (for transferring state between solvers)
        /// </summary>
        public void SetIterationCount(int count)
        {
            _iterationCount = count;
        }

        /// <summary>
        /// Creates a new IK solver for a robot with specified joint count
        /// </summary>
        /// <param name="jointCount">Number of robot joints</param>
        /// <param name="dampingFactor">Damping factor for pseudo-inverse stability</param>
        public IKSolver(int jointCount, float dampingFactor)
        {
            _dampingFactor = dampingFactor;

            _jacobianMatrix = DenseMatrix.Build.Dense(6, jointCount);
            _errorVector = Vector<double>.Build.Dense(6);
            _jointDelta = Vector<double>.Build.Dense(jointCount);
            _cachedIdentity = DenseMatrix.Build.DenseIdentity(6);
        }

        /// <summary>
        /// Compute joint angle deltas to move toward target.
        /// Returns null if already converged to target.
        /// </summary>
        /// <param name="currentState">Current end effector state</param>
        /// <param name="targetState">Target end effector state</param>
        /// <param name="joints">Array of joint information</param>
        /// <param name="convergenceThreshold">Distance threshold for convergence</param>
        /// <param name="orientationWeight">Weight for orientation error (0.0-2.0, default 1.0)</param>
        /// <param name="orientationConvergenceThreshold">Orientation convergence threshold in radians (default 0.3)</param>
        /// <param name="overrideDamping">Optional damping override for dynamic damping control</param>
        /// <returns>Joint deltas in radians, or null if converged</returns>
        public Vector<double> ComputeJointDeltas(
            IKState currentState,
            IKState targetState,
            JointInfo[] joints,
            float convergenceThreshold,
            float orientationWeight = 1.0f,
            float orientationConvergenceThreshold = 0.3f,
            float? overrideDamping = null
        )
        {
            _iterationCount++;

            Vector3 positionError = targetState.Position - currentState.Position;
            Vector3 orientationError = CalculateOrientationError(
                currentState.Rotation,
                targetState.Rotation
            );

            if (
                positionError.magnitude < convergenceThreshold
                && orientationError.magnitude < orientationConvergenceThreshold
            )
            {
                return null;
            }

            orientationError *= orientationWeight;

            BuildErrorVector(positionError, orientationError);

            CalculateJacobian(currentState, joints);
            ComputePseudoInverse(overrideDamping);

            return _jointDelta;
        }

        /// <summary>
        /// Compute joint angle deltas with velocity-level IK (PD control).
        /// This is the KEY IMPROVEMENT that eliminates oscillation.
        /// Combines position error with velocity error for natural damping.
        /// </summary>
        /// <param name="currentState">Current end effector state</param>
        /// <param name="targetState">Target end effector state</param>
        /// <param name="currentEndEffectorVelocity">Current end effector velocity (from ArticulationBody)</param>
        /// <param name="targetVelocity">Target velocity from trajectory</param>
        /// <param name="joints">Array of joint information</param>
        /// <param name="convergenceThreshold">Distance threshold for convergence</param>
        /// <param name="Kp">Position gain (how strongly to correct position error)</param>
        /// <param name="Kd">Velocity gain (damping term to prevent overshoot)</param>
        /// <param name="orientationWeight">Weight for orientation error</param>
        /// <param name="orientationConvergenceThreshold">Orientation convergence threshold in radians</param>
        /// <param name="overrideDamping">Optional damping override for pseudo-inverse</param>
        /// <returns>Joint deltas in radians, or null if converged</returns>
        public Vector<double> ComputeJointDeltasWithVelocity(
            IKState currentState,
            IKState targetState,
            Vector3 currentEndEffectorVelocity,
            Vector3 targetVelocity,
            JointInfo[] joints,
            float convergenceThreshold,
            float Kp = 1.0f,
            float Kd = 0.5f,
            float orientationWeight = 1.0f,
            float orientationConvergenceThreshold = 0.3f,
            float? overrideDamping = null
        )
        {
            _iterationCount++;

            Vector3 posError = targetState.Position - currentState.Position;
            Vector3 velError = targetVelocity - currentEndEffectorVelocity;

            Vector3 orientationError = CalculateOrientationError(
                currentState.Rotation,
                targetState.Rotation
            );

            // NOTE: Convergence check disabled in IKSolver
            // Velocity convergence is now handled by RobotController with adaptive thresholds
            // This allows different velocity requirements for grasp vs normal movement
            // IKSolver always returns joint deltas, letting RobotController decide when to stop

            orientationError *= orientationWeight;

            Vector3 combinedError = Kp * posError + Kd * velError;

            // Safety clamp: Prevent matrix instability if target teleports far away
            const float maxErrorMagnitude = 1.0f;
            if (combinedError.magnitude > maxErrorMagnitude)
            {
                combinedError = combinedError.normalized * maxErrorMagnitude;
            }

            BuildErrorVector(combinedError, orientationError);

            CalculateJacobian(currentState, joints);
            ComputePseudoInverse(overrideDamping);

            // ⚠️ CRITICAL: Clamp joint velocities near singularities
            // When arm is fully stretched, Jacobian becomes ill-conditioned
            // and requested velocities can spike to infinity
            const float maxJointVelocity = 5.0f; // rad/sec (configurable)
            for (int i = 0; i < _jointDelta.Count; i++)
            {
                _jointDelta[i] = System.Math.Clamp(
                    _jointDelta[i],
                    -maxJointVelocity,
                    maxJointVelocity
                );
            }

            return _jointDelta;
        }

        /// <summary>
        /// Calculate orientation error as angle-axis representation
        /// </summary>
        private Vector3 CalculateOrientationError(Quaternion current, Quaternion target)
        {
            Quaternion quaternionError = target * Quaternion.Inverse(current);
            quaternionError.ToAngleAxis(out float angleDegree, out Vector3 axis);

            if (float.IsNaN(axis.x) || float.IsInfinity(axis.x))
                return Vector3.zero;

            if (angleDegree > 180f)
                angleDegree -= 360f;

            return axis.normalized * (angleDegree * Mathf.Deg2Rad);
        }

        /// <summary>
        /// Build the 6D error vector (position + orientation)
        /// </summary>
        private void BuildErrorVector(Vector3 posError, Vector3 rotError)
        {
            _errorVector[0] = posError.x;
            _errorVector[1] = posError.y;
            _errorVector[2] = posError.z;
            _errorVector[3] = rotError.x;
            _errorVector[4] = rotError.y;
            _errorVector[5] = rotError.z;
        }

        /// <summary>
        /// Compute the 6xN Jacobian matrix for the robot at its current configuration
        /// </summary>
        private void CalculateJacobian(IKState currentState, JointInfo[] joints)
        {
            if (_jacobianMatrix.ColumnCount != joints.Length)
                _jacobianMatrix = DenseMatrix.Build.Dense(6, joints.Length);
                _jointDelta = Vector<double>.Build.Dense(joints.Length);

            for (int i = 0; i < joints.Length; i++)
            {
                JointInfo joint = joints[i];

                Vector3 vectorJointToEndEffector = currentState.Position - joint.WorldPosition;

                Vector3 linearComponent = Vector3.Cross(joint.WorldAxis, vectorJointToEndEffector);
                Vector3 angularComponent = joint.WorldAxis;

                _jacobianMatrix[0, i] = linearComponent.x;
                _jacobianMatrix[1, i] = linearComponent.y;
                _jacobianMatrix[2, i] = linearComponent.z;
                _jacobianMatrix[3, i] = angularComponent.x;
                _jacobianMatrix[4, i] = angularComponent.y;
                _jacobianMatrix[5, i] = angularComponent.z;
            }
        }

        /// <summary>
        /// Compute the damped least squares pseudo-inverse of the Jacobian
        /// and update joint deltas
        /// </summary>
        /// <param name="overrideDamping">Optional damping override (null uses default)</param>
        private void ComputePseudoInverse(float? overrideDamping = null)
        {
            var jacobianTranspose = _jacobianMatrix.Transpose();
            var jacobianJacobianTranspose = _jacobianMatrix * jacobianTranspose;

            float damping = overrideDamping ?? _dampingFactor;

            var regularized = jacobianJacobianTranspose + damping * damping * _cachedIdentity;

            var y = regularized.LU().Solve(_errorVector);

            _jointDelta = jacobianTranspose * y;
        }
    }

    /// <summary>
    /// Data structure for IK state (end effector position and rotation)
    /// </summary>
    public struct IKState
    {
        public Vector3 Position;
        public Quaternion Rotation;

        public IKState(Vector3 position, Quaternion rotation)
        {
            Position = position;
            Rotation = rotation;
        }
    }

    /// <summary>
    /// Data structure for joint information (position and axis in IK frame)
    /// </summary>
    public struct JointInfo
    {
        public Vector3 WorldPosition;
        public Vector3 WorldAxis;

        public JointInfo(Vector3 worldPosition, Vector3 worldAxis)
        {
            WorldPosition = worldPosition;
            WorldAxis = worldAxis;
        }
    }
}
