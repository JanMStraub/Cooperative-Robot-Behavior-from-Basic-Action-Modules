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

            // Pre-allocate matrices (6 DOF: 3 position + 3 orientation)
            _jacobianMatrix = DenseMatrix.Build.Dense(6, jointCount);
            _errorVector = Vector<double>.Build.Dense(6);
            _jointDelta = Vector<double>.Build.Dense(jointCount);
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
            // Increment iteration counter
            _iterationCount++;

            // Calculate position and orientation errors
            Vector3 positionError = targetState.Position - currentState.Position;
            Vector3 orientationError = CalculateOrientationError(
                currentState.Rotation,
                targetState.Rotation
            );

            if (positionError.magnitude < convergenceThreshold && orientationError.magnitude < orientationConvergenceThreshold)
            {
                return null; // Converged
            }

            // Apply orientation weight to emphasize/de-emphasize orientation precision
            orientationError *= orientationWeight;

            // Build 6D error vector
            BuildErrorVector(positionError, orientationError);

            // Compute Jacobian and solve for joint deltas
            CalculateJacobian(currentState, joints);
            ComputePseudoInverse(overrideDamping);

            return _jointDelta;
        }

        /// <summary>
        /// Calculate orientation error as angle-axis representation
        /// </summary>
        private Vector3 CalculateOrientationError(Quaternion current, Quaternion target)
        {
            Quaternion quaternionError = target * Quaternion.Inverse(current);
            quaternionError.ToAngleAxis(out float angleDegree, out Vector3 axis);

            // Safety check for singularities
            if (float.IsNaN(axis.x) || float.IsInfinity(axis.x))
                return Vector3.zero;

            // Ensure shortest path: if angle > 180, map to negative range (e.g. 350 -> -10)
            if (angleDegree > 180f)
                angleDegree -= 360f;

            // Convert to radians and return axis-angle vector
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

            for (int i = 0; i < joints.Length; i++)
            {
                JointInfo joint = joints[i];

                // Vector from joint to end effector (in IK frame)
                Vector3 vectorJointToEndEffector = currentState.Position - joint.WorldPosition;

                // Jacobian column (linear and angular components)
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
            // Jacobian: 6xN
            var jacobianTranspose = _jacobianMatrix.Transpose(); // Nx6
            var jacobianJacobianTranspose = _jacobianMatrix * jacobianTranspose; // 6x6
            var identity = DenseMatrix.Build.DenseIdentity(jacobianJacobianTranspose.RowCount);

            // Use override damping if provided, otherwise use default
            float damping = overrideDamping ?? _dampingFactor;

            // Damped least squares regularization
            var regularized = jacobianJacobianTranspose + damping * damping * identity;

            // Solve for intermediate vector: y = (J*J^T + λ²I)^-1 * error
            var y = regularized.Solve(_errorVector); // 6x1

            // Compute joint deltas: Δθ = J^T * y
            _jointDelta = jacobianTranspose * y; // Nx1
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
