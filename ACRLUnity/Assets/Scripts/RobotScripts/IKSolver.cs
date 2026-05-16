using Core;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using UnityEngine;

namespace Robotics
{
    /// <summary>Damped least-squares IK solver. Pure C#, no MonoBehaviour dependency.</summary>
    public class IKSolver
    {
        private readonly float _dampingFactor;

        // Pre-alloc MathNet matrices — Jacobian build + J*J^T
        private Matrix<double> _jacobianMatrix;
        private Vector<double> _errorVector;
        private Vector<double> _jointDelta;
        private Matrix<double> _jacobianTranspose;
        private Matrix<double> _jacobianJacobianTranspose;

        // Zero-alloc 6x6 LU solve — avoids MathNet LU().Solve() allocation
        private readonly double[,] _luA = new double[6, 6];
        private readonly double[] _luB = new double[6];
        private readonly double[] _luY = new double[6];
        private readonly int[] _luPiv = new int[6];

        private int _iterationCount;
        public int IterationCount => _iterationCount;

        public void ResetIterationCount()
        {
            _iterationCount = 0;
        }

        public void SetIterationCount(int count)
        {
            _iterationCount = count;
        }

        public IKSolver(int jointCount, float dampingFactor)
        {
            _dampingFactor = dampingFactor;

            _jacobianMatrix = DenseMatrix.Build.Dense(6, jointCount);
            _errorVector = Vector<double>.Build.Dense(6);
            _jointDelta = Vector<double>.Build.Dense(jointCount);

            _jacobianTranspose = DenseMatrix.Build.Dense(jointCount, 6);
            _jacobianJacobianTranspose = DenseMatrix.Build.Dense(6, 6);
        }

        /// <summary>Compute joint deltas toward target. Returns null if already converged.</summary>
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

        /// <summary>PD IK: Kp*posError + Kd*velError. Convergence decision left to RobotController.</summary>
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

            // Convergence handled by RobotController — adaptive thresholds differ per mode
            orientationError *= orientationWeight;

            Vector3 combinedError = Kp * posError + Kd * velError;

            // Guard against matrix instability on large teleport jumps
            const float maxErrorMagnitude = 1.0f;
            if (combinedError.magnitude > maxErrorMagnitude)
            {
                combinedError = combinedError.normalized * maxErrorMagnitude;
            }

            BuildErrorVector(combinedError, orientationError);

            CalculateJacobian(currentState, joints);
            ComputePseudoInverse(overrideDamping);

            // Clamp near singularities — ill-conditioned Jacobian → velocity spikes
            for (int i = 0; i < _jointDelta.Count; i++)
            {
                _jointDelta[i] = System.Math.Clamp(
                    _jointDelta[i],
                    -RobotConstants.MAX_JOINT_VELOCITY_RAD_PER_SEC,
                    RobotConstants.MAX_JOINT_VELOCITY_RAD_PER_SEC
                );
            }

            return _jointDelta;
        }

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

        private void BuildErrorVector(Vector3 posError, Vector3 rotError)
        {
            _errorVector[0] = posError.x;
            _errorVector[1] = posError.y;
            _errorVector[2] = posError.z;
            _errorVector[3] = rotError.x;
            _errorVector[4] = rotError.y;
            _errorVector[5] = rotError.z;
        }

        private void CalculateJacobian(IKState currentState, JointInfo[] joints)
        {
            if (_jacobianMatrix.ColumnCount != joints.Length)
            {
                _jacobianMatrix = DenseMatrix.Build.Dense(6, joints.Length);
                _jointDelta = Vector<double>.Build.Dense(joints.Length);
                _jacobianTranspose = DenseMatrix.Build.Dense(joints.Length, 6);
                _jacobianJacobianTranspose = DenseMatrix.Build.Dense(6, 6);
            }

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

        private void ComputePseudoInverse(float? overrideDamping = null)
        {
            _jacobianMatrix.Transpose(_jacobianTranspose);
            _jacobianMatrix.Multiply(_jacobianTranspose, _jacobianJacobianTranspose);

            double damping = overrideDamping ?? _dampingFactor;
            double lambda2 = damping * damping;

            // JJ^T + λ²I into raw array — skip MathNet LU alloc
            for (int r = 0; r < 6; r++)
            {
                for (int c = 0; c < 6; c++)
                    _luA[r, c] = _jacobianJacobianTranspose[r, c];
                _luA[r, r] += lambda2;
                _luB[r] = _errorVector[r];
            }

            SolveLU6x6(_luA, _luB, _luY, _luPiv);

            // J^T * y → joint deltas
            int n = _jointDelta.Count;
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < 6; j++)
                    sum += _jacobianTranspose[i, j] * _luY[j];
                _jointDelta[i] = sum;
            }
        }

        private static void SolveLU6x6(double[,] a, double[] b, double[] y, int[] piv)
        {
            const int n = 6;
            for (int i = 0; i < n; i++)
                piv[i] = i;

            for (int k = 0; k < n; k++)
            {
                int maxRow = k;
                double maxVal = System.Math.Abs(a[k, k]);
                for (int i = k + 1; i < n; i++)
                {
                    double v = System.Math.Abs(a[i, k]);
                    if (v > maxVal)
                    {
                        maxVal = v;
                        maxRow = i;
                    }
                }
                if (maxRow != k)
                {
                    for (int j = 0; j < n; j++)
                    {
                        double t = a[k, j];
                        a[k, j] = a[maxRow, j];
                        a[maxRow, j] = t;
                    }
                    int tmp = piv[k];
                    piv[k] = piv[maxRow];
                    piv[maxRow] = tmp;
                }
                if (a[k, k] == 0.0)
                    continue; // singular column — skip
                double inv = 1.0 / a[k, k];
                for (int i = k + 1; i < n; i++)
                {
                    a[i, k] *= inv;
                    for (int j = k + 1; j < n; j++)
                        a[i, j] -= a[i, k] * a[k, j];
                }
            }

            for (int i = 0; i < n; i++)
                y[i] = b[piv[i]];
            // Forward sub (unit lower triangular L)
            for (int i = 1; i < n; i++)
            for (int j = 0; j < i; j++)
                y[i] -= a[i, j] * y[j];
            // Back sub
            for (int i = n - 1; i >= 0; i--)
            {
                for (int j = i + 1; j < n; j++)
                    y[i] -= a[i, j] * y[j];
                y[i] /= a[i, i];
            }
        }
    }

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
