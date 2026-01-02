using NUnit.Framework;
using Robotics;
using UnityEngine;

namespace Tests.EditMode
{
    /// <summary>
    /// Tests for IKSolver inverse kinematics computation
    /// </summary>
    public class IKSolverTests
    {
        private const float EPSILON = 0.001f;

        [Test]
        public void ComputeJointDeltas_Returns_Null_When_Converged()
        {
            // Arrange
            var solver = new IKSolver(jointCount: 2, dampingFactor: 0.1f);
            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1f, 0f, 0f), // Same position (converged)
                rotation: Quaternion.identity
            );
            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.up),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.up)
            };

            // Act
            var deltas = solver.ComputeJointDeltas(
                currentState, targetState, joints,
                convergenceThreshold: 0.01f);

            // Assert
            Assert.IsNull(deltas, "Should return null when already converged");
        }

        [Test]
        public void ComputeJointDeltas_Returns_NonNull_When_Not_Converged()
        {
            // Arrange
            var solver = new IKSolver(jointCount: 2, dampingFactor: 0.1f);
            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1.5f, 0.2f, 0f), // Different position
                rotation: Quaternion.identity
            );
            // Use Z-axis rotation (forward) to enable X-Y plane movement
            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.forward),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.forward)
            };

            // Act
            var deltas = solver.ComputeJointDeltas(
                currentState, targetState, joints,
                convergenceThreshold: 0.01f);

            // Assert
            Assert.IsNotNull(deltas, "Should return joint deltas when not converged");
            Assert.AreEqual(2, deltas.Count, "Should return deltas for all joints");
        }

        [Test]
        public void ComputeJointDeltas_Produces_Valid_Vector_Dimensions()
        {
            // Arrange
            int jointCount = 6; // AR4 has 6 joints
            var solver = new IKSolver(jointCount, dampingFactor: 0.1f);

            var currentState = new IKState(
                position: Vector3.zero,
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(0.5f, 0.5f, 0.5f),
                rotation: Quaternion.Euler(45f, 0f, 0f)
            );

            var joints = new JointInfo[jointCount];
            for (int i = 0; i < jointCount; i++)
            {
                joints[i] = new JointInfo(
                    worldPosition: new Vector3(i * 0.1f, 0f, 0f),
                    worldAxis: Vector3.up
                );
            }

            // Act
            var deltas = solver.ComputeJointDeltas(
                currentState, targetState, joints,
                convergenceThreshold: 0.01f);

            // Assert
            Assert.IsNotNull(deltas);
            Assert.AreEqual(jointCount, deltas.Count, "Delta vector should match joint count");
        }

        [Test]
        public void ComputeJointDeltas_Moves_Toward_Target()
        {
            // Arrange
            var solver = new IKSolver(jointCount: 2, dampingFactor: 0.1f);
            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1f, 0.5f, 0f), // Move in +Y direction
                rotation: Quaternion.identity
            );
            // Use Z-axis rotation (forward) to enable X-Y plane movement
            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.forward),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.forward)
            };

            // Act
            var deltas = solver.ComputeJointDeltas(
                currentState, targetState, joints,
                convergenceThreshold: 0.01f);

            // Assert
            Assert.IsNotNull(deltas);
            // At least one joint should have non-zero delta to move toward target
            bool hasMovement = false;
            for (int i = 0; i < deltas.Count; i++)
            {
                if (System.Math.Abs(deltas[i]) > EPSILON)
                {
                    hasMovement = true;
                    break;
                }
            }
            Assert.IsTrue(hasMovement, "Should produce non-zero deltas to move toward target");
        }

        [Test]
        public void IKState_Stores_Position_And_Rotation()
        {
            // Arrange
            var position = new Vector3(1.2f, 3.4f, 5.6f);
            var rotation = Quaternion.Euler(10f, 20f, 30f);

            // Act
            var state = new IKState(position, rotation);

            // Assert
            Assert.AreEqual(position, state.Position);
            Assert.AreEqual(rotation, state.Rotation);
        }

        [Test]
        public void JointInfo_Stores_Position_And_Axis()
        {
            // Arrange
            var worldPosition = new Vector3(0.5f, 0.2f, 0.1f);
            var worldAxis = Vector3.forward;

            // Act
            var joint = new JointInfo(worldPosition, worldAxis);

            // Assert
            Assert.AreEqual(worldPosition, joint.WorldPosition);
            Assert.AreEqual(worldAxis, joint.WorldAxis);
        }

        [Test]
        public void Different_Convergence_Thresholds_Affect_Convergence_Detection()
        {
            // Arrange
            var solver = new IKSolver(jointCount: 2, dampingFactor: 0.1f);
            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1.005f, 0f, 0f), // 5mm difference
                rotation: Quaternion.identity
            );
            // Use Z-axis rotation (forward) to enable X-Y plane movement
            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.forward),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.forward)
            };

            // Act - Loose threshold (should converge)
            var deltasLoose = solver.ComputeJointDeltas(
                currentState, targetState, joints,
                convergenceThreshold: 0.01f); // 10mm

            // Act - Tight threshold (should not converge)
            var deltasTight = solver.ComputeJointDeltas(
                currentState, targetState, joints,
                convergenceThreshold: 0.001f); // 1mm

            // Assert
            Assert.IsNull(deltasLoose, "Should converge with loose threshold");
            Assert.IsNotNull(deltasTight, "Should not converge with tight threshold");
        }

        [Test]
        public void Damping_Factor_Is_Applied()
        {
            // Arrange - Two solvers with different damping
            var solverLowDamping = new IKSolver(jointCount: 2, dampingFactor: 0.01f);
            var solverHighDamping = new IKSolver(jointCount: 2, dampingFactor: 1.0f);

            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1f, 0.5f, 0f),
                rotation: Quaternion.identity
            );
            // Use Z-axis rotation (forward) to enable X-Y plane movement
            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.forward),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.forward)
            };

            // Act
            var deltasLow = solverLowDamping.ComputeJointDeltas(
                currentState, targetState, joints,
                convergenceThreshold: 0.01f);
            var deltasHigh = solverHighDamping.ComputeJointDeltas(
                currentState, targetState, joints,
                convergenceThreshold: 0.01f);

            // Assert
            Assert.IsNotNull(deltasLow);
            Assert.IsNotNull(deltasHigh);

            // High damping should produce smaller deltas (more conservative movement)
            double magnitudeLow = deltasLow.L2Norm();
            double magnitudeHigh = deltasHigh.L2Norm();

            Assert.Greater(magnitudeLow, magnitudeHigh,
                "Low damping should produce larger joint deltas than high damping");
        }
    }
}
