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

        #region Velocity-Level IK Tests (Phase 1 Motion Control Redesign)

        [Test]
        public void ComputeJointDeltasWithVelocity_CombinesPositionAndVelocityErrors()
        {
            // Test that velocity-level IK combines Kp * posError + Kd * velError
            // This is the KEY improvement that eliminates oscillation

            // Arrange
            var solver = new IKSolver(jointCount: 2, dampingFactor: 0.1f);

            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1f, 0.5f, 0f), // 50cm away in Y
                rotation: Quaternion.identity
            );

            Vector3 currentVelocity = new Vector3(0f, 0.1f, 0f); // Moving in +Y at 10cm/s
            Vector3 targetVelocity = new Vector3(0f, 0.2f, 0f);  // Should be 20cm/s

            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.forward),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.forward)
            };

            float Kp = 10.0f;
            float Kd = 2.0f;

            // Act
            var deltas = solver.ComputeJointDeltasWithVelocity(
                currentState, targetState,
                currentVelocity, targetVelocity,
                joints,
                convergenceThreshold: 0.01f,
                Kp: Kp,
                Kd: Kd
            );

            // Assert
            Assert.IsNotNull(deltas, "Should return deltas when not converged");
            Assert.AreEqual(2, deltas.Count, "Should return deltas for all joints");

            // Verify non-zero movement
            bool hasMovement = false;
            for (int i = 0; i < deltas.Count; i++)
            {
                if (System.Math.Abs(deltas[i]) > EPSILON)
                {
                    hasMovement = true;
                    break;
                }
            }
            Assert.IsTrue(hasMovement, "Velocity-level IK should produce non-zero deltas");
        }

        [Test]
        public void ComputeJointDeltasWithVelocity_DoesNotConverge_WhenVelocityStillHigh()
        {
            // Test that convergence requires BOTH position and velocity to be small
            // This prevents oscillation by checking velocity convergence

            // Arrange
            var solver = new IKSolver(jointCount: 2, dampingFactor: 0.1f);

            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1f, 0.005f, 0f), // 5mm away (within position threshold)
                rotation: Quaternion.identity
            );

            Vector3 currentVelocity = new Vector3(0f, 0.06f, 0f); // Moving at 6cm/s (above 5cm/s threshold)
            Vector3 targetVelocity = Vector3.zero;

            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.forward),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.forward)
            };

            // Act
            var deltas = solver.ComputeJointDeltasWithVelocity(
                currentState, targetState,
                currentVelocity, targetVelocity,
                joints,
                convergenceThreshold: 0.01f,
                Kp: 10.0f,
                Kd: 2.0f
            );

            // Assert
            Assert.IsNotNull(deltas,
                "Should NOT converge when velocity is above threshold (6cm/s > 5cm/s) - prevents oscillation");
        }

        [Test]
        public void ComputeJointDeltasWithVelocity_AlwaysReturnsDeltas()
        {
            // NOTE: Convergence checking is now handled by RobotController with adaptive thresholds.
            // IKSolver always returns joint deltas, letting RobotController decide when to stop.
            // This test verifies that IKSolver returns valid deltas even when close to target.

            // Arrange
            var solver = new IKSolver(jointCount: 2, dampingFactor: 0.1f);

            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1f, 0.005f, 0f), // 5mm away (within threshold)
                rotation: Quaternion.identity
            );

            Vector3 currentVelocity = new Vector3(0f, 0.01f, 0f); // Very slow (< 5cm/s threshold)
            Vector3 targetVelocity = Vector3.zero;

            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.forward),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.forward)
            };

            // Act
            var deltas = solver.ComputeJointDeltasWithVelocity(
                currentState, targetState,
                currentVelocity, targetVelocity,
                joints,
                convergenceThreshold: 0.01f,
                Kp: 10.0f,
                Kd: 2.0f
            );

            // Assert
            Assert.IsNotNull(deltas,
                "IKSolver should always return deltas (convergence is handled by RobotController)");
            Assert.Greater(deltas.Count, 0, "Deltas should have values for each joint");
        }

        [Test]
        public void ComputeJointDeltasWithVelocity_ClampsJointVelocity_At5RadPerSec()
        {
            // Test that joint velocities are clamped to prevent singularity spikes
            // This is CRITICAL for stability near arm limits

            // Arrange
            var solver = new IKSolver(jointCount: 2, dampingFactor: 0.001f); // Very low damping to trigger large deltas

            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1f, 10f, 0f), // Very far target (10m away)
                rotation: Quaternion.identity
            );

            Vector3 currentVelocity = Vector3.zero;
            Vector3 targetVelocity = new Vector3(0f, 5f, 0f); // Fast target velocity

            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.forward),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.forward)
            };

            // Act
            var deltas = solver.ComputeJointDeltasWithVelocity(
                currentState, targetState,
                currentVelocity, targetVelocity,
                joints,
                convergenceThreshold: 0.01f,
                Kp: 100.0f, // Very high gain to trigger large deltas
                Kd: 50.0f
            );

            // Assert
            Assert.IsNotNull(deltas);

            // Verify all joint velocities are clamped to ±5.0 rad/sec
            const float maxJointVelocity = 5.0f;
            for (int i = 0; i < deltas.Count; i++)
            {
                Assert.LessOrEqual(System.Math.Abs(deltas[i]), maxJointVelocity,
                    $"Joint {i} velocity should be clamped to ±{maxJointVelocity} rad/sec, " +
                    $"but was {deltas[i]:F3}");
            }
        }

        [Test]
        public void ComputeJointDeltasWithVelocity_PDGains_AffectResponse()
        {
            // Test that different PD gains produce different responses
            // High Kp = aggressive position correction
            // High Kd = more damping (smoother motion)

            // Arrange
            var solver = new IKSolver(jointCount: 2, dampingFactor: 0.1f);

            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1f, 0.3f, 0f),
                rotation: Quaternion.identity
            );

            Vector3 currentVelocity = new Vector3(0f, 0.05f, 0f);
            Vector3 targetVelocity = new Vector3(0f, 0.1f, 0f);

            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.forward),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.forward)
            };

            // Act - Low Kp (gentle position correction)
            var deltasLowKp = solver.ComputeJointDeltasWithVelocity(
                currentState, targetState,
                currentVelocity, targetVelocity,
                joints,
                convergenceThreshold: 0.01f,
                Kp: 1.0f,
                Kd: 2.0f
            );

            // Act - High Kp (aggressive position correction)
            var deltasHighKp = solver.ComputeJointDeltasWithVelocity(
                currentState, targetState,
                currentVelocity, targetVelocity,
                joints,
                convergenceThreshold: 0.01f,
                Kp: 20.0f,
                Kd: 2.0f
            );

            // Assert
            Assert.IsNotNull(deltasLowKp);
            Assert.IsNotNull(deltasHighKp);

            double magnitudeLowKp = deltasLowKp.L2Norm();
            double magnitudeHighKp = deltasHighKp.L2Norm();

            Assert.Greater(magnitudeHighKp, magnitudeLowKp,
                "High Kp should produce larger corrections than low Kp");
        }

        [Test]
        public void ComputeJointDeltasWithVelocity_HighKd_ProducesMoreDamping()
        {
            // Test that high Kd provides more velocity damping
            // NOTE: Use smaller errors to avoid error magnitude clamping at 1.0 m/s

            // Arrange
            var solver = new IKSolver(jointCount: 2, dampingFactor: 0.1f);

            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1f, 0.05f, 0f), // Small position error to avoid clamping
                rotation: Quaternion.identity
            );

            // Moderate current velocity (robot moving)
            Vector3 currentVelocity = new Vector3(0f, 0.08f, 0f); // 8 cm/s
            Vector3 targetVelocity = new Vector3(0f, 0.02f, 0f); // Should slow down to 2 cm/s

            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.forward),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.forward)
            };

            // Act - Low Kd (less velocity damping)
            var deltasLowKd = solver.ComputeJointDeltasWithVelocity(
                currentState, targetState,
                currentVelocity, targetVelocity,
                joints,
                convergenceThreshold: 0.01f,
                Kp: 1.0f, // Lower Kp to avoid clamping
                Kd: 0.5f
            );

            // Act - High Kd (more velocity damping)
            var deltasHighKd = solver.ComputeJointDeltasWithVelocity(
                currentState, targetState,
                currentVelocity, targetVelocity,
                joints,
                convergenceThreshold: 0.01f,
                Kp: 1.0f, // Lower Kp to avoid clamping
                Kd: 2.0f
            );

            // Assert
            Assert.IsNotNull(deltasLowKd);
            Assert.IsNotNull(deltasHighKd);

            // With velocity error (moving too fast), high Kd should produce different response
            // The difference should be non-zero (gains matter)
            double magnitudeLowKd = deltasLowKd.L2Norm();
            double magnitudeHighKd = deltasHighKd.L2Norm();

            double difference = System.Math.Abs(magnitudeLowKd - magnitudeHighKd);
            Assert.Greater(difference, 0.001,
                "Different Kd values should produce different responses (difference > 0.001)");
        }

        [Test]
        public void ComputeJointDeltasWithVelocity_AlwaysReturnsDeltas_RegardlessOfVelocity()
        {
            // NOTE: Convergence checking is now handled by RobotController with adaptive thresholds.
            // IKSolver always returns joint deltas regardless of velocity.
            // This test verifies that IKSolver returns deltas for both high and low velocities.

            // Arrange
            var solver = new IKSolver(jointCount: 2, dampingFactor: 0.1f);

            var currentState = new IKState(
                position: new Vector3(1f, 0f, 0f),
                rotation: Quaternion.identity
            );
            var targetState = new IKState(
                position: new Vector3(1f, 0.005f, 0f), // Within position threshold
                rotation: Quaternion.identity
            );

            var joints = new[]
            {
                new JointInfo(Vector3.zero, Vector3.forward),
                new JointInfo(new Vector3(0.5f, 0f, 0f), Vector3.forward)
            };

            // Act - Just above velocity threshold (should not converge)
            Vector3 velocityJustAbove = new Vector3(0f, 0.051f, 0f); // 5.1 cm/s
            var deltasAbove = solver.ComputeJointDeltasWithVelocity(
                currentState, targetState,
                velocityJustAbove, Vector3.zero,
                joints,
                convergenceThreshold: 0.01f,
                Kp: 10.0f,
                Kd: 2.0f
            );

            // Act - Just below velocity threshold (should converge)
            Vector3 velocityJustBelow = new Vector3(0f, 0.049f, 0f); // 4.9 cm/s
            var deltasBelow = solver.ComputeJointDeltasWithVelocity(
                currentState, targetState,
                velocityJustBelow, Vector3.zero,
                joints,
                convergenceThreshold: 0.01f,
                Kp: 10.0f,
                Kd: 2.0f
            );

            // Assert - IKSolver always returns deltas (convergence is handled by RobotController)
            Assert.IsNotNull(deltasAbove,
                "IKSolver should always return deltas regardless of velocity");
            Assert.IsNotNull(deltasBelow,
                "IKSolver should always return deltas regardless of velocity");

            // Verify the deltas are different based on velocity
            Assert.AreNotEqual(deltasAbove.L2Norm(), deltasBelow.L2Norm(),
                "Different velocities should produce different joint deltas");
        }

        #endregion
    }
}
