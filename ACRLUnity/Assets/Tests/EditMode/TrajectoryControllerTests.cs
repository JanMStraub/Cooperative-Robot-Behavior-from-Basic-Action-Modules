using System;
using NUnit.Framework;
using UnityEngine;
using Robotics;
using RobotScripts;

namespace Tests.EditMode
{
    /// <summary>
    /// Tests for TrajectoryController - Phase 1 Motion Control Redesign validation.
    /// Tests PD control, feedforward terms, velocity profiles, and trajectory caching.
    ///
    /// Key Features Tested:
    /// - PD control law: correction = K_p*(pos error) + K_d*(vel error)
    /// - Trapezoidal velocity profiles (accel -> cruise -> decel)
    /// - Feedforward terms from trajectory
    /// - Caching to avoid Update/FixedUpdate jitter
    /// </summary>
    public class TrajectoryControllerTests
    {
        private const float EPSILON = TestConstants.EPSILON;

        // Default controller used by the majority of tests (Kp=10, Kd=2)
        private TrajectoryController _controller;

        [SetUp]
        public void SetUp()
        {
            _controller = new TrajectoryController();
        }

        #region PD Control Tests

        [Test]
        public void Constructor_UsesDefaultGains_WhenNotSpecified()
        {
            // Arrange & Act — use shared instance for default-gains test
            var controller = _controller;

            // Compute correction to verify gains are applied
            Vector3 posError = new Vector3(0.1f, 0f, 0f);
            Vector3 velError = new Vector3(0.05f, 0f, 0f);

            Vector3 correction = controller.ComputeCartesianCorrection(
                currentPos: Vector3.zero,
                targetPos: posError,
                currentVel: Vector3.zero,
                targetVel: velError
            );

            // Assert - Default gains are Kp=10, Kd=2
            // Expected: correction = 10*0.1 + 2*0.05 = 1.0 + 0.1 = 1.1
            Assert.AreEqual(1.1f, correction.x, EPSILON,
                "Should use default Kp=10.0 and Kd=2.0");
        }

        [Test]
        public void Constructor_UsesCustomGains_WhenSpecified()
        {
            // Test that custom PD gains are respected

            // Arrange
            Vector3 customKp = new Vector3(5f, 5f, 5f);
            Vector3 customKd = new Vector3(1f, 1f, 1f);
            var controller = new TrajectoryController(customKp, customKd);

            // Act
            Vector3 correction = controller.ComputeCartesianCorrection(
                currentPos: Vector3.zero,
                targetPos: new Vector3(0.1f, 0f, 0f),
                currentVel: Vector3.zero,
                targetVel: new Vector3(0.05f, 0f, 0f)
            );

            // Assert - Expected: 5*0.1 + 1*0.05 = 0.5 + 0.05 = 0.55
            Assert.AreEqual(0.55f, correction.x, EPSILON,
                "Should use custom Kp=5.0 and Kd=1.0");
        }

        [Test]
        public void SetGains_UpdatesPDGains()
        {
            // Test that SetGains updates the controller's gains

            // Arrange & Act - Set new gains on shared controller
            var controller = _controller;
            controller.SetGains(
                positionGains: new Vector3(20f, 20f, 20f),
                velocityGains: new Vector3(4f, 4f, 4f)
            );

            Vector3 correction = controller.ComputeCartesianCorrection(
                currentPos: Vector3.zero,
                targetPos: new Vector3(0.1f, 0f, 0f),
                currentVel: Vector3.zero,
                targetVel: new Vector3(0.05f, 0f, 0f)
            );

            // Assert - Expected: 20*0.1 + 4*0.05 = 2.0 + 0.2 = 2.2
            Assert.AreEqual(2.2f, correction.x, EPSILON,
                "SetGains should update to Kp=20.0 and Kd=4.0");
        }

        [Test]
        public void ComputeCartesianCorrection_CombinesPositionAndVelocityErrors()
        {
            // Test the KEY improvement: PD control combines position and velocity errors
            // This eliminates oscillation by adding damping

            // Arrange
            var controller = new TrajectoryController(
                positionGains: new Vector3(10f, 10f, 10f),
                velocityGains: new Vector3(2f, 2f, 2f)
            );

            Vector3 currentPos = new Vector3(1f, 2f, 3f);
            Vector3 targetPos = new Vector3(1.5f, 2.3f, 3.2f);
            Vector3 currentVel = new Vector3(0.1f, 0.05f, 0.0f);
            Vector3 targetVel = new Vector3(0.2f, 0.1f, 0.05f);

            // Act
            Vector3 correction = controller.ComputeCartesianCorrection(
                currentPos, targetPos, currentVel, targetVel
            );

            // Assert - Verify PD control law: correction = Kp*posError + Kd*velError
            Vector3 posError = targetPos - currentPos; // (0.5, 0.3, 0.2)
            Vector3 velError = targetVel - currentVel; // (0.1, 0.05, 0.05)
            Vector3 expected = new Vector3(
                10f * posError.x + 2f * velError.x, // 10*0.5 + 2*0.1 = 5.2
                10f * posError.y + 2f * velError.y, // 10*0.3 + 2*0.05 = 3.1
                10f * posError.z + 2f * velError.z  // 10*0.2 + 2*0.05 = 2.1
            );

            Assert.AreEqual(expected.x, correction.x, EPSILON, "X correction should match PD law");
            Assert.AreEqual(expected.y, correction.y, EPSILON, "Y correction should match PD law");
            Assert.AreEqual(expected.z, correction.z, EPSILON, "Z correction should match PD law");
        }

        [Test]
        public void ComputeCartesianCorrection_HighKp_ProducesLargerPositionCorrection()
        {
            // Test that high Kp produces more aggressive position correction

            // Arrange
            var controllerLowKp = new TrajectoryController(
                positionGains: new Vector3(5f, 5f, 5f),
                velocityGains: new Vector3(2f, 2f, 2f)
            );
            var controllerHighKp = new TrajectoryController(
                positionGains: new Vector3(20f, 20f, 20f),
                velocityGains: new Vector3(2f, 2f, 2f)
            );

            Vector3 posError = new Vector3(0.1f, 0f, 0f);

            // Act
            Vector3 correctionLow = controllerLowKp.ComputeCartesianCorrection(
                Vector3.zero, posError, Vector3.zero, Vector3.zero
            );
            Vector3 correctionHigh = controllerHighKp.ComputeCartesianCorrection(
                Vector3.zero, posError, Vector3.zero, Vector3.zero
            );

            // Assert
            Assert.Greater(correctionHigh.x, correctionLow.x,
                "High Kp should produce larger position correction than low Kp");
        }

        [Test]
        public void ComputeCartesianCorrection_HighKd_ProducesMoreDamping()
        {
            // Test that high Kd provides more velocity damping

            // Arrange
            var controllerLowKd = new TrajectoryController(
                positionGains: new Vector3(10f, 10f, 10f),
                velocityGains: new Vector3(0.5f, 0.5f, 0.5f)
            );
            var controllerHighKd = new TrajectoryController(
                positionGains: new Vector3(10f, 10f, 10f),
                velocityGains: new Vector3(5f, 5f, 5f)
            );

            Vector3 velError = new Vector3(0.5f, 0f, 0f); // High velocity error

            // Act
            Vector3 correctionLow = controllerLowKd.ComputeCartesianCorrection(
                Vector3.zero, Vector3.zero, Vector3.zero, velError
            );
            Vector3 correctionHigh = controllerHighKd.ComputeCartesianCorrection(
                Vector3.zero, Vector3.zero, Vector3.zero, velError
            );

            // Assert
            Assert.Greater(correctionHigh.x, correctionLow.x,
                "High Kd should produce larger damping correction than low Kd");
        }

        [Test]
        public void ComputeCartesianCorrection_WorksPerAxis()
        {
            // Test that PD gains are applied independently per axis

            // Arrange
            var controller = new TrajectoryController(
                positionGains: new Vector3(10f, 5f, 2f), // Different per axis
                velocityGains: new Vector3(2f, 1f, 0.5f)
            );

            Vector3 posError = new Vector3(0.1f, 0.1f, 0.1f);
            Vector3 velError = new Vector3(0.05f, 0.05f, 0.05f);

            // Act
            Vector3 correction = controller.ComputeCartesianCorrection(
                Vector3.zero, posError, Vector3.zero, velError
            );

            // Assert
            // X: 10*0.1 + 2*0.05 = 1.1
            // Y: 5*0.1 + 1*0.05 = 0.55
            // Z: 2*0.1 + 0.5*0.05 = 0.225
            Assert.AreEqual(1.1f, correction.x, EPSILON, "X should use Kp=10, Kd=2");
            Assert.AreEqual(0.55f, correction.y, EPSILON, "Y should use Kp=5, Kd=1");
            Assert.AreEqual(0.225f, correction.z, EPSILON, "Z should use Kp=2, Kd=0.5");
        }

        #endregion

        #region Trajectory State Tests

        [Test]
        public void GetTrajectoryState_ReturnsPositionVelocityAcceleration()
        {
            // Test that GetTrajectoryState returns all three trajectory components

            // Arrange
            var controller = _controller;
            var path = CreateSimplePath();
            var profile = VelocityProfile.CreateTrapezoidal(
                totalDistance: path.totalDistance,
                maxVelocity: 0.5f,
                acceleration: 1.0f
            );

            // Act
            var (targetPos, targetVel, targetAccel) = controller.GetTrajectoryState(
                currentTime: 0.1f,
                path: path,
                velocityProfile: profile
            );

            // Assert
            Assert.IsNotNull(targetPos, "Should return target position");
            Assert.IsNotNull(targetVel, "Should return target velocity");
            Assert.IsNotNull(targetAccel, "Should return target acceleration");
        }

        [Test]
        public void GetTrajectoryState_CachesResult_ForSameTime()
        {
            // Test that trajectory state is cached to avoid recomputation
            // This prevents Update/FixedUpdate jitter

            // Arrange
            var controller = _controller;
            var path = CreateSimplePath();
            var profile = VelocityProfile.CreateTrapezoidal(path.totalDistance, 0.5f, 1.0f);

            // Act - Call twice with same time
            var (pos1, vel1, accel1) = controller.GetTrajectoryState(0.1f, path, profile);
            var (pos2, vel2, accel2) = controller.GetTrajectoryState(0.1f, path, profile);

            // Assert - Should return identical cached values
            Assert.AreEqual(pos1, pos2, "Position should be cached");
            Assert.AreEqual(vel1, vel2, "Velocity should be cached");
            Assert.AreEqual(accel1, accel2, "Acceleration should be cached");
        }

        [Test]
        public void GetTrajectoryState_RecomputesWhenTimeChanges()
        {
            // Test that cache is invalidated when time changes

            // Arrange
            var controller = _controller;
            var path = CreateSimplePath();
            var profile = VelocityProfile.CreateTrapezoidal(path.totalDistance, 0.5f, 1.0f);

            // Act
            var (pos1, vel1, _) = controller.GetTrajectoryState(0.0f, path, profile);
            var (pos2, vel2, _) = controller.GetTrajectoryState(0.5f, path, profile);

            // Assert - Should be different (trajectory progressed)
            Assert.AreNotEqual(pos1, pos2, "Position should change as time progresses");
            Assert.AreNotEqual(vel1, vel2, "Velocity should change as time progresses");
        }

        [Test]
        public void GetTrajectoryState_ReturnsEndPosition_AtEndOfTrajectory()
        {
            // Test that trajectory returns end position when at end of path
            // Note: Testing at a reasonable time within the trajectory rather than far past end

            // Arrange
            var controller = _controller;
            var path = CreateSimplePath(); // 1m total
            var profile = VelocityProfile.CreateTrapezoidal(path.totalDistance, 0.5f, 1.0f);

            // Calculate actual time to complete trajectory: with trapezoidal profile
            // For 1m path with max_vel=0.5 and accel=1.0:
            // Time to accel: t = v/a = 0.5/1.0 = 0.5s
            // Distance during accel: d = 0.5*1.0*0.5^2 = 0.125m
            // Cruise distance: 1.0 - 2*0.125 = 0.75m
            // Time during cruise: 0.75 / 0.5 = 1.5s
            // Time during decel: 0.5s
            // Total time: 0.5 + 1.5 + 0.5 = 2.5s

            // Act - Request time at end of trajectory
            var (targetPos, _, _) = controller.GetTrajectoryState(
                currentTime: 2.5f, // At end of trajectory
                path: path,
                velocityProfile: profile
            );

            // Assert - Should be at end of path
            Vector3 endPos = path.GetWaypointAtDistance(path.totalDistance).position;

            Assert.AreEqual(endPos.x, targetPos.x, 0.01f,
                $"X should be at end of path: expected {endPos.x}, got {targetPos.x}");
            Assert.AreEqual(endPos.y, targetPos.y, 0.01f,
                $"Y should be at end of path: expected {endPos.y}, got {targetPos.y}");
            Assert.AreEqual(endPos.z, targetPos.z, 0.01f,
                $"Z should be at end of path: expected {endPos.z}, got {targetPos.z}");
        }

        [Test]
        public void GetTrajectoryState_VelocityClampedToMaximum()
        {
            // Test that velocity is clamped to MAX_VELOCITY (0.5 m/s safety limit)

            // Arrange
            var controller = _controller;
            var path = CreateSimplePath();
            var profile = VelocityProfile.CreateTrapezoidal(
                totalDistance: path.totalDistance,
                maxVelocity: 10.0f, // Unreasonably high
                acceleration: 5.0f
            );

            // Act
            var (_, targetVel, _) = controller.GetTrajectoryState(0.5f, path, profile);

            // Assert - Should be clamped to 0.5 m/s
            Assert.LessOrEqual(targetVel.magnitude, 0.5f + EPSILON,
                "Velocity should be clamped to MAX_VELOCITY (0.5 m/s)");
        }

        #endregion

        #region Velocity Profile Tests

        [Test]
        public void GetTrajectoryState_FollowsVelocityProfile_AccelerationPhase()
        {
            // Test that velocity increases during acceleration phase

            // Arrange
            var controller = _controller;
            var path = CreateSimplePath();
            var profile = VelocityProfile.CreateTrapezoidal(path.totalDistance, 0.5f, 1.0f);

            // Act - Sample at start and during acceleration
            var (_, vel1, _) = controller.GetTrajectoryState(0.0f, path, profile);
            var (_, vel2, _) = controller.GetTrajectoryState(0.1f, path, profile);
            var (_, vel3, _) = controller.GetTrajectoryState(0.2f, path, profile);

            // Assert - Velocity should increase
            Assert.Less(vel1.magnitude, vel2.magnitude,
                "Velocity should increase during acceleration phase");
            Assert.Less(vel2.magnitude, vel3.magnitude,
                "Velocity should continue increasing");
        }

        [Test]
        public void GetTrajectoryState_FollowsVelocityProfile_CruisePhase()
        {
            // Test that velocity is constant during cruise phase

            // Arrange
            var controller = _controller;
            var path = CreateLongPath(); // Long path to ensure cruise phase exists
            var profile = VelocityProfile.CreateTrapezoidal(path.totalDistance, 0.3f, 1.0f);

            // Act - Sample during cruise phase (after acceleration)
            float cruiseTime = 0.5f; // Mid-trajectory
            var (_, vel1, _) = controller.GetTrajectoryState(cruiseTime, path, profile);
            var (_, vel2, _) = controller.GetTrajectoryState(cruiseTime + 0.1f, path, profile);

            // Assert - Velocity should be approximately constant
            Assert.AreEqual(vel1.magnitude, vel2.magnitude, 0.05f,
                "Velocity should be constant during cruise phase");
        }

        [Test]
        public void GetTrajectoryState_TriangularProfile_NoCruisePhase()
        {
            // Test triangular profile (short distance, no cruise phase)

            // Arrange
            var controller = _controller;
            var path = CreateShortPath(); // Very short path (0.1m)
            var profile = VelocityProfile.CreateTrapezoidal(
                totalDistance: path.totalDistance,
                maxVelocity: 0.5f,
                acceleration: 1.0f
            );

            // Assert - Should create triangular profile (no cruise)
            Assert.AreEqual(0f, profile.cruisePhaseDistance, EPSILON,
                "Short path should have no cruise phase (triangular profile)");
        }

        #endregion

        #region Feedforward Tests

        [Test]
        public void GetTrajectoryState_ProvidesTargetVelocity_ForFeedforward()
        {
            // Test that trajectory provides target velocity for feedforward control

            // Arrange
            var controller = _controller;
            var path = CreateSimplePath();
            var profile = VelocityProfile.CreateTrapezoidal(path.totalDistance, 0.5f, 1.0f);

            // Act
            var (_, targetVel, _) = controller.GetTrajectoryState(0.2f, path, profile);

            // Assert
            Assert.Greater(targetVel.magnitude, 0f,
                "Should provide non-zero target velocity for feedforward");
        }

        [Test]
        public void GetCachedTargetVelocity_ReturnsLastComputedVelocity()
        {
            // Test external access to cached target velocity

            // Arrange
            var controller = _controller;
            var path = CreateSimplePath();
            var profile = VelocityProfile.CreateTrapezoidal(path.totalDistance, 0.5f, 1.0f);

            // Act
            var (_, expectedVel, _) = controller.GetTrajectoryState(0.2f, path, profile);
            Vector3 cachedVel = controller.GetCachedTargetVelocity();

            // Assert
            Assert.AreEqual(expectedVel, cachedVel,
                "Cached velocity should match last GetTrajectoryState call");
        }

        #endregion

        #region Reset Tests

        [Test]
        public void Reset_ClearsCachedState()
        {
            // Test that Reset clears the cached trajectory state

            // Arrange
            var controller = _controller;
            var path = CreateSimplePath();
            var profile = VelocityProfile.CreateTrapezoidal(path.totalDistance, 0.5f, 1.0f);

            // Build up cached state
            controller.GetTrajectoryState(0.5f, path, profile);
            Vector3 cachedVelBefore = controller.GetCachedTargetVelocity();
            Assert.Greater(cachedVelBefore.magnitude, 0f, "Setup: Should have cached velocity");

            // Act
            controller.Reset();

            // Assert
            Vector3 cachedVelAfter = controller.GetCachedTargetVelocity();
            Assert.AreEqual(Vector3.zero, cachedVelAfter,
                "Reset should clear cached velocity to zero");
        }

        [Test]
        public void Reset_ForcesRecomputation_OnNextCall()
        {
            // Test that Reset invalidates cache and forces recomputation

            // Arrange
            var controller = _controller;
            var path = CreateSimplePath();
            var profile = VelocityProfile.CreateTrapezoidal(path.totalDistance, 0.5f, 1.0f);

            // Cache state at t=0.5
            var (pos1, _, _) = controller.GetTrajectoryState(0.5f, path, profile);

            // Act - Reset and query again at same time
            controller.Reset();
            var (pos2, _, _) = controller.GetTrajectoryState(0.5f, path, profile);

            // Assert - Should recompute (positions should match since same time/path)
            Assert.AreEqual(pos1.x, pos2.x, EPSILON,
                "Should recompute same position after reset");
        }

        #endregion

        #region Edge Case Tests

        [Test]
        public void GetTrajectoryState_ZeroDistancePath_ReturnsStartPosition()
        {
            // A path with totalDistance == 0 should return the start position without
            // division-by-zero. The velocity and acceleration should be zero.

            // Arrange
            var controller = _controller;
            var path = new CartesianPath
            {
                waypoints = new System.Collections.Generic.List<CartesianWaypoint>
                {
                    new CartesianWaypoint
                    {
                        position = new Vector3(0.5f, 0.2f, 0.1f),
                        rotation = Quaternion.identity,
                        distanceFromStart = 0f
                    }
                },
                totalDistance = 0f,
                maxVelocity = 0.5f,
                acceleration = 1.0f
            };
            var profile = VelocityProfile.CreateTrapezoidal(0f, 0.5f, 1.0f);

            // Act — should not throw
            var (targetPos, targetVel, _) = controller.GetTrajectoryState(0f, path, profile);

            // Assert — position should be the single waypoint; velocity should be zero
            Assert.AreEqual(path.waypoints[0].position, targetPos,
                "Zero-distance path should return the only waypoint position");
            Assert.AreEqual(0f, targetVel.magnitude, EPSILON,
                "Zero-distance path should have zero target velocity");
        }

        [Test]
        public void VelocityProfile_ZeroAcceleration_DoesNotDivideByZero()
        {
            // VelocityProfile.CreateTrapezoidal with acceleration == 0 should either throw a
            // clear ArgumentException or produce a valid (non-NaN) profile.
            // This test documents the expected behavior.

            // Act & Assert — document that zero acceleration is caught early
            Assert.Throws<ArgumentException>(() =>
                VelocityProfile.CreateTrapezoidal(totalDistance: 1f, maxVelocity: 0.5f, acceleration: 0f),
                "Zero acceleration should throw ArgumentException (would cause division by zero internally)");
        }

        [Test]
        public void VelocityProfile_ZeroMaxVelocity_DoesNotDivideByZero()
        {
            // VelocityProfile.CreateTrapezoidal with maxVelocity == 0 should throw.

            Assert.Throws<ArgumentException>(() =>
                VelocityProfile.CreateTrapezoidal(totalDistance: 1f, maxVelocity: 0f, acceleration: 1.0f),
                "Zero maxVelocity should throw ArgumentException");
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Create a simple 2-waypoint path for testing (1 meter long)
        /// </summary>
        private CartesianPath CreateSimplePath()
        {
            return new CartesianPath
            {
                waypoints = new System.Collections.Generic.List<CartesianWaypoint>
                {
                    new CartesianWaypoint
                    {
                        position = Vector3.zero,
                        rotation = Quaternion.identity,
                        distanceFromStart = 0f
                    },
                    new CartesianWaypoint
                    {
                        position = new Vector3(1f, 0f, 0f),
                        rotation = Quaternion.identity,
                        distanceFromStart = 1f
                    }
                },
                totalDistance = 1f,
                maxVelocity = 0.5f,
                acceleration = 1.0f
            };
        }

        /// <summary>
        /// Create a long path to ensure cruise phase exists (5 meters)
        /// </summary>
        private CartesianPath CreateLongPath()
        {
            return new CartesianPath
            {
                waypoints = new System.Collections.Generic.List<CartesianWaypoint>
                {
                    new CartesianWaypoint
                    {
                        position = Vector3.zero,
                        rotation = Quaternion.identity,
                        distanceFromStart = 0f
                    },
                    new CartesianWaypoint
                    {
                        position = new Vector3(5f, 0f, 0f),
                        rotation = Quaternion.identity,
                        distanceFromStart = 5f
                    }
                },
                totalDistance = 5f,
                maxVelocity = 0.3f,
                acceleration = 1.0f
            };
        }

        /// <summary>
        /// Create a very short path for triangular profile testing (0.1 meters)
        /// </summary>
        private CartesianPath CreateShortPath()
        {
            return new CartesianPath
            {
                waypoints = new System.Collections.Generic.List<CartesianWaypoint>
                {
                    new CartesianWaypoint
                    {
                        position = Vector3.zero,
                        rotation = Quaternion.identity,
                        distanceFromStart = 0f
                    },
                    new CartesianWaypoint
                    {
                        position = new Vector3(0.1f, 0f, 0f),
                        rotation = Quaternion.identity,
                        distanceFromStart = 0.1f
                    }
                },
                totalDistance = 0.1f,
                maxVelocity = 0.5f,
                acceleration = 1.0f
            };
        }

        #endregion
    }
}
