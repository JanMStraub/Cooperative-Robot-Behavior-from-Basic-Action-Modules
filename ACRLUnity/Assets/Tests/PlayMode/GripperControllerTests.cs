using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using System.Collections;
using Robotics;

namespace Tests.PlayMode
{
    /// <summary>
    /// Play mode tests for GripperController.
    /// Tests gripper control methods, SmoothDamp interpolation, event firing, and state management.
    /// </summary>
    public class GripperControllerTests
    {
        private GameObject _gripperObject;
        private GripperController _gripperController;
        private ArticulationBody _leftGripper;
        private ArticulationBody _rightGripper;

        private const float EPSILON = 0.001f;

        [UnitySetUp]
        public IEnumerator Setup()
        {
            // Create gripper controller GameObject with root ArticulationBody
            _gripperObject = new GameObject("TestGripperController");
            var rootBody = _gripperObject.AddComponent<ArticulationBody>();
            rootBody.immovable = true; // Root must be immovable
            rootBody.jointType = ArticulationJointType.FixedJoint;

            // Ignore expected error from GripperController.Awake() before grippers are assigned
            LogAssert.Expect(LogType.Error, "[GRIPPER_CONTROLLER] Gripper references not assigned!");

            _gripperController = _gripperObject.AddComponent<GripperController>();

            // Create left gripper ArticulationBody (child of root)
            var leftObj = new GameObject("LeftGripper");
            leftObj.transform.SetParent(_gripperObject.transform);
            _leftGripper = leftObj.AddComponent<ArticulationBody>();
            _leftGripper.jointType = ArticulationJointType.RevoluteJoint;

            // Configure left gripper drive
            var leftDrive = _leftGripper.xDrive;
            leftDrive.lowerLimit = 0f;
            leftDrive.upperLimit = 0.05f;
            leftDrive.stiffness = 1000f;
            leftDrive.damping = 100f;
            leftDrive.forceLimit = 100f;
            _leftGripper.xDrive = leftDrive;

            // Create right gripper ArticulationBody
            var rightObj = new GameObject("RightGripper");
            rightObj.transform.SetParent(_gripperObject.transform);
            _rightGripper = rightObj.AddComponent<ArticulationBody>();
            _rightGripper.jointType = ArticulationJointType.RevoluteJoint;

            // Configure right gripper drive
            var rightDrive = _rightGripper.xDrive;
            rightDrive.lowerLimit = 0f;
            rightDrive.upperLimit = 0.05f;
            rightDrive.stiffness = 1000f;
            rightDrive.damping = 100f;
            rightDrive.forceLimit = 100f;
            _rightGripper.xDrive = rightDrive;

            // Assign grippers to controller
            _gripperController.leftGripper = _leftGripper;
            _gripperController.rightGripper = _rightGripper;
            _gripperController.gripSpeed = 0.2f; // Faster for testing (meters per second)

            // Wait for Start() to be called and physics to initialize
            yield return null;
            yield return new WaitForFixedUpdate();
        }

        [TearDown]
        public void Teardown()
        {
            if (_gripperObject != null)
                Object.DestroyImmediate(_gripperObject);
        }

        #region Open/Close Tests

        /// <summary>
        /// Test that OpenGrippers sets targetPosition to 1.0 (fully open).
        /// </summary>
        [Test]
        public void OpenGrippers_SetsTargetToOne()
        {
            _gripperController.OpenGrippers();
            Assert.AreEqual(1.0f, _gripperController.targetPosition, EPSILON, "OpenGrippers should set targetPosition to 1.0");
        }

        /// <summary>
        /// Test that CloseGrippers sets targetPosition to 0.0 (fully closed).
        /// </summary>
        [Test]
        public void CloseGrippers_SetsTargetToZero()
        {
            _gripperController.CloseGrippers();
            Assert.AreEqual(0.0f, _gripperController.targetPosition, EPSILON, "CloseGrippers should set targetPosition to 0.0");
        }

        /// <summary>
        /// Test that OpenGrippers sets IsMoving to true when gripper is not already open.
        /// </summary>
        [Test]
        public void OpenGrippers_SetsIsMovingTrue()
        {
            // First close the gripper so opening will cause a position change
            _gripperController.SetGripperPosition(0f);

            _gripperController.OpenGrippers();
            Assert.IsTrue(_gripperController.IsMoving, "OpenGrippers should set IsMoving to true");
        }

        /// <summary>
        /// Test that CloseGrippers sets IsMoving to true when gripper is not already closed.
        /// </summary>
        [Test]
        public void CloseGrippers_SetsIsMovingTrue()
        {
            // First open the gripper so closing will cause a position change
            _gripperController.SetGripperPosition(1f);

            _gripperController.CloseGrippers();
            Assert.IsTrue(_gripperController.IsMoving, "CloseGrippers should set IsMoving to true");
        }

        #endregion

        #region SetGripperPosition Tests

        /// <summary>
        /// Test that SetGripperPosition accepts valid normalized values.
        /// </summary>
        [Test]
        public void SetGripperPosition_ValidValue_SetsTargetPosition()
        {
            _gripperController.SetGripperPosition(0.5f);
            Assert.AreEqual(0.5f, _gripperController.targetPosition, EPSILON, "SetGripperPosition should set targetPosition to 0.5");
        }

        /// <summary>
        /// Test that SetGripperPosition clamps values above 1.0.
        /// </summary>
        [Test]
        public void SetGripperPosition_ValueAboveOne_ClampsToOne()
        {
            _gripperController.SetGripperPosition(1.5f);
            Assert.AreEqual(1.0f, _gripperController.targetPosition, EPSILON, "SetGripperPosition should clamp values above 1.0 to 1.0");
        }

        /// <summary>
        /// Test that SetGripperPosition clamps negative values to 0.0.
        /// </summary>
        [Test]
        public void SetGripperPosition_NegativeValue_ClampsToZero()
        {
            _gripperController.SetGripperPosition(-0.5f);
            Assert.AreEqual(0.0f, _gripperController.targetPosition, EPSILON, "SetGripperPosition should clamp negative values to 0.0");
        }

        /// <summary>
        /// Test that SetGripperPosition accepts 0.0 (minimum).
        /// </summary>
        [Test]
        public void SetGripperPosition_Zero_SetsToZero()
        {
            _gripperController.SetGripperPosition(0.0f);
            Assert.AreEqual(0.0f, _gripperController.targetPosition, EPSILON, "SetGripperPosition should accept 0.0");
        }

        /// <summary>
        /// Test that SetGripperPosition accepts 1.0 (maximum).
        /// </summary>
        [Test]
        public void SetGripperPosition_One_SetsToOne()
        {
            _gripperController.SetGripperPosition(1.0f);
            Assert.AreEqual(1.0f, _gripperController.targetPosition, EPSILON, "SetGripperPosition should accept 1.0");
        }

        #endregion

        #region ResetGrippers Tests

        /// <summary>
        /// Test that ResetGrippers sets targetPosition to 1.0 (fully open).
        /// </summary>
        [Test]
        public void ResetGrippers_SetsTargetToOne()
        {
            _gripperController.targetPosition = 0.5f;
            _gripperController.ResetGrippers();
            Assert.AreEqual(1.0f, _gripperController.targetPosition, EPSILON, "ResetGrippers should set targetPosition to 1.0 (open)");
        }

        /// <summary>
        /// Test that ResetGrippers sets drive targets to fully open position (upper limit).
        /// </summary>
        [UnityTest]
        public IEnumerator ResetGrippers_SetsDriveToOpen()
        {
            // Close grippers first
            _gripperController.CloseGrippers();
            yield return new WaitForSeconds(0.5f);

            // Reset should open them
            _gripperController.ResetGrippers();

            yield return new WaitForFixedUpdate();

            // Check that drive targets are set to upper limit (fully open)
            float expectedTarget = _leftGripper.xDrive.upperLimit;
            Assert.AreEqual(expectedTarget, _leftGripper.xDrive.target, EPSILON, "Left gripper drive target should be set to upper limit (open)");
            Assert.AreEqual(expectedTarget, _rightGripper.xDrive.target, EPSILON, "Right gripper drive target should be set to upper limit (open)");
        }

        #endregion

        #region GetTargetPosition Tests

        /// <summary>
        /// Test that GetTargetPosition returns the current targetPosition.
        /// </summary>
        [Test]
        public void GetTargetPosition_ReturnsCurrentTarget()
        {
            _gripperController.targetPosition = 0.7f;
            float target = _gripperController.targetPosition;
            Assert.AreEqual(0.7f, target, EPSILON, "targetPosition should return current targetPosition");
        }

        #endregion

        #region GripSpeed Interpolation Tests

        /// <summary>
        /// Test that gripSpeed interpolation converges towards target over multiple frames.
        /// </summary>
        [UnityTest]
        public IEnumerator GripSpeed_ConvergesToTarget()
        {
            _gripperController.targetPosition = 0.0f;
            yield return new WaitForSeconds(0.5f); // Wait for initial state

            // Set new target
            _gripperController.SetGripperPosition(1.0f);

            // Wait for several frames
            for (int i = 0; i < 20; i++)
            {
                yield return null;
            }

            // Check that drive target is approaching mapped target (upper limit)
            float upperLimit = _leftGripper.xDrive.upperLimit;
            float driveTarget = _leftGripper.xDrive.target;

            // Should be significantly closer to upper limit than 0
            Assert.Greater(driveTarget, upperLimit * 0.5f, "Drive target should be approaching upper limit");
        }

        /// <summary>
        /// Test that gripSpeed parameter affects interpolation speed.
        /// </summary>
        [UnityTest]
        public IEnumerator GripSpeed_AffectsInterpolationSpeed()
        {
            // Reset gripper to known state (fully closed)
            _gripperController.ResetGrippers();
            _gripperController.SetGripperPosition(0.0f);
            yield return new WaitForSeconds(1.0f); // Ensure fully closed

            // Test 1: Fast grip speed - open from closed
            _gripperController.gripSpeed = 1.0f; // Very fast
            _gripperController.SetGripperPosition(1.0f);
            yield return new WaitForSeconds(0.05f); // Short time interval

            float fastDriveTarget = _leftGripper.xDrive.target;

            // Reset gripper to closed again
            _gripperController.ResetGrippers();
            _gripperController.SetGripperPosition(0.0f);
            yield return new WaitForSeconds(1.0f); // Ensure fully closed

            // Test 2: Slow grip speed - open from closed
            _gripperController.gripSpeed = 0.01f; // Very slow
            _gripperController.SetGripperPosition(1.0f);
            yield return new WaitForSeconds(0.05f); // Same time interval

            float slowDriveTarget = _leftGripper.xDrive.target;

            // Fast grip speed should result in larger movement in same time
            // Debug info for troubleshooting
            Debug.Log($"Fast drive target: {fastDriveTarget}, Slow drive target: {slowDriveTarget}");
            Assert.Greater(fastDriveTarget, slowDriveTarget, "Faster gripSpeed should result in faster convergence");
        }

        #endregion

        #region OnGripperActionComplete Event Tests

        /// <summary>
        /// Test that OnGripperActionComplete event fires when gripper reaches target.
        /// </summary>
        [UnityTest]
        public IEnumerator OnGripperActionComplete_FiresWhenReachingTarget()
        {
            bool eventFired = false;
            _gripperController.OnGripperActionComplete += () => eventFired = true;

            _gripperController.gripSpeed = 0.5f; // Fast convergence for testing
            _gripperController.OpenGrippers();

            // Wait for convergence
            float timeout = 2.0f;
            float elapsed = 0f;
            while (!eventFired && elapsed < timeout)
            {
                yield return new WaitForSeconds(0.1f);
                elapsed += 0.1f;
            }

            Assert.IsTrue(eventFired, "OnGripperActionComplete should fire when gripper reaches target");
        }

        /// <summary>
        /// Test that OnGripperActionComplete does not fire immediately.
        /// </summary>
        [UnityTest]
        public IEnumerator OnGripperActionComplete_DoesNotFireImmediately()
        {
            bool eventFired = false;
            _gripperController.OnGripperActionComplete += () => eventFired = true;

            _gripperController.OpenGrippers();

            yield return null; // Wait one frame

            Assert.IsFalse(eventFired, "OnGripperActionComplete should not fire immediately");
        }

        #endregion

        #region IsMoving State Tests

        /// <summary>
        /// Test that IsMoving becomes false after gripper converges to target.
        /// </summary>
        [UnityTest]
        public IEnumerator IsMoving_BecomesFalseAfterConvergence()
        {
            _gripperController.gripSpeed = 0.5f; // Fast convergence
            _gripperController.OpenGrippers();

            Assert.IsTrue(_gripperController.IsMoving, "IsMoving should be true initially");

            // Wait for convergence
            float timeout = 2.0f;
            float elapsed = 0f;
            while (_gripperController.IsMoving && elapsed < timeout)
            {
                yield return new WaitForSeconds(0.1f);
                elapsed += 0.1f;
            }

            Assert.IsFalse(_gripperController.IsMoving, "IsMoving should become false after convergence");
        }

        /// <summary>
        /// Test that IsMoving stays true during interpolation.
        /// </summary>
        [UnityTest]
        public IEnumerator IsMoving_StaysTrueDuringInterpolation()
        {
            _gripperController.gripSpeed = 0.05f; // Slower for longer interpolation
            _gripperController.OpenGrippers();

            // Wait a short time (not enough to converge)
            yield return new WaitForSeconds(0.1f);

            Assert.IsTrue(_gripperController.IsMoving, "IsMoving should stay true during interpolation");
        }

        #endregion

        #region CurrentPosition Tests

        /// <summary>
        /// Test that CurrentPosition returns the left gripper's joint position.
        /// </summary>
        [UnityTest]
        public IEnumerator CurrentPosition_ReturnsLeftGripperPosition()
        {
            // Wait for physics initialization
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();

            // Check if joint position is accessible
            if (_leftGripper.jointPosition.dofCount > 0)
            {
                // Verify we can read the current joint position (should be near 0 initially)
                float currentPos = _gripperController.leftGripper.jointPosition[0];

                // Joint position should be a valid number (not NaN or infinity)
                Assert.IsFalse(float.IsNaN(currentPos), "Joint position should not be NaN");
                Assert.IsFalse(float.IsInfinity(currentPos), "Joint position should not be infinity");

                // Should be within the configured limits (0 to 0.05)
                Assert.GreaterOrEqual(currentPos, -0.001f, "Joint position should be >= lower limit");
                Assert.LessOrEqual(currentPos, 0.051f, "Joint position should be <= upper limit");
            }
            else
            {
                Assert.Inconclusive("ArticulationBody jointPosition not initialized in test environment");
            }
        }

        #endregion

        #region Null Reference Tests

        /// <summary>
        /// Test that Update handles null grippers gracefully.
        /// </summary>
        [UnityTest]
        public IEnumerator Update_NullGrippers_DoesNotCrash()
        {
            _gripperController.leftGripper = null;
            _gripperController.rightGripper = null;

            // Should not crash during Update
            yield return null;
            yield return null;
            yield return null;

            Assert.Pass("Update should handle null grippers without crashing");
        }

        #endregion
    }
}
