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

        [SetUp]
        public void Setup()
        {
            // Create gripper controller GameObject
            _gripperObject = new GameObject("TestGripperController");

            // Expect error from GripperController.Awake before references are assigned
            LogAssert.Expect(LogType.Error, "[GRIPPER_CONTROLLER] Gripper references not assigned!");

            _gripperController = _gripperObject.AddComponent<GripperController>();

            // Create left gripper ArticulationBody
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
            _gripperController.smoothTime = 0.1f; // Faster for testing
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
        /// Test that OpenGrippers sets IsMoving to true.
        /// </summary>
        [Test]
        public void OpenGrippers_SetsIsMovingTrue()
        {
            _gripperController.OpenGrippers();
            Assert.IsTrue(_gripperController.IsMoving, "OpenGrippers should set IsMoving to true");
        }

        /// <summary>
        /// Test that CloseGrippers sets IsMoving to true.
        /// </summary>
        [Test]
        public void CloseGrippers_SetsIsMovingTrue()
        {
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
        /// Test that ResetGrippers sets targetPosition to 0.0.
        /// </summary>
        [Test]
        public void ResetGrippers_SetsTargetToZero()
        {
            _gripperController.targetPosition = 0.8f;
            _gripperController.ResetGrippers();
            Assert.AreEqual(0.0f, _gripperController.targetPosition, EPSILON, "ResetGrippers should set targetPosition to 0.0");
        }

        /// <summary>
        /// Test that ResetGrippers clears joint positions.
        /// </summary>
        [UnityTest]
        public IEnumerator ResetGrippers_ClearsJointState()
        {
            // Set non-zero joint state
            _leftGripper.jointPosition = new ArticulationReducedSpace(0.02f);
            _rightGripper.jointPosition = new ArticulationReducedSpace(0.02f);

            yield return new WaitForFixedUpdate();

            _gripperController.ResetGrippers();

            yield return new WaitForFixedUpdate();

            // Check that joint positions are reset (drive targets should be 0)
            Assert.AreEqual(0.0f, _leftGripper.xDrive.target, EPSILON, "Left gripper drive target should be reset to 0");
            Assert.AreEqual(0.0f, _rightGripper.xDrive.target, EPSILON, "Right gripper drive target should be reset to 0");
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
            float target = _gripperController.GetTargetPosition();
            Assert.AreEqual(0.7f, target, EPSILON, "GetTargetPosition should return current targetPosition");
        }

        #endregion

        #region SmoothDamp Interpolation Tests

        /// <summary>
        /// Test that SmoothDamp interpolation converges towards target over multiple frames.
        /// </summary>
        [UnityTest]
        public IEnumerator SmoothDamp_ConvergesToTarget()
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
        /// Test that SmoothDamp respects smoothTime parameter for interpolation speed.
        /// </summary>
        [UnityTest]
        public IEnumerator SmoothDamp_RespectsSmootTime()
        {
            // Set fast smooth time
            _gripperController.smoothTime = 0.05f;
            _gripperController.targetPosition = 0.0f;
            yield return new WaitForSeconds(0.2f);

            _gripperController.SetGripperPosition(1.0f);
            yield return new WaitForSeconds(0.15f); // Wait 3x smoothTime

            float fastDriveTarget = _leftGripper.xDrive.target;

            // Reset and test with slow smooth time
            _gripperController.ResetGrippers();
            _gripperController.smoothTime = 0.5f; // Much slower
            yield return new WaitForSeconds(0.2f);

            _gripperController.SetGripperPosition(1.0f);
            yield return new WaitForSeconds(0.15f); // Same wait time

            float slowDriveTarget = _leftGripper.xDrive.target;

            // Fast smooth time should result in larger movement
            Assert.Greater(fastDriveTarget, slowDriveTarget, "Faster smoothTime should result in faster convergence");
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

            _gripperController.smoothTime = 0.05f; // Fast convergence for testing
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
            _gripperController.smoothTime = 0.05f; // Fast convergence
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
            _gripperController.smoothTime = 0.5f; // Slower for longer interpolation
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
            // Set known joint position
            _leftGripper.jointPosition = new ArticulationReducedSpace(0.025f);

            yield return new WaitForFixedUpdate();

            float currentPos = _gripperController.CurrentPosition;
            Assert.AreEqual(0.025f, currentPos, EPSILON, "CurrentPosition should return left gripper joint position");
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

        #region Gripper Geometry Tests

        /// <summary>
        /// Test that Geometry property returns the configured gripper geometry.
        /// </summary>
        [Test]
        public void Geometry_ReturnsConfiguredGeometry()
        {
            var geometry = _gripperController.Geometry;
            Assert.IsNotNull(geometry, "Geometry should not be null");
        }

        #endregion
    }
}
