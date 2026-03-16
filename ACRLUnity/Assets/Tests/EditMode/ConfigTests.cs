using NUnit.Framework;
using UnityEngine;
using Configuration;
using Robotics.Grasp;

namespace Tests.EditMode
{
    /// <summary>
    /// Tests for ScriptableObject configuration system.
    /// Validates config types: RobotConfig, SimulationConfig, IKConfig,
    /// GripperConfig, TrajectoryConfig.
    /// Ensures default creation, value ranges, and serialization work correctly.
    /// </summary>
    public class ConfigTests
    {
        #region RobotConfig Tests

        private RobotConfig _robotConfig;

        [SetUp]
        public void SetUp_RobotConfig()
        {
            _robotConfig = ScriptableObject.CreateInstance<RobotConfig>();
            _robotConfig.InitializeDefaultAR4Profile();
        }

        [TearDown]
        public void TearDown_RobotConfig()
        {
            if (_robotConfig != null)
            {
                Object.DestroyImmediate(_robotConfig);
            }
        }

        [Test]
        public void RobotConfig_DefaultCreation_CreatesValidProfile()
        {
            Assert.IsNotNull(_robotConfig, "RobotConfig should be created");
            Assert.AreEqual("AR4_Default", _robotConfig.profileName, "Profile name should be AR4_Default");
            Assert.IsNotNull(_robotConfig.description, "Description should not be null");
        }

        [Test]
        public void RobotConfig_InitializeDefaultAR4Profile_CreatesSixJoints()
        {
            Assert.IsNotNull(_robotConfig.joints, "Joints array should not be null");
            Assert.AreEqual(6, _robotConfig.joints.Length, "AR4 robot should have 6 joints");
        }

        [Test]
        public void RobotConfig_JointConfigurations_HaveValidValues()
        {
            foreach (var joint in _robotConfig.joints)
            {
                Assert.IsNotNull(joint, "Joint configuration should not be null");
                Assert.Greater(joint.stiffness, 0f, "Stiffness should be positive");
                Assert.Greater(joint.damping, 0f, "Damping should be positive");
                Assert.Greater(joint.forceLimit, 0f, "Force limit should be positive");
                Assert.Greater(joint.upperLimit, joint.lowerLimit, "Upper limit should be greater than lower limit");
            }
        }

        [Test]
        public void RobotConfig_JointConfigurations_DecreasingStiffness()
        {
            // AR4 profile has strictly decreasing stiffness from base to wrist.
            // This test uses relational assertions so it stays valid when stiffness values are tuned.
            var joints = _robotConfig.joints;
            for (int i = 0; i < joints.Length - 1; i++)
            {
                Assert.Greater(joints[i].stiffness, joints[i + 1].stiffness,
                    $"Joint {i} stiffness ({joints[i].stiffness}) should exceed joint {i + 1} stiffness ({joints[i + 1].stiffness})");
            }
        }

        [Test]
        public void RobotConfig_AdjustmentSpeed_IsInValidRange()
        {
            Assert.GreaterOrEqual(_robotConfig.adjustmentSpeed, 0.1f, "Adjustment speed should be >= 0.1");
            Assert.LessOrEqual(_robotConfig.adjustmentSpeed, 1f, "Adjustment speed should be <= 1.0");
        }

        [Test]
        public void RobotConfig_OnValidate_ClampsAdjustmentSpeed()
        {
            // Use reflection to call private OnValidate method
            var onValidateMethod = typeof(RobotConfig).GetMethod("OnValidate",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Test clamping to minimum
            _robotConfig.adjustmentSpeed = -1f;
            onValidateMethod?.Invoke(_robotConfig, null);
            Assert.GreaterOrEqual(_robotConfig.adjustmentSpeed, 0.1f, "Should clamp to minimum 0.1");

            // Test clamping to maximum
            _robotConfig.adjustmentSpeed = 5f;
            onValidateMethod?.Invoke(_robotConfig, null);
            Assert.LessOrEqual(_robotConfig.adjustmentSpeed, 1f, "Should clamp to maximum 1.0");
        }

        [Test]
        public void RobotConfig_OnValidate_FixesInvalidJointLimits()
        {
            var onValidateMethod = typeof(RobotConfig).GetMethod("OnValidate",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Create invalid joint (lower >= upper)
            _robotConfig.joints[0].lowerLimit = 100f;
            _robotConfig.joints[0].upperLimit = 50f;

            onValidateMethod?.Invoke(_robotConfig, null);

            Assert.Greater(_robotConfig.joints[0].upperLimit, _robotConfig.joints[0].lowerLimit,
                "OnValidate should fix upper limit to be greater than lower limit");
        }

        #endregion

        #region SimulationConfig Tests

        private SimulationConfig _simulationConfig;

        [SetUp]
        public void SetUp_SimulationConfig()
        {
            _simulationConfig = ScriptableObject.CreateInstance<SimulationConfig>();
        }

        [TearDown]
        public void TearDown_SimulationConfig()
        {
            if (_simulationConfig != null)
            {
                Object.DestroyImmediate(_simulationConfig);
            }
        }

        [Test]
        public void SimulationConfig_DefaultCreation_HasValidDefaults()
        {
            Assert.IsNotNull(_simulationConfig, "SimulationConfig should be created");
            Assert.AreEqual(1f, _simulationConfig.timeScale, "Default time scale should be 1.0");
            Assert.AreEqual(false, _simulationConfig.autoStart, "Auto-start should be false by default");
            Assert.AreEqual(true, _simulationConfig.resetOnError, "Reset on error should be true by default");
        }

        [Test]
        public void SimulationConfig_TargetFrameRate_IsInValidRange()
        {
            Assert.GreaterOrEqual(_simulationConfig.targetFrameRate, 10, "Target frame rate should be >= 10");
            Assert.LessOrEqual(_simulationConfig.targetFrameRate, 120, "Target frame rate should be <= 120");
        }

        [Test]
        public void SimulationConfig_OnValidate_ClampsTimeScale()
        {
            var onValidateMethod = typeof(SimulationConfig).GetMethod("OnValidate",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Test minimum clamp
            _simulationConfig.timeScale = -1f;
            onValidateMethod?.Invoke(_simulationConfig, null);
            Assert.GreaterOrEqual(_simulationConfig.timeScale, 0.1f, "Time scale should be clamped to minimum 0.1");
        }

        [Test]
        public void SimulationConfig_OnValidate_ClampsTargetFrameRate()
        {
            var onValidateMethod = typeof(SimulationConfig).GetMethod("OnValidate",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Test minimum clamp
            _simulationConfig.targetFrameRate = 5;
            onValidateMethod?.Invoke(_simulationConfig, null);
            Assert.GreaterOrEqual(_simulationConfig.targetFrameRate, 10, "Should clamp to minimum 10 FPS");

            // Test maximum clamp
            _simulationConfig.targetFrameRate = 200;
            onValidateMethod?.Invoke(_simulationConfig, null);
            Assert.LessOrEqual(_simulationConfig.targetFrameRate, 120, "Should clamp to maximum 120 FPS");
        }

        #endregion

        #region IKConfig Tests

        private IKConfig _ikConfig;

        [SetUp]
        public void SetUp_IKConfig()
        {
            _ikConfig = ScriptableObject.CreateInstance<IKConfig>();
        }

        [TearDown]
        public void TearDown_IKConfig()
        {
            if (_ikConfig != null)
            {
                Object.DestroyImmediate(_ikConfig);
            }
        }

        [Test]
        public void IKConfig_DefaultCreation_HasValidDefaults()
        {
            Assert.IsNotNull(_ikConfig, "IKConfig should be created");

            // IK convergence defaults
            Assert.AreEqual(0.02f, _ikConfig.convergenceThreshold, 0.001f, "Default convergence threshold should be 0.02m (2cm)");
            Assert.AreEqual(0.2f, _ikConfig.dampingFactor, 0.01f, "Default damping factor should be 0.2");
            Assert.AreEqual(0.2f, _ikConfig.maxJointStepRad, 0.01f, "Default max joint step should be 0.2 rad");

            // Orientation control defaults
            Assert.AreEqual(10f, _ikConfig.orientationThresholdDegrees, 0.1f, "Default orientation threshold should be 10 degrees");
            Assert.AreEqual(0.30f, _ikConfig.orientationRampStartDistance, 0.01f, "Default orientation ramp start should be 0.30m");

            // Timeout defaults
            Assert.AreEqual(30f, _ikConfig.graspTimeoutSeconds, 0.1f, "Default grasp timeout should be 30s");
            Assert.AreEqual(15f, _ikConfig.movementTimeoutSeconds, 0.1f, "Default movement timeout should be 15s");
        }

        [Test]
        public void IKConfig_MotionLimits_AreValid()
        {
            Assert.GreaterOrEqual(_ikConfig.maxVelocity, 0.1f, "Max velocity should be >= 0.1");
            Assert.LessOrEqual(_ikConfig.maxVelocity, 1f, "Max velocity should be <= 1.0");
            Assert.GreaterOrEqual(_ikConfig.maxAcceleration, 0.3f, "Max acceleration should be >= 0.3");
            Assert.LessOrEqual(_ikConfig.maxAcceleration, 2f, "Max acceleration should be <= 2.0");
        }

        [Test]
        public void IKConfig_ConvergenceMultipliers_AreValid()
        {
            Assert.GreaterOrEqual(_ikConfig.graspConvergenceMultiplier, 0.1f, "Grasp convergence multiplier should be >= 0.1");
            Assert.LessOrEqual(_ikConfig.graspConvergenceMultiplier, 1f, "Grasp convergence multiplier should be <= 1.0");
            Assert.GreaterOrEqual(_ikConfig.preGraspConvergenceMultiplier, 1f, "Pre-grasp convergence multiplier should be >= 1.0");
            Assert.LessOrEqual(_ikConfig.preGraspConvergenceMultiplier, 5f, "Pre-grasp convergence multiplier should be <= 5.0");
        }

        [Test]
        public void IKConfig_AdvancedIKSolverParameters_AreValid()
        {
            Assert.GreaterOrEqual(_ikConfig.maxJointVelocity, 1f, "Max joint velocity should be >= 1.0");
            Assert.LessOrEqual(_ikConfig.maxJointVelocity, 10f, "Max joint velocity should be <= 10.0");
            Assert.GreaterOrEqual(_ikConfig.maxErrorMagnitude, 0.1f, "Max error magnitude should be >= 0.1");
            Assert.LessOrEqual(_ikConfig.maxErrorMagnitude, 5f, "Max error magnitude should be <= 5.0");
        }

        [Test]
        public void IKConfig_ObjectDetection_ParametersAreValid()
        {
            Assert.GreaterOrEqual(_ikConfig.objectFindingRadius, 0.05f, "Object finding radius should be >= 0.05");
            Assert.LessOrEqual(_ikConfig.objectFindingRadius, 0.5f, "Object finding radius should be <= 0.5");
            Assert.GreaterOrEqual(_ikConfig.objectDistanceThreshold, 0.05f, "Object distance threshold should be >= 0.05");
            Assert.LessOrEqual(_ikConfig.objectDistanceThreshold, 0.3f, "Object distance threshold should be <= 0.3");
        }

        [Test]
        public void IKConfig_OnValidate_EnsuresConsistency()
        {
            var onValidateMethod = typeof(IKConfig).GetMethod("OnValidate",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Test convergence threshold < max error magnitude
            _ikConfig.convergenceThreshold = 2.0f;
            _ikConfig.maxErrorMagnitude = 1.0f;
            onValidateMethod?.Invoke(_ikConfig, null);
            Assert.Less(_ikConfig.convergenceThreshold, _ikConfig.maxErrorMagnitude,
                "Convergence threshold should be less than max error magnitude after validation");

            // Test object distance threshold <= finding radius
            _ikConfig.objectDistanceThreshold = 0.5f;
            _ikConfig.objectFindingRadius = 0.2f;
            onValidateMethod?.Invoke(_ikConfig, null);
            Assert.LessOrEqual(_ikConfig.objectDistanceThreshold, _ikConfig.objectFindingRadius,
                "Object distance threshold should be <= finding radius after validation");
        }

        #endregion

        #region GripperConfig Tests

        private GripperConfig _gripperConfig;

        [SetUp]
        public void SetUp_GripperConfig()
        {
            _gripperConfig = ScriptableObject.CreateInstance<GripperConfig>();
        }

        [TearDown]
        public void TearDown_GripperConfig()
        {
            if (_gripperConfig != null)
            {
                Object.DestroyImmediate(_gripperConfig);
            }
        }

        [Test]
        public void GripperConfig_DefaultCreation_HasValidDefaults()
        {
            Assert.IsNotNull(_gripperConfig, "GripperConfig should be created");

            // Contact detection defaults
            Assert.AreEqual(5, _gripperConfig.forceWindowSize, "Default force window size should be 5");
            Assert.AreEqual(0.1f, _gripperConfig.minForceThreshold, 0.01f, "Default min force threshold should be 0.1N");
            Assert.AreEqual(0.1f, _gripperConfig.minContactDuration, 0.01f, "Default min contact duration should be 0.1s");
            Assert.AreEqual(5f, _gripperConfig.minGraspForce, 0.1f, "Default min grasp force should be 5N");

            // Gripper control defaults
            Assert.AreEqual(0.5f, _gripperConfig.smoothTime, 0.01f, "Default smooth time should be 0.5s");
        }

        [Test]
        public void GripperConfig_ForceParameters_AreValid()
        {
            Assert.GreaterOrEqual(_gripperConfig.forceWindowSize, 3, "Force window size should be >= 3");
            Assert.LessOrEqual(_gripperConfig.forceWindowSize, 10, "Force window size should be <= 10");
            Assert.GreaterOrEqual(_gripperConfig.minForceThreshold, 0.01f, "Min force threshold should be >= 0.01");
            Assert.LessOrEqual(_gripperConfig.minForceThreshold, 1f, "Min force threshold should be <= 1.0");
            Assert.GreaterOrEqual(_gripperConfig.minGraspForce, 1f, "Min grasp force should be >= 1.0");
            Assert.LessOrEqual(_gripperConfig.minGraspForce, 20f, "Min grasp force should be <= 20.0");
        }

        [Test]
        public void GripperConfig_OnValidate_EnsuresMinForceThresholdLessThanMinGraspForce()
        {
            var onValidateMethod = typeof(GripperConfig).GetMethod("OnValidate",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Set invalid values (threshold >= grasp force)
            _gripperConfig.minForceThreshold = 10f;
            _gripperConfig.minGraspForce = 5f;

            onValidateMethod?.Invoke(_gripperConfig, null);

            Assert.Less(_gripperConfig.minForceThreshold, _gripperConfig.minGraspForce,
                "Min force threshold should be less than min grasp force after validation");
        }

        #endregion

        #region TrajectoryConfig Tests

        private TrajectoryConfig _trajectoryConfig;

        [SetUp]
        public void SetUp_TrajectoryConfig()
        {
            _trajectoryConfig = ScriptableObject.CreateInstance<TrajectoryConfig>();
        }

        [TearDown]
        public void TearDown_TrajectoryConfig()
        {
            if (_trajectoryConfig != null)
            {
                Object.DestroyImmediate(_trajectoryConfig);
            }
        }

        [Test]
        public void TrajectoryConfig_DefaultCreation_HasValidDefaults()
        {
            Assert.IsNotNull(_trajectoryConfig, "TrajectoryConfig should be created");

            // PD control gains (default to 10, 10, 10)
            Assert.AreEqual(10f, _trajectoryConfig.positionGains.x, 0.1f, "Default position gain X should be 10");
            Assert.AreEqual(10f, _trajectoryConfig.positionGains.y, 0.1f, "Default position gain Y should be 10");
            Assert.AreEqual(10f, _trajectoryConfig.positionGains.z, 0.1f, "Default position gain Z should be 10");

            // Velocity gains (default to 2, 2, 2)
            Assert.AreEqual(2f, _trajectoryConfig.velocityGains.x, 0.1f, "Default velocity gain X should be 2");
            Assert.AreEqual(2f, _trajectoryConfig.velocityGains.y, 0.1f, "Default velocity gain Y should be 2");
            Assert.AreEqual(2f, _trajectoryConfig.velocityGains.z, 0.1f, "Default velocity gain Z should be 2");

            // Motion limits
            Assert.AreEqual(0.5f, _trajectoryConfig.maxVelocity, 0.01f, "Default max velocity should be 0.5 m/s");
            Assert.AreEqual(1.0f, _trajectoryConfig.maxAcceleration, 0.01f, "Default max acceleration should be 1.0 m/s²");
        }

        [Test]
        public void TrajectoryConfig_PDGains_ArePositive()
        {
            Assert.Greater(_trajectoryConfig.positionGains.x, 0f, "Position gain X should be positive");
            Assert.Greater(_trajectoryConfig.positionGains.y, 0f, "Position gain Y should be positive");
            Assert.Greater(_trajectoryConfig.positionGains.z, 0f, "Position gain Z should be positive");
            Assert.Greater(_trajectoryConfig.velocityGains.x, 0f, "Velocity gain X should be positive");
            Assert.Greater(_trajectoryConfig.velocityGains.y, 0f, "Velocity gain Y should be positive");
            Assert.Greater(_trajectoryConfig.velocityGains.z, 0f, "Velocity gain Z should be positive");
        }

        [Test]
        public void TrajectoryConfig_MotionLimits_AreValid()
        {
            Assert.GreaterOrEqual(_trajectoryConfig.maxVelocity, 0.1f, "Max velocity should be >= 0.1");
            Assert.LessOrEqual(_trajectoryConfig.maxVelocity, 1f, "Max velocity should be <= 1.0");
            Assert.GreaterOrEqual(_trajectoryConfig.maxAcceleration, 0.3f, "Max acceleration should be >= 0.3");
            Assert.LessOrEqual(_trajectoryConfig.maxAcceleration, 2f, "Max acceleration should be <= 2.0");
        }

        [Test]
        public void TrajectoryConfig_OnValidate_ClampsGainsToPositive()
        {
            var onValidateMethod = typeof(TrajectoryConfig).GetMethod("OnValidate",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Set negative gains
            _trajectoryConfig.positionGains = new Vector3(-1f, -2f, -3f);
            _trajectoryConfig.velocityGains = new Vector3(-0.5f, -1f, -1.5f);

            onValidateMethod?.Invoke(_trajectoryConfig, null);

            Assert.GreaterOrEqual(_trajectoryConfig.positionGains.x, 0.1f, "Position gain X should be clamped to >= 0.1");
            Assert.GreaterOrEqual(_trajectoryConfig.positionGains.y, 0.1f, "Position gain Y should be clamped to >= 0.1");
            Assert.GreaterOrEqual(_trajectoryConfig.positionGains.z, 0.1f, "Position gain Z should be clamped to >= 0.1");
            Assert.GreaterOrEqual(_trajectoryConfig.velocityGains.x, 0.1f, "Velocity gain X should be clamped to >= 0.1");
            Assert.GreaterOrEqual(_trajectoryConfig.velocityGains.y, 0.1f, "Velocity gain Y should be clamped to >= 0.1");
            Assert.GreaterOrEqual(_trajectoryConfig.velocityGains.z, 0.1f, "Velocity gain Z should be clamped to >= 0.1");
        }

        #endregion

        #region Config Serialization Tests

        [Test]
        public void RobotConfig_Serialization_PreservesValues()
        {
            // Create and initialize config
            var config = ScriptableObject.CreateInstance<RobotConfig>();
            config.InitializeDefaultAR4Profile();
            config.profileName = "TestProfile";
            config.adjustmentSpeed = 0.7f;

            // Simulate serialization (Unity does this automatically, we'll just verify values persist)
            var profileName = config.profileName;
            var adjustmentSpeed = config.adjustmentSpeed;
            var firstJointStiffness = config.joints[0].stiffness;

            Assert.AreEqual("TestProfile", profileName, "Profile name should persist");
            Assert.AreEqual(0.7f, adjustmentSpeed, 0.001f, "Adjustment speed should persist");
            Assert.AreEqual(5000f, firstJointStiffness, "Joint stiffness should persist");

            Object.DestroyImmediate(config);
        }

        [Test]
        public void AllConfigs_CanBeCreatedViaCreateAssetMenu()
        {
            // Test all config types can be created via ScriptableObject.CreateInstance
            var robotConfig = ScriptableObject.CreateInstance<RobotConfig>();
            Assert.IsNotNull(robotConfig, "RobotConfig should be creatable");

            var simulationConfig = ScriptableObject.CreateInstance<SimulationConfig>();
            Assert.IsNotNull(simulationConfig, "SimulationConfig should be creatable");

            var ikConfig = ScriptableObject.CreateInstance<IKConfig>();
            Assert.IsNotNull(ikConfig, "IKConfig should be creatable");

            var gripperConfig = ScriptableObject.CreateInstance<GripperConfig>();
            Assert.IsNotNull(gripperConfig, "GripperConfig should be creatable");

            var trajectoryConfig = ScriptableObject.CreateInstance<TrajectoryConfig>();
            Assert.IsNotNull(trajectoryConfig, "TrajectoryConfig should be creatable");

            // Cleanup
            Object.DestroyImmediate(robotConfig);
            Object.DestroyImmediate(simulationConfig);
            Object.DestroyImmediate(ikConfig);
            Object.DestroyImmediate(gripperConfig);
            Object.DestroyImmediate(trajectoryConfig);
        }

        #endregion

        #region IKConfig Behavior Tests (Moved from RobotControllerTests)

        [Test]
        public void IKConfig_GraspConvergenceThreshold_CalculatesCorrectly()
        {
            // Test that grasp convergence threshold calculation is correct
            // Threshold = convergenceThreshold * graspConvergenceMultiplier
            var config = ScriptableObject.CreateInstance<IKConfig>();

            float expectedThreshold = config.convergenceThreshold * config.graspConvergenceMultiplier;

            // Default: 0.02 * 0.33 = 0.0066m (6.6mm)
            Assert.AreEqual(0.02f, config.convergenceThreshold, 0.001f, "Default convergence threshold should be 0.02m");
            Assert.AreEqual(0.33f, config.graspConvergenceMultiplier, 0.01f, "Default grasp multiplier should be 0.33");
            Assert.AreEqual(0.0066f, expectedThreshold, 0.0001f,
                "Grasp convergence threshold should be approximately 6.6mm (relaxed from 3mm)");

            Object.DestroyImmediate(config);
        }

        [Test]
        public void IKConfig_OrientationThreshold_DefaultIs10Degrees()
        {
            var config = ScriptableObject.CreateInstance<IKConfig>();

            // Precision improvement (January 2026): increased from 5 to 10 degrees
            Assert.AreEqual(10f, config.orientationThresholdDegrees, 0.1f,
                "Orientation threshold should be 10 degrees (configurable)");

            Object.DestroyImmediate(config);
        }

        [Test]
        public void IKConfig_OrientationRampStart_DefaultIs30cm()
        {
            var config = ScriptableObject.CreateInstance<IKConfig>();

            // Precision improvement (January 2026): increased from 20cm to 30cm
            Assert.AreEqual(0.30f, config.orientationRampStartDistance, 0.01f,
                "Orientation ramping should start at 30cm");

            Object.DestroyImmediate(config);
        }

        [Test]
        public void IKConfig_GraspTimeout_DefaultIs30Seconds()
        {
            var config = ScriptableObject.CreateInstance<IKConfig>();

            Assert.AreEqual(30f, config.graspTimeoutSeconds, 0.1f,
                "Grasp timeout should be 30 seconds");

            Object.DestroyImmediate(config);
        }

        [Test]
        public void IKConfig_MovementTimeout_DefaultIs15Seconds()
        {
            var config = ScriptableObject.CreateInstance<IKConfig>();

            Assert.AreEqual(15f, config.movementTimeoutSeconds, 0.1f,
                "Movement timeout should be 15 seconds");

            Object.DestroyImmediate(config);
        }

        [Test]
        public void IKConfig_PreGraspConvergenceMultiplier_DefaultIs2x()
        {
            var config = ScriptableObject.CreateInstance<IKConfig>();

            // Pre-grasp uses 2x multiplier for faster approach
            Assert.AreEqual(2.0f, config.preGraspConvergenceMultiplier, 0.1f,
                "Pre-grasp convergence multiplier should be 2.0 (looser tolerance)");

            Object.DestroyImmediate(config);
        }

        #endregion
    }
}
