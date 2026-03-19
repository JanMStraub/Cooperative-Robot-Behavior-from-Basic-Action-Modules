using NUnit.Framework;
using Core;

namespace Tests.EditMode
{
    /// <summary>
    /// Tests for Constants classes.
    /// Validates constant values are correctly defined and within expected ranges.
    /// NOTE: Many constants have been moved to ScriptableObject configs (IKConfig, GripperConfig, TrajectoryConfig)
    /// for runtime tuning. This test file now only validates infrastructure constants.
    /// </summary>
    public class ConstantsTests
    {
        #region RobotConstants Tests

        [Test]
        public void RobotConstants_JacobianDimensions_IsCorrect()
        {
            Assert.AreEqual(6, RobotConstants.JACOBIAN_DIMENSIONS);
        }

        [Test]
        public void RobotConstants_MovementThreshold_IsSmall()
        {
            // 1cm threshold
            Assert.AreEqual(0.01f, RobotConstants.MOVEMENT_THRESHOLD);
        }

        #endregion

        #region SceneConstants Tests

        [Test]
        public void SceneConstants_SizeThresholds_AreOrdered()
        {
            Assert.Less(SceneConstants.SMALL_OBJECT_SIZE_THRESHOLD,
                SceneConstants.GRASPABLE_OBJECT_SIZE_THRESHOLD);
        }

        [Test]
        public void SceneConstants_Thresholds_ArePositive()
        {
            Assert.Greater(SceneConstants.SMALL_OBJECT_SIZE_THRESHOLD, 0f);
            Assert.Greater(SceneConstants.GRASPABLE_OBJECT_SIZE_THRESHOLD, 0f);
        }

        #endregion

        #region CameraConstants Tests

        [Test]
        public void CameraConstants_Thresholds_ArePositive()
        {
            Assert.Greater(CameraConstants.TARGET_DISTANCE_THRESHOLD, 0f);
            Assert.Greater(CameraConstants.POSITION_REACHED_THRESHOLD, 0f);
        }

        #endregion

        #region LoggingConstants Tests

        [Test]
        public void LoggingConstants_SampleRates_ArePositive()
        {
            Assert.Greater(LoggingConstants.DEFAULT_ENVIRONMENT_SAMPLE_RATE, 0f);
            Assert.Greater(LoggingConstants.DEFAULT_TRAJECTORY_SAMPLE_RATE, 0f);
        }

        [Test]
        public void LoggingConstants_TrajectorySampleRate_IsFasterThanEnvironment()
        {
            // Trajectory should sample more frequently than environment snapshots
            Assert.Less(LoggingConstants.DEFAULT_TRAJECTORY_SAMPLE_RATE,
                LoggingConstants.DEFAULT_ENVIRONMENT_SAMPLE_RATE);
        }

        #endregion

        #region CollisionConstants Tests

        [Test]
        public void CollisionConstants_Cooldown_IsPositive()
        {
            Assert.Greater(CollisionConstants.DEFAULT_COLLISION_COOLDOWN, 0f);
        }

        [Test]
        public void CollisionConstants_TargetReward_IsPositive()
        {
            Assert.Greater(CollisionConstants.DEFAULT_TARGET_REWARD, 0f);
        }

        #endregion

        #region CommunicationConstants Tests

        [Test]
        public void CommunicationConstants_Ports_AreValid()
        {
            // Valid port range is 1-65535
            Assert.Greater(CommunicationConstants.COMMAND_SERVER_PORT, 0);
            Assert.LessOrEqual(CommunicationConstants.COMMAND_SERVER_PORT, 65535);

            Assert.Greater(CommunicationConstants.SEQUENCE_SERVER_PORT, 0);
            Assert.LessOrEqual(CommunicationConstants.SEQUENCE_SERVER_PORT, 65535);
        }

        [Test]
        public void CommunicationConstants_Ports_AreUnique()
        {
            var ports = new[]
            {
                CommunicationConstants.COMMAND_SERVER_PORT,
                CommunicationConstants.SEQUENCE_SERVER_PORT
            };

            // Check all ports are unique
            var uniquePorts = new System.Collections.Generic.HashSet<int>(ports);
            Assert.AreEqual(ports.Length, uniquePorts.Count, "Some ports are duplicated");
        }

        [Test]
        public void CommunicationConstants_MaxJsonLength_Matches_ProtocolLimit()
        {
            // Should match UnityProtocol.MAX_IMAGE_SIZE
            Assert.AreEqual(10 * 1024 * 1024, CommunicationConstants.MAX_JSON_LENGTH);
        }

        [Test]
        public void CommunicationConstants_ReconnectInterval_IsPositive()
        {
            Assert.Greater(CommunicationConstants.RECONNECT_INTERVAL, 0f);
        }

        #endregion
    }
}
