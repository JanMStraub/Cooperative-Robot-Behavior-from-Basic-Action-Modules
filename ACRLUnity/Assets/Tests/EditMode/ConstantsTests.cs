using NUnit.Framework;
using Core;

namespace Tests.EditMode
{
    /// <summary>
    /// Tests for Constants classes.
    /// Validates constant values are correctly defined and within expected ranges.
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
        public void RobotConstants_DampingFactor_IsPositive()
        {
            Assert.Greater(RobotConstants.DEFAULT_DAMPING_FACTOR, 0f);
            Assert.LessOrEqual(RobotConstants.DEFAULT_DAMPING_FACTOR, 1f);
        }

        [Test]
        public void RobotConstants_ConvergenceThreshold_IsReasonable()
        {
            // Should be small but not too small (10cm is reasonable)
            Assert.Greater(RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD, 0f);
            Assert.LessOrEqual(RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD, 1f);
        }

        [Test]
        public void RobotConstants_MaxJointStep_IsReasonable()
        {
            // Should be positive and less than 1 radian (~57 degrees)
            Assert.Greater(RobotConstants.DEFAULT_MAX_JOINT_STEP_RAD, 0f);
            Assert.Less(RobotConstants.DEFAULT_MAX_JOINT_STEP_RAD, 1f);
        }

        [Test]
        public void RobotConstants_StepSpeeds_AreOrdered()
        {
            Assert.LessOrEqual(RobotConstants.MIN_STEP_SPEED_NEAR_TARGET, RobotConstants.MAX_STEP_SPEED);
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
            Assert.Greater(CommunicationConstants.LLM_RESULTS_PORT, 0);
            Assert.LessOrEqual(CommunicationConstants.LLM_RESULTS_PORT, 65535);

            Assert.Greater(CommunicationConstants.RAG_SERVER_PORT, 0);
            Assert.LessOrEqual(CommunicationConstants.RAG_SERVER_PORT, 65535);

            Assert.Greater(CommunicationConstants.STATUS_SERVER_PORT, 0);
            Assert.LessOrEqual(CommunicationConstants.STATUS_SERVER_PORT, 65535);

            Assert.Greater(CommunicationConstants.SEQUENCE_SERVER_PORT, 0);
            Assert.LessOrEqual(CommunicationConstants.SEQUENCE_SERVER_PORT, 65535);
        }

        [Test]
        public void CommunicationConstants_Ports_AreUnique()
        {
            var ports = new[]
            {
                CommunicationConstants.LLM_RESULTS_PORT,
                CommunicationConstants.RAG_SERVER_PORT,
                CommunicationConstants.STATUS_SERVER_PORT,
                CommunicationConstants.SEQUENCE_SERVER_PORT
            };

            // Check all ports are unique
            var uniquePorts = new System.Collections.Generic.HashSet<int>(ports);
            Assert.AreEqual(ports.Length, uniquePorts.Count, "Some ports are duplicated");
        }

        [Test]
        public void CommunicationConstants_Ports_AreInExpectedRange()
        {
            // All ports should be in the 5000s range for this project
            Assert.AreEqual(5010, CommunicationConstants.LLM_RESULTS_PORT);
            Assert.AreEqual(5011, CommunicationConstants.RAG_SERVER_PORT);
            Assert.AreEqual(5012, CommunicationConstants.STATUS_SERVER_PORT);
            Assert.AreEqual(5013, CommunicationConstants.SEQUENCE_SERVER_PORT);
        }

        [Test]
        public void CommunicationConstants_PythonTimeout_IsReasonable()
        {
            // 5 minutes default timeout
            Assert.AreEqual(300, CommunicationConstants.DEFAULT_PYTHON_TIMEOUT);
        }

        [Test]
        public void CommunicationConstants_ProcessLimits_AreReasonable()
        {
            Assert.Greater(CommunicationConstants.MAX_CONCURRENT_PROCESSES, 0);
            Assert.LessOrEqual(CommunicationConstants.MAX_CONCURRENT_PROCESSES, 10);
        }

        [Test]
        public void CommunicationConstants_BufferSize_IsReasonable()
        {
            // 4KB buffer
            Assert.AreEqual(4096, CommunicationConstants.OUTPUT_BUFFER_SIZE);
        }

        [Test]
        public void CommunicationConstants_ThreadJoinTimeout_IsReasonable()
        {
            Assert.Greater(CommunicationConstants.THREAD_JOIN_TIMEOUT_MS, 0);
            Assert.LessOrEqual(CommunicationConstants.THREAD_JOIN_TIMEOUT_MS, 10000);
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

        #region GripperConstants Tests

        [Test]
        public void GripperConstants_SmoothTime_IsPositive()
        {
            Assert.Greater(GripperConstants.DEFAULT_SMOOTH_TIME, 0f);
        }

        #endregion
    }
}
