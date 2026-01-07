using NUnit.Framework;
using UnityEngine;
using Configuration;

namespace Tests.EditMode
{
    /// <summary>
    /// Tests for GraspConfig ScriptableObject default values.
    /// Validates that precision improvements (January 2026) are properly configured.
    /// </summary>
    public class GraspConfigTests
    {
        private GraspConfig _config;

        [SetUp]
        public void SetUp()
        {
            // Create a new GraspConfig instance for testing
            _config = ScriptableObject.CreateInstance<GraspConfig>();
            _config.InitializeDefaultConfig();
        }

        [TearDown]
        public void TearDown()
        {
            if (_config != null)
            {
                Object.DestroyImmediate(_config);
            }
        }

        #region Precision Improvement Tests (January 2026)

        [Test]
        public void InitializeDefaultConfig_CandidatesPerApproach_IsEight()
        {
            // Test that default candidates per approach was increased from 5 to 8
            // for better grasp selection (Phase 5.2)
            Assert.AreEqual(8, _config.candidatesPerApproach,
                "Candidates per approach should be 8 for improved grasp selection");
        }

        [Test]
        public void InitializeDefaultConfig_MaxIKValidationIterations_IsFifty()
        {
            // Test that max IK iterations was increased from 20 to 50
            // for better convergence (Phase 5.2)
            Assert.AreEqual(50, _config.maxIKValidationIterations,
                "Max IK iterations should be 50 for better convergence");
        }

        [Test]
        public void InitializeDefaultConfig_IKValidationThreshold_IsTighter()
        {
            // Test that IK validation threshold was tightened from 0.01f to 0.005f
            // for more precise validation (Phase 5.2)
            Assert.AreEqual(0.005f, _config.ikValidationThreshold, 0.0001f,
                "IK validation threshold should be 0.005f (5mm) for tighter validation");
        }

        [Test]
        public void InitializeDefaultConfig_ValuesAreReasonable()
        {
            // Verify all default values are within reasonable ranges

            // Candidate generation
            Assert.Greater(_config.candidatesPerApproach, 0);
            Assert.LessOrEqual(_config.candidatesPerApproach, 20);

            // Pre-grasp distances
            Assert.Greater(_config.preGraspDistanceFactor, 0f);
            Assert.Greater(_config.minPreGraspDistance, 0f);
            Assert.Greater(_config.maxPreGraspDistance, _config.minPreGraspDistance);

            // IK validation
            Assert.Greater(_config.maxIKValidationIterations, 0);
            Assert.LessOrEqual(_config.maxIKValidationIterations, 500);
            Assert.Greater(_config.ikValidationThreshold, 0f);
            Assert.Less(_config.ikValidationThreshold, 0.1f);

            // Angle variation
            Assert.Greater(_config.angleVariationRange, 0f);
            Assert.Less(_config.angleVariationRange, 90f);
        }

        [Test]
        public void InitializeDefaultConfig_IKValidationEnabled()
        {
            // IK validation should be enabled by default
            Assert.IsTrue(_config.enableIKValidation,
                "IK validation should be enabled by default");
        }

        [Test]
        public void InitializeDefaultConfig_ScoreWeightsArePositive()
        {
            // All scoring weights should be positive
            Assert.Greater(_config.ikScoreWeight, 0f);
            Assert.Greater(_config.approachScoreWeight, 0f);
            Assert.Greater(_config.depthScoreWeight, 0f);
            Assert.Greater(_config.stabilityScoreWeight, 0f);
            Assert.Greater(_config.antipodalScoreWeight, 0f);
        }

        [Test]
        public void InitializeDefaultConfig_VariationRangesAreReasonable()
        {
            // Variation ranges should be positive and reasonable
            Assert.Greater(_config.angleVariationRange, 0f);
            Assert.Less(_config.angleVariationRange, 45f);

            Assert.Greater(_config.distanceVariationRange, 0f);
            Assert.Less(_config.distanceVariationRange, 1f);

            Assert.Greater(_config.depthVariationRange, 0f);
            Assert.Less(_config.depthVariationRange, 1f);
        }

        #endregion

        #region Basic Configuration Tests

        [Test]
        public void InitializeDefaultConfig_EnablesRetreat()
        {
            Assert.IsTrue(_config.enableRetreat,
                "Retreat should be enabled by default");
        }

        [Test]
        public void InitializeDefaultConfig_EnablesCollisionChecking()
        {
            Assert.IsTrue(_config.enableCollisionChecking,
                "Collision checking should be enabled by default");
        }

        [Test]
        public void InitializeDefaultConfig_CollisionParameters_AreReasonable()
        {
            Assert.Greater(_config.collisionCheckWaypoints, 0);
            Assert.Greater(_config.collisionCheckRadius, 0f);
        }

        [Test]
        public void InitializeDefaultConfig_GripperGeometry_IsValid()
        {
            Assert.IsNotNull(_config.gripperGeometry,
                "Gripper geometry should be initialized");
        }

        [Test]
        public void InitializeDefaultConfig_TargetGraspDepth_IsReasonable()
        {
            Assert.Greater(_config.targetGraspDepth, 0f);
            Assert.LessOrEqual(_config.targetGraspDepth, 1f);
        }

        #endregion
    }
}
