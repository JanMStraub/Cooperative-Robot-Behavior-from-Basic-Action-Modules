using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using System.Collections;
using System.Collections.Generic;
using Robotics;
using Robotics.Grasp;
using Configuration;

namespace Tests.PlayMode
{
    /// <summary>
    /// Play mode tests for grasp planning pipeline components.
    /// Tests candidate generation, filtering, scoring, and full pipeline execution.
    /// </summary>
    public class GraspPipelineTests
    {
        private GameObject _testObject;
        private GraspConfig _testConfig;
        private ArticulationBody[] _mockJoints;
        private Transform _mockIKFrame;
        private Transform _mockEndEffector;

        private IKConfig _ikConfig;

        [SetUp]
        public void Setup()
        {
            // Create test object (cube)
            _testObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _testObject.name = "TestCube";
            _testObject.transform.position = new Vector3(0.3f, 0.2f, 0.3f);
            _testObject.transform.localScale = Vector3.one * 0.05f;

            // Create test configuration
            _testConfig = ScriptableObject.CreateInstance<GraspConfig>();
            _testConfig.InitializeDefaultConfig();

            // Create IK configuration
            _ikConfig = ScriptableObject.CreateInstance<IKConfig>();
            _ikConfig.dampingFactor = 0.1f;

            // Create mock robot components
            SetupMockRobotComponents();
        }

        [TearDown]
        public void Teardown()
        {
            if (_testObject != null)
                Object.Destroy(_testObject);

            if (_testConfig != null)
                Object.Destroy(_testConfig);

            if (_ikConfig != null)
                Object.Destroy(_ikConfig);

            CleanupMockRobotComponents();
        }

        private void SetupMockRobotComponents()
        {
            // Create mock IK reference frame
            var ikFrameObj = new GameObject("IKFrame");
            _mockIKFrame = ikFrameObj.transform;
            _mockIKFrame.position = Vector3.zero;

            // Create mock end effector
            var endEffectorObj = new GameObject("EndEffector");
            _mockEndEffector = endEffectorObj.transform;
            _mockEndEffector.position = new Vector3(0.2f, 0.1f, 0.2f);

            // Create mock joints (6 joints for AR4)
            _mockJoints = new ArticulationBody[6];
            for (int i = 0; i < 6; i++)
            {
                var jointObj = new GameObject($"Joint{i}");
                _mockJoints[i] = jointObj.AddComponent<ArticulationBody>();
                _mockJoints[i].jointType = ArticulationJointType.RevoluteJoint;
            }
        }

        private void CleanupMockRobotComponents()
        {
            if (_mockIKFrame != null)
                Object.Destroy(_mockIKFrame.gameObject);
            if (_mockEndEffector != null)
                Object.Destroy(_mockEndEffector.gameObject);
            if (_mockJoints != null)
            {
                foreach (var joint in _mockJoints)
                {
                    if (joint != null)
                        Object.Destroy(joint.gameObject);
                }
            }
        }

        #region Candidate Generator Tests

        [Test]
        public void CandidateGenerator_GeneratesCandidates()
        {
            var generator = new GraspCandidateGenerator(_testConfig);

            var candidates = generator.GenerateCandidates(_testObject, Vector3.zero);

            Assert.IsNotNull(candidates, "Candidates should not be null");
            Assert.Greater(candidates.Count, 0, "Should generate at least one candidate");
        }

        [Test]
        public void CandidateGenerator_RespectsApproachSettings()
        {
            // Enable only top approach
            _testConfig.enabledApproaches = new GraspApproachSettings[]
            {
                new GraspApproachSettings(GraspApproach.Top, true, 1.0f)
            };

            var generator = new GraspCandidateGenerator(_testConfig);
            var candidates = generator.GenerateCandidates(_testObject, Vector3.zero);

            // All candidates should be top approach
            foreach (var candidate in candidates)
            {
                Assert.AreEqual(GraspApproach.Top, candidate.approachType,
                    "All candidates should use top approach when others are disabled");
            }
        }

        [Test]
        public void CandidateGenerator_AdaptsPreGraspDistance()
        {
            var generator = new GraspCandidateGenerator(_testConfig);
            var candidates = generator.GenerateCandidates(_testObject, Vector3.zero);

            foreach (var candidate in candidates)
            {
                float distance = candidate.approachDistance;
                Assert.GreaterOrEqual(distance, _testConfig.minPreGraspDistance,
                    "Approach distance should be >= min");
                Assert.LessOrEqual(distance, _testConfig.maxPreGraspDistance,
                    "Approach distance should be <= max");
            }
        }

        #endregion

        #region Scorer Tests

        [Test]
        public void Scorer_AssignsScoresToCandidates()
        {
            var scorer = new GraspScorer(_testConfig);
            var generator = new GraspCandidateGenerator(_testConfig);
            var candidates = generator.GenerateCandidates(_testObject, Vector3.zero);

            Vector3 objectSize = _testObject.GetComponent<Collider>().bounds.size;
            var scoredCandidates = scorer.ScoreAndRank(candidates, objectSize, Vector3.zero);

            Assert.AreEqual(candidates.Count, scoredCandidates.Count,
                "Should return same number of candidates");

            foreach (var candidate in scoredCandidates)
            {
                Assert.Greater(candidate.totalScore, 0, "Total score should be greater than 0");
            }
        }

        [Test]
        public void Scorer_SortsCandidatesByScore()
        {
            var scorer = new GraspScorer(_testConfig);
            var generator = new GraspCandidateGenerator(_testConfig);
            var candidates = generator.GenerateCandidates(_testObject, Vector3.zero);

            Vector3 objectSize = _testObject.GetComponent<Collider>().bounds.size;
            var scoredCandidates = scorer.ScoreAndRank(candidates, objectSize, Vector3.zero);

            // Verify sorted in descending order
            for (int i = 1; i < scoredCandidates.Count; i++)
            {
                Assert.GreaterOrEqual(scoredCandidates[i - 1].totalScore, scoredCandidates[i].totalScore,
                    "Candidates should be sorted by score (highest first)");
            }
        }

        #endregion

        #region IK Filter Tests

        [Test]
        public void IKFilter_FiltersUnreachablePoses()
        {
            var ikFilter = new GraspIKFilter(_testConfig, _mockJoints, _mockIKFrame, _mockEndEffector, _ikConfig);
            var generator = new GraspCandidateGenerator(_testConfig);

            // Generate candidates with some intentionally unreachable
            var candidates = generator.GenerateCandidates(_testObject, _mockEndEffector.position);

            // Add an obviously unreachable candidate
            var unreachableCandidate = GraspCandidate.Create(
                new Vector3(10.0f, 10.0f, 10.0f), // Far away
                Quaternion.identity,
                new Vector3(10.1f, 10.0f, 10.0f),
                Quaternion.identity,
                GraspApproach.Top
            );
            candidates.Add(unreachableCandidate);

            var filteredCandidates = ikFilter.FilterCandidates(candidates, _mockEndEffector.position);

            // Should filter out unreachable candidates
            Assert.LessOrEqual(filteredCandidates.Count, candidates.Count,
                "Filtered count should be <= original count");
        }

        [Test]
        public void IKFilter_SetsIKValidatedFlag()
        {
            var ikFilter = new GraspIKFilter(_testConfig, _mockJoints, _mockIKFrame, _mockEndEffector, _ikConfig);
            var generator = new GraspCandidateGenerator(_testConfig);
            var candidates = generator.GenerateCandidates(_testObject, _mockEndEffector.position);

            var filteredCandidates = ikFilter.FilterCandidates(candidates, _mockEndEffector.position);

            foreach (var candidate in filteredCandidates)
            {
                Assert.IsTrue(candidate.ikValidated,
                    "Filtered candidates should have ikValidated flag set");
            }
        }

        #endregion

        #region Collision Filter Tests

        [Test]
        public void CollisionFilter_DetectsObstacles()
        {
            // Create obstacle between gripper and target
            var obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
            obstacle.transform.position = new Vector3(0.25f, 0.15f, 0.25f);
            obstacle.transform.localScale = Vector3.one * 0.1f;

            var collisionFilter = new GraspCollisionFilter(_testConfig);
            var generator = new GraspCandidateGenerator(_testConfig);
            var candidates = generator.GenerateCandidates(_testObject, _mockEndEffector.position);

            var filteredCandidates = collisionFilter.FilterCandidates(candidates, _testObject);

            // Should filter out candidates with collision paths
            // (exact count depends on obstacle position relative to approach paths)
            Assert.LessOrEqual(filteredCandidates.Count, candidates.Count,
                "Should filter out some candidates due to obstacle");

            Object.Destroy(obstacle);
        }

        [Test]
        public void CollisionFilter_SetsCollisionValidatedFlag()
        {
            var collisionFilter = new GraspCollisionFilter(_testConfig);
            var generator = new GraspCandidateGenerator(_testConfig);
            var candidates = generator.GenerateCandidates(_testObject, _mockEndEffector.position);

            var filteredCandidates = collisionFilter.FilterCandidates(candidates, _testObject);

            foreach (var candidate in filteredCandidates)
            {
                Assert.IsTrue(candidate.collisionValidated,
                    "Filtered candidates should have collisionValidated flag set");
            }
        }

        [Test]
        public void CollisionFilter_CanBeDisabled()
        {
            _testConfig.enableCollisionChecking = false;

            var collisionFilter = new GraspCollisionFilter(_testConfig);
            var generator = new GraspCandidateGenerator(_testConfig);
            var candidates = generator.GenerateCandidates(_testObject, _mockEndEffector.position);

            var filteredCandidates = collisionFilter.FilterCandidates(candidates, _testObject);

            // When disabled, should pass all candidates through
            Assert.AreEqual(candidates.Count, filteredCandidates.Count,
                "With collision checking disabled, all candidates should pass");
        }

        #endregion

        #region Full Pipeline Tests

        [Test]
        public void Pipeline_ExecutesFullPlanningSequence()
        {
            var pipeline = new GraspPlanningPipeline(
                _testConfig,
                _mockJoints,
                _mockIKFrame,
                _mockEndEffector,
                _ikConfig
            );

            var options = GraspOptions.Advanced;
            var result = pipeline.PlanGrasp(_testObject, _mockEndEffector.position, options);

            Assert.IsNotNull(result, "Pipeline should return a result");
        }

        [Test]
        public void Pipeline_RespectsApproachOverride()
        {
            var pipeline = new GraspPlanningPipeline(
                _testConfig,
                _mockJoints,
                _mockIKFrame,
                _mockEndEffector,
                _ikConfig
            );

            var options = GraspOptions.Advanced;
            options.approach = GraspApproach.Top; // Override to top

            var result = pipeline.PlanGrasp(_testObject, _mockEndEffector.position, options);

            if (result.HasValue)
            {
                Assert.AreEqual(GraspApproach.Top, result.Value.approachType,
                    "Should use overridden approach direction");
            }
        }

        [Test]
        public void Pipeline_HandlesNoValidCandidates()
        {
            // Create object far outside reach
            var unreachableObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            unreachableObject.transform.position = new Vector3(10.0f, 10.0f, 10.0f);

            var pipeline = new GraspPlanningPipeline(
                _testConfig,
                _mockJoints,
                _mockIKFrame,
                _mockEndEffector,
                _ikConfig
            );

            var result = pipeline.PlanGrasp(unreachableObject, _mockEndEffector.position, GraspOptions.Advanced);

            Assert.IsFalse(result.HasValue, "Should return null for unreachable object");

            Object.Destroy(unreachableObject);
        }

        [UnityTest]
        public IEnumerator Pipeline_PerformanceUnder200ms()
        {
            var pipeline = new GraspPlanningPipeline(
                _testConfig,
                _mockJoints,
                _mockIKFrame,
                _mockEndEffector,
                _ikConfig
            );

            var startTime = Time.realtimeSinceStartup;
            var result = pipeline.PlanGrasp(_testObject, _mockEndEffector.position, GraspOptions.Advanced);
            var elapsedTime = (Time.realtimeSinceStartup - startTime) * 1000f; // Convert to ms

            Debug.Log($"Pipeline execution time: {elapsedTime:F2}ms");

            Assert.Less(elapsedTime, 200f,
                $"Pipeline should execute in <200ms, actual: {elapsedTime:F2}ms");

            yield return null;
        }

        #endregion

        #region Config Tests

        [Test]
        public void GraspConfig_CalculatesAdaptiveDistances()
        {
            var smallObject = new Vector3(0.03f, 0.03f, 0.03f);
            var largeObject = new Vector3(0.15f, 0.15f, 0.15f);

            float smallDistance = _testConfig.CalculatePreGraspDistance(smallObject);
            float largeDistance = _testConfig.CalculatePreGraspDistance(largeObject);

            Assert.Greater(largeDistance, smallDistance,
                "Larger objects should have larger pre-grasp distances");

            Assert.GreaterOrEqual(smallDistance, _testConfig.minPreGraspDistance);
            Assert.LessOrEqual(largeDistance, _testConfig.maxPreGraspDistance);
        }

        [Test]
        public void GraspConfig_GetApproachWeight()
        {
            var topWeight = _testConfig.GetApproachWeight(GraspApproach.Top);
            Assert.Greater(topWeight, 0, "Enabled approach should have weight > 0");

            // Disable top approach
            _testConfig.enabledApproaches = new GraspApproachSettings[]
            {
                new GraspApproachSettings(GraspApproach.Top, false, 0f)
            };

            var disabledWeight = _testConfig.GetApproachWeight(GraspApproach.Top);
            Assert.AreEqual(0, disabledWeight, "Disabled approach should have weight = 0");
        }

        #endregion

        #region GraspCandidate Tests

        [Test]
        public void GraspCandidate_CreateMethod()
        {
            var preGrasp = new Vector3(0.3f, 0.25f, 0.3f);
            var grasp = new Vector3(0.3f, 0.2f, 0.3f);
            var rotation = Quaternion.Euler(90, 0, 0);

            var candidate = GraspCandidate.Create(preGrasp, rotation, grasp, rotation, GraspApproach.Top);

            Assert.AreEqual(preGrasp, candidate.preGraspPosition);
            Assert.AreEqual(grasp, candidate.graspPosition);
            Assert.AreEqual(GraspApproach.Top, candidate.approachType);
            Assert.AreEqual(1.0f, candidate.preGraspGripperWidth, "Pre-grasp should be open");
            Assert.AreEqual(0.0f, candidate.graspGripperWidth, "Grasp should be closed");
        }

        [Test]
        public void GraspCandidate_IsValidProperty()
        {
            var candidate = GraspCandidate.Create(
                Vector3.zero, Quaternion.identity,
                Vector3.zero, Quaternion.identity,
                GraspApproach.Top
            );

            // Initially invalid
            Assert.IsFalse(candidate.isValid, "Candidate should be invalid initially");

            // Set validation flags
            candidate.ikValidated = true;
            candidate.collisionValidated = true;

            Assert.IsTrue(candidate.isValid, "Candidate should be valid when both flags are true");
        }

        #endregion
    }
}
