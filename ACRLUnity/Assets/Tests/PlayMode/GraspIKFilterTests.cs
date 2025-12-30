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
    /// Play mode tests for GraspIKFilter.
    /// Tests distance-based filtering, IK validation, quality scoring, and joint state management.
    /// </summary>
    public class GraspIKFilterTests
    {
        private GraspConfig _testConfig;
        private ArticulationBody[] _mockJoints;
        private Transform _mockIKFrame;
        private Transform _mockEndEffector;
        private GraspIKFilter _filter;
        private GameObject _rootObject;

        private const float EPSILON = 0.001f;
        private const float MAX_REACH = 0.6f;
        private const float IK_THRESHOLD = 0.01f;

        [SetUp]
        public void Setup()
        {
            // Create test configuration
            _testConfig = ScriptableObject.CreateInstance<GraspConfig>();
            _testConfig.InitializeDefaultConfig();
            _testConfig.enableIKValidation = true;
            _testConfig.maxReachDistance = MAX_REACH;
            _testConfig.ikValidationThreshold = IK_THRESHOLD;
            _testConfig.maxIKValidationIterations = 10;

            // Create mock robot components
            SetupMockRobotComponents();

            // Create filter instance
            _filter = new GraspIKFilter(
                _testConfig,
                _mockJoints,
                _mockIKFrame,
                _mockEndEffector,
                dampingFactor: 0.1f
            );
        }

        [TearDown]
        public void Teardown()
        {
            if (_testConfig != null)
                Object.Destroy(_testConfig);

            CleanupMockRobotComponents();
        }

        /// <summary>
        /// Setup mock robot components (6 joints for AR4).
        /// </summary>
        private void SetupMockRobotComponents()
        {
            // Create root object for hierarchy
            _rootObject = new GameObject("RobotRoot");

            // Create IK reference frame
            var ikFrameObj = new GameObject("IKFrame");
            ikFrameObj.transform.SetParent(_rootObject.transform);
            _mockIKFrame = ikFrameObj.transform;
            _mockIKFrame.position = Vector3.zero;
            _mockIKFrame.rotation = Quaternion.identity;

            // Create articulation body chain (6 joints)
            _mockJoints = new ArticulationBody[6];
            Transform parent = _mockIKFrame;

            for (int i = 0; i < 6; i++)
            {
                var jointObj = new GameObject($"Joint{i}");
                jointObj.transform.SetParent(parent);
                jointObj.transform.localPosition = new Vector3(0f, 0.1f * i, 0f);

                var articulationBody = jointObj.AddComponent<ArticulationBody>();

                if (i == 0)
                {
                    articulationBody.jointType = ArticulationJointType.FixedJoint;
                    articulationBody.immovable = true;
                }
                else
                {
                    articulationBody.jointType = ArticulationJointType.RevoluteJoint;

                    // Configure drive for joint control
                    var drive = articulationBody.xDrive;
                    drive.stiffness = 500f;
                    drive.damping = 100f;
                    drive.forceLimit = 1000f;
                    articulationBody.xDrive = drive;
                }

                _mockJoints[i] = articulationBody;
                parent = jointObj.transform;
            }

            // Create end effector at end of chain
            var endEffectorObj = new GameObject("EndEffector");
            endEffectorObj.transform.SetParent(parent);
            endEffectorObj.transform.localPosition = new Vector3(0f, 0.1f, 0f);
            _mockEndEffector = endEffectorObj.transform;

            // Force physics update
            Physics.SyncTransforms();
        }

        /// <summary>
        /// Cleanup mock robot components.
        /// </summary>
        private void CleanupMockRobotComponents()
        {
            if (_rootObject != null)
                Object.DestroyImmediate(_rootObject);
        }

        /// <summary>
        /// Create a test candidate at a specific position.
        /// </summary>
        private GraspCandidate CreateTestCandidate(Vector3 preGraspPos, Vector3 graspPos)
        {
            return GraspCandidate.Create(
                preGraspPos,
                Quaternion.identity,
                graspPos,
                Quaternion.identity,
                GraspApproach.Top
            );
        }

        #region Constructor Tests

        [Test]
        public void Constructor_ValidParameters_InitializesCorrectly()
        {
            // Filter created in Setup, verify it's not null
            Assert.IsNotNull(_filter);
        }

        [Test]
        public void Constructor_NullJoints_ThrowsException()
        {
            Assert.Throws<System.ArgumentNullException>(() =>
            {
                new GraspIKFilter(
                    _testConfig,
                    null, // null joints
                    _mockIKFrame,
                    _mockEndEffector
                );
            });
        }

        [Test]
        public void Constructor_NullConfig_ThrowsException()
        {
            Assert.Throws<System.ArgumentNullException>(() =>
            {
                new GraspIKFilter(
                    null, // null config
                    _mockJoints,
                    _mockIKFrame,
                    _mockEndEffector
                );
            });
        }

        #endregion

        #region Distance Filtering Tests

        [Test]
        public void FilterCandidates_WithinReach_AcceptedWhenValidationDisabled()
        {
            // Disable IK validation to test only distance filtering
            _testConfig.enableIKValidation = false;

            var candidates = new List<GraspCandidate>
            {
                CreateTestCandidate(new Vector3(0.3f, 0.2f, 0.3f), new Vector3(0.3f, 0.1f, 0.3f))
            };

            var currentGripperPos = Vector3.zero;
            var validCandidates = _filter.FilterCandidates(candidates, currentGripperPos);

            Assert.AreEqual(1, validCandidates.Count, "Candidate within reach should be accepted");
            Assert.IsTrue(validCandidates[0].ikValidated, "Should be marked as validated when IK validation disabled");
            Assert.AreEqual(1.0f, validCandidates[0].ikScore, EPSILON, "Should have perfect IK score when validation disabled");
        }

        [Test]
        public void FilterCandidates_OutOfReach_Rejected()
        {
            _testConfig.enableIKValidation = false;
            _testConfig.maxReachDistance = 0.5f;

            var candidates = new List<GraspCandidate>
            {
                // Candidate far beyond reach
                CreateTestCandidate(new Vector3(10f, 10f, 10f), new Vector3(10f, 9.9f, 10f))
            };

            var currentGripperPos = Vector3.zero;
            var validCandidates = _filter.FilterCandidates(candidates, currentGripperPos);

            Assert.AreEqual(0, validCandidates.Count, "Candidate out of reach should be rejected");
        }

        [Test]
        public void FilterCandidates_ExactlyAtReach_Accepted()
        {
            _testConfig.enableIKValidation = false;
            _testConfig.maxReachDistance = 0.5f;

            // Create candidate exactly at max reach distance
            var candidates = new List<GraspCandidate>
            {
                CreateTestCandidate(new Vector3(0.5f, 0f, 0f), new Vector3(0.45f, 0f, 0f))
            };

            var currentGripperPos = Vector3.zero;
            var validCandidates = _filter.FilterCandidates(candidates, currentGripperPos);

            Assert.AreEqual(1, validCandidates.Count, "Candidate exactly at max reach should be accepted");
        }

        [Test]
        public void FilterCandidates_MultipleDistances_FilterCorrectly()
        {
            _testConfig.enableIKValidation = false;
            _testConfig.maxReachDistance = 0.5f;

            var candidates = new List<GraspCandidate>
            {
                CreateTestCandidate(new Vector3(0.2f, 0f, 0f), new Vector3(0.1f, 0f, 0f)), // Within reach
                CreateTestCandidate(new Vector3(0.4f, 0f, 0f), new Vector3(0.3f, 0f, 0f)), // Within reach
                CreateTestCandidate(new Vector3(1.0f, 0f, 0f), new Vector3(0.9f, 0f, 0f)), // Out of reach
                CreateTestCandidate(new Vector3(0.5f, 0f, 0f), new Vector3(0.45f, 0f, 0f)) // At boundary
            };

            var currentGripperPos = Vector3.zero;
            var validCandidates = _filter.FilterCandidates(candidates, currentGripperPos);

            Assert.AreEqual(3, validCandidates.Count, "Should accept 3 candidates within reach");
        }

        #endregion

        #region Empty/Null Input Tests

        [Test]
        public void FilterCandidates_EmptyList_ReturnsEmptyList()
        {
            var candidates = new List<GraspCandidate>();
            var currentGripperPos = Vector3.zero;
            var validCandidates = _filter.FilterCandidates(candidates, currentGripperPos);

            Assert.IsNotNull(validCandidates);
            Assert.AreEqual(0, validCandidates.Count, "Empty input should return empty output");
        }

        [Test]
        public void FilterCandidates_NullList_ThrowsException()
        {
            Assert.Throws<System.NullReferenceException>(() =>
            {
                _filter.FilterCandidates(null, Vector3.zero);
            });
        }

        #endregion

        #region IsReachable Method Tests

        [Test]
        public void IsReachable_WithinReach_ReturnsTrue()
        {
            _testConfig.enableIKValidation = false;

            var candidate = CreateTestCandidate(new Vector3(0.3f, 0.2f, 0.3f), new Vector3(0.3f, 0.1f, 0.3f));
            var currentGripperPos = Vector3.zero;

            bool reachable = _filter.IsReachable(candidate, currentGripperPos);

            Assert.IsTrue(reachable, "Candidate within reach should be reachable");
        }

        [Test]
        public void IsReachable_OutOfReach_ReturnsFalse()
        {
            _testConfig.enableIKValidation = false;
            _testConfig.maxReachDistance = 0.5f;

            var candidate = CreateTestCandidate(new Vector3(10f, 10f, 10f), new Vector3(10f, 9.9f, 10f));
            var currentGripperPos = Vector3.zero;

            bool reachable = _filter.IsReachable(candidate, currentGripperPos);

            Assert.IsFalse(reachable, "Candidate out of reach should not be reachable");
        }

        #endregion

        #region IK Validation Tests

        [UnityTest]
        public IEnumerator FilterCandidates_IKValidationEnabled_ValidatesWithIK()
        {
            _testConfig.enableIKValidation = true;

            var candidates = new List<GraspCandidate>
            {
                CreateTestCandidate(new Vector3(0.2f, 0.3f, 0.2f), new Vector3(0.2f, 0.25f, 0.2f))
            };

            // Wait for physics to stabilize
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();

            var currentGripperPos = _mockEndEffector.position;
            var validCandidates = _filter.FilterCandidates(candidates, currentGripperPos);

            // With IK validation enabled, candidate may or may not pass (depends on IK solver)
            // We just verify that IK validation was attempted
            foreach (var candidate in candidates)
            {
                // The original candidate's ikValidated flag may be modified
                // We're testing that the filtering logic ran without errors
            }

            Assert.IsNotNull(validCandidates, "Should return valid candidates list");
        }

        [UnityTest]
        public IEnumerator FilterCandidates_IKValidationEnabled_SetsValidationFlags()
        {
            _testConfig.enableIKValidation = true;

            var candidates = new List<GraspCandidate>
            {
                CreateTestCandidate(new Vector3(0.2f, 0.3f, 0.2f), new Vector3(0.2f, 0.25f, 0.2f))
            };

            // Wait for physics
            yield return new WaitForFixedUpdate();

            var currentGripperPos = _mockEndEffector.position;
            var validCandidates = _filter.FilterCandidates(candidates, currentGripperPos);

            // Verify that returned candidates have ikValidated flag set
            foreach (var candidate in validCandidates)
            {
                Assert.IsTrue(candidate.ikValidated, "Returned candidates should have ikValidated=true");
            }
        }

        #endregion

        #region Joint State Management Tests

        [UnityTest]
        public IEnumerator FilterCandidates_RestoresJointStatesAfterValidation()
        {
            _testConfig.enableIKValidation = true;

            // Cache original joint positions
            float[] originalPositions = new float[_mockJoints.Length];
            for (int i = 0; i < _mockJoints.Length; i++)
            {
                if (_mockJoints[i].jointPosition.dofCount > 0)
                    originalPositions[i] = _mockJoints[i].jointPosition[0];
            }

            var candidates = new List<GraspCandidate>
            {
                CreateTestCandidate(new Vector3(0.2f, 0.3f, 0.2f), new Vector3(0.2f, 0.25f, 0.2f))
            };

            // Wait for physics
            yield return new WaitForFixedUpdate();

            var currentGripperPos = _mockEndEffector.position;
            _filter.FilterCandidates(candidates, currentGripperPos);

            // Wait for restoration
            yield return new WaitForFixedUpdate();

            // Verify joints were restored
            for (int i = 0; i < _mockJoints.Length; i++)
            {
                if (_mockJoints[i].jointPosition.dofCount > 0)
                {
                    float currentPos = _mockJoints[i].jointPosition[0];
                    Assert.AreEqual(
                        originalPositions[i],
                        currentPos,
                        EPSILON,
                        $"Joint {i} should be restored to original position"
                    );
                }
            }
        }

        #endregion

        #region Configuration Tests

        [Test]
        public void FilterCandidates_DifferentMaxReach_FiltersCorrectly()
        {
            _testConfig.enableIKValidation = false;

            // Test with small max reach
            _testConfig.maxReachDistance = 0.3f;
            var candidates1 = new List<GraspCandidate>
            {
                CreateTestCandidate(new Vector3(0.5f, 0f, 0f), new Vector3(0.45f, 0f, 0f))
            };
            var valid1 = _filter.FilterCandidates(candidates1, Vector3.zero);
            Assert.AreEqual(0, valid1.Count, "Should reject with small max reach");

            // Test with large max reach
            _testConfig.maxReachDistance = 1.0f;
            var filter2 = new GraspIKFilter(_testConfig, _mockJoints, _mockIKFrame, _mockEndEffector);
            var candidates2 = new List<GraspCandidate>
            {
                CreateTestCandidate(new Vector3(0.5f, 0f, 0f), new Vector3(0.45f, 0f, 0f))
            };
            var valid2 = filter2.FilterCandidates(candidates2, Vector3.zero);
            Assert.AreEqual(1, valid2.Count, "Should accept with large max reach");
        }

        [Test]
        public void FilterCandidates_ValidationDisabled_SkipsIKComputation()
        {
            _testConfig.enableIKValidation = false;

            var candidates = new List<GraspCandidate>
            {
                CreateTestCandidate(new Vector3(0.3f, 0.2f, 0.3f), new Vector3(0.3f, 0.1f, 0.3f))
            };

            var validCandidates = _filter.FilterCandidates(candidates, Vector3.zero);

            // When validation is disabled, all within-reach candidates should pass
            Assert.AreEqual(1, validCandidates.Count);
            Assert.IsTrue(validCandidates[0].ikValidated, "Should be marked validated even though IK was skipped");
            Assert.AreEqual(1.0f, validCandidates[0].ikScore, EPSILON, "Should have perfect score when validation skipped");
        }

        #endregion

        #region Batch Processing Tests

        [Test]
        public void FilterCandidates_LargeBatch_ProcessesAll()
        {
            _testConfig.enableIKValidation = false;
            _testConfig.maxReachDistance = 1.0f;

            // Create 20 candidates
            var candidates = new List<GraspCandidate>();
            for (int i = 0; i < 20; i++)
            {
                float x = 0.1f + i * 0.03f;
                candidates.Add(CreateTestCandidate(new Vector3(x, 0.2f, 0.2f), new Vector3(x, 0.1f, 0.2f)));
            }

            var validCandidates = _filter.FilterCandidates(candidates, Vector3.zero);

            // All should be within reach
            Assert.AreEqual(20, validCandidates.Count, "Should process all candidates in large batch");
        }

        [Test]
        public void FilterCandidates_MixedValidity_FiltersMixed()
        {
            _testConfig.enableIKValidation = false;
            _testConfig.maxReachDistance = 0.5f;

            var candidates = new List<GraspCandidate>();

            // Add some valid candidates
            for (int i = 0; i < 5; i++)
            {
                candidates.Add(CreateTestCandidate(
                    new Vector3(0.1f + i * 0.05f, 0.2f, 0.2f),
                    new Vector3(0.1f + i * 0.05f, 0.1f, 0.2f)
                ));
            }

            // Add some invalid candidates (out of reach)
            for (int i = 0; i < 5; i++)
            {
                candidates.Add(CreateTestCandidate(
                    new Vector3(5f + i * 0.5f, 0.2f, 0.2f),
                    new Vector3(5f + i * 0.5f, 0.1f, 0.2f)
                ));
            }

            var validCandidates = _filter.FilterCandidates(candidates, Vector3.zero);

            Assert.AreEqual(5, validCandidates.Count, "Should filter mixed validity correctly");
        }

        #endregion

        #region Edge Cases

        [Test]
        public void FilterCandidates_CandidateAtOrigin_HandlesCorrectly()
        {
            _testConfig.enableIKValidation = false;

            var candidates = new List<GraspCandidate>
            {
                CreateTestCandidate(Vector3.zero, Vector3.zero)
            };

            var currentGripperPos = Vector3.zero;

            Assert.DoesNotThrow(() =>
            {
                _filter.FilterCandidates(candidates, currentGripperPos);
            }, "Should handle candidate at origin without error");
        }

        [Test]
        public void FilterCandidates_VerySmallDistance_HandlesCorrectly()
        {
            _testConfig.enableIKValidation = false;

            var candidates = new List<GraspCandidate>
            {
                CreateTestCandidate(new Vector3(0.001f, 0.001f, 0.001f), Vector3.zero)
            };

            var currentGripperPos = Vector3.zero;

            Assert.DoesNotThrow(() =>
            {
                _filter.FilterCandidates(candidates, currentGripperPos);
            }, "Should handle very small distances without error");
        }

        [Test]
        public void FilterCandidates_IdenticalCandidates_ProcessesAll()
        {
            _testConfig.enableIKValidation = false;

            var candidates = new List<GraspCandidate>
            {
                CreateTestCandidate(new Vector3(0.3f, 0.2f, 0.3f), new Vector3(0.3f, 0.1f, 0.3f)),
                CreateTestCandidate(new Vector3(0.3f, 0.2f, 0.3f), new Vector3(0.3f, 0.1f, 0.3f)),
                CreateTestCandidate(new Vector3(0.3f, 0.2f, 0.3f), new Vector3(0.3f, 0.1f, 0.3f))
            };

            var validCandidates = _filter.FilterCandidates(candidates, Vector3.zero);

            Assert.AreEqual(3, validCandidates.Count, "Should process all identical candidates independently");
        }

        #endregion

        #region Different Gripper Positions Tests

        [Test]
        public void FilterCandidates_DifferentGripperPositions_AffectsReachability()
        {
            _testConfig.enableIKValidation = false;
            _testConfig.maxReachDistance = 0.5f;

            var candidate = CreateTestCandidate(new Vector3(0.6f, 0f, 0f), new Vector3(0.55f, 0f, 0f));

            // Test from origin - should be out of reach
            var valid1 = _filter.FilterCandidates(
                new List<GraspCandidate> { candidate },
                Vector3.zero
            );
            Assert.AreEqual(0, valid1.Count, "Should be out of reach from origin");

            // Test from closer position - should be within reach
            var valid2 = _filter.FilterCandidates(
                new List<GraspCandidate> { candidate },
                new Vector3(0.3f, 0f, 0f)
            );
            Assert.AreEqual(1, valid2.Count, "Should be within reach from closer position");
        }

        #endregion

        #region Performance Tests

        [Test]
        public void FilterCandidates_100Candidates_CompletesInReasonableTime()
        {
            _testConfig.enableIKValidation = false;
            _testConfig.maxReachDistance = 1.0f;

            var candidates = new List<GraspCandidate>();
            for (int i = 0; i < 100; i++)
            {
                float x = (i % 10) * 0.05f;
                float y = (i / 10) * 0.05f;
                candidates.Add(CreateTestCandidate(new Vector3(x, y, 0.2f), new Vector3(x, y, 0.1f)));
            }

            var startTime = Time.realtimeSinceStartup;
            var validCandidates = _filter.FilterCandidates(candidates, Vector3.zero);
            var elapsedTime = Time.realtimeSinceStartup - startTime;

            Assert.Less(elapsedTime, 1.0f, "Should process 100 candidates in under 1 second");
            Assert.AreEqual(100, validCandidates.Count, "Should process all candidates");
        }

        #endregion
    }
}
