using NUnit.Framework;
using System.Collections;
using UnityEngine;
using UnityEngine.TestTools;
using Simulation;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for WorkspaceManager.
    /// Validates workspace geometry queries, region management, and safe separation.
    /// </summary>
    public class WorkspaceManagerTests
    {
        private GameObject _managerObject;
        private WorkspaceManager _manager;

        [UnitySetUp]
        public IEnumerator Setup()
        {
            // Clean up any existing instance
            if (WorkspaceManager.Instance != null)
            {
                Object.DestroyImmediate(WorkspaceManager.Instance.gameObject);
            }

            _managerObject = new GameObject("TestWorkspaceManager");
            _manager = _managerObject.AddComponent<WorkspaceManager>();

            // Wait for Awake() to complete and default regions to initialize
            yield return null;
        }

        [TearDown]
        public void TearDown()
        {
            if (_managerObject != null)
            {
                Object.DestroyImmediate(_managerObject);
            }
        }

        #region Singleton Tests

        [Test]
        public void WorkspaceManager_Singleton_IsSet()
        {
            Assert.IsNotNull(WorkspaceManager.Instance);
            Assert.AreEqual(_manager, WorkspaceManager.Instance);
        }

        [UnityTest]
        public IEnumerator WorkspaceManager_SecondInstance_IsDestroyed()
        {
            var duplicateObject = new GameObject("DuplicateWorkspaceManager");
            var duplicate = duplicateObject.AddComponent<WorkspaceManager>();

            yield return null;

            Assert.AreEqual(_manager, WorkspaceManager.Instance);
            Assert.IsTrue(duplicateObject == null);
        }

        #endregion

        #region Initialization Tests

        [Test]
        public void WorkspaceManager_DefaultRegions_AreCreated()
        {
            var regions = _manager.GetAllRegions();
            Assert.IsNotNull(regions);
            Assert.GreaterOrEqual(regions.Count, 4); // left, right, shared, center
        }

        [Test]
        public void WorkspaceManager_DefaultRegions_HaveCorrectNames()
        {
            Assert.IsNotNull(_manager.GetRegion("left_workspace"));
            Assert.IsNotNull(_manager.GetRegion("right_workspace"));
            Assert.IsNotNull(_manager.GetRegion("shared_zone"));
            Assert.IsNotNull(_manager.GetRegion("center"));
        }

        #endregion

        #region Position-based Region Tests

        [Test]
        public void GetRegionAtPosition_LeftWorkspace_ReturnsCorrectRegion()
        {
            Vector3 leftPosition = new Vector3(-0.5f, 0f, 0.2f);
            var region = _manager.GetRegionAtPosition(leftPosition);

            Assert.IsNotNull(region);
            Assert.AreEqual("left_workspace", region.regionName);
        }

        [Test]
        public void GetRegionAtPosition_RightWorkspace_ReturnsCorrectRegion()
        {
            Vector3 rightPosition = new Vector3(0.5f, 0f, 0.2f);
            var region = _manager.GetRegionAtPosition(rightPosition);

            Assert.IsNotNull(region);
            Assert.AreEqual("right_workspace", region.regionName);
        }

        [Test]
        public void GetRegionAtPosition_SharedZone_ReturnsCorrectRegion()
        {
            Vector3 sharedPosition = new Vector3(0f, 0f, 0.2f);
            var region = _manager.GetRegionAtPosition(sharedPosition);

            Assert.IsNotNull(region);
            // Position (0, 0, 0.2) is in both "center" and "shared_zone"
            // GetRegionAtPosition returns the smallest region, which is "center"
            Assert.AreEqual("center", region.regionName);
        }

        [Test]
        public void GetRegionAtPosition_OutsideAllRegions_ReturnsNull()
        {
            Vector3 outsidePosition = new Vector3(10f, 10f, 10f);
            var region = _manager.GetRegionAtPosition(outsidePosition);

            Assert.IsNull(region);
        }

        #endregion

        #region Safety Separation Tests

        [Test]
        public void IsSafeSeparation_FarApart_ReturnsTrue()
        {
            Vector3 pos1 = new Vector3(0f, 0f, 0f);
            Vector3 pos2 = new Vector3(1f, 0f, 0f);

            bool safe = _manager.IsSafeSeparation(pos1, pos2);
            Assert.IsTrue(safe);
        }

        [Test]
        public void IsSafeSeparation_TooClose_ReturnsFalse()
        {
            Vector3 pos1 = new Vector3(0f, 0f, 0f);
            Vector3 pos2 = new Vector3(0.1f, 0f, 0f); // Less than 0.2m default separation

            bool safe = _manager.IsSafeSeparation(pos1, pos2);
            Assert.IsFalse(safe);
        }

        [Test]
        public void IsSafeSeparation_ExactlyAtThreshold_ReturnsTrue()
        {
            Vector3 pos1 = new Vector3(0f, 0f, 0f);
            Vector3 pos2 = new Vector3(0.2f, 0f, 0f); // Exactly 0.2m

            bool safe = _manager.IsSafeSeparation(pos1, pos2);
            Assert.IsTrue(safe);
        }

        #endregion

        #region Robot Base Position Tests

        [Test]
        public void GetRobotBasePosition_Robot1_ReturnsCorrectPosition()
        {
            var pos = _manager.GetRobotBasePosition("Robot1");
            Assert.AreEqual(new Vector3(-0.475f, 0f, 0f), pos);
        }

        [Test]
        public void GetRobotBasePosition_Robot2_ReturnsCorrectPosition()
        {
            var pos = _manager.GetRobotBasePosition("Robot2");
            Assert.AreEqual(new Vector3(0.475f, 0f, 0f), pos);
        }

        [Test]
        public void GetRobotBasePosition_UnknownRobot_ReturnsZero()
        {
            LogAssert.Expect(LogType.Warning, "[WORKSPACE_MANAGER] Unknown robotId 'RobotX' in GetRobotBasePosition");
            var pos = _manager.GetRobotBasePosition("RobotX");
            Assert.AreEqual(Vector3.zero, pos);
        }

        #endregion

        #region WorkspaceRegion Tests

        [Test]
        public void WorkspaceRegion_ContainsPosition_ChecksBounds()
        {
            var region = new WorkspaceRegion(
                "test_region",
                new Vector3(-1f, -1f, 0f),
                new Vector3(1f, 1f, 0.5f)
            );

            Assert.IsTrue(region.ContainsPosition(new Vector3(0f, 0f, 0.25f)));
            Assert.IsFalse(region.ContainsPosition(new Vector3(2f, 0f, 0f)));
        }

        [Test]
        public void WorkspaceRegion_GetCenter_ReturnsCorrectCenter()
        {
            var region = new WorkspaceRegion(
                "test_region",
                new Vector3(-1f, -1f, 0f),
                new Vector3(1f, 1f, 0.5f)
            );

            Vector3 center = region.GetCenter();
            Assert.AreEqual(new Vector3(0f, 0f, 0.25f), center);
        }

        #endregion
    }
}
