using NUnit.Framework;
using System.Collections;
using UnityEngine;
using UnityEngine.TestTools;
using Simulation;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for WorkspaceManager (Phase 4).
    /// Validates workspace allocation, region management, and collision zone tracking.
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

        [Test]
        public void WorkspaceManager_DefaultRegions_AreUnallocated()
        {
            var leftRegion = _manager.GetRegion("left_workspace");
            var rightRegion = _manager.GetRegion("right_workspace");

            Assert.IsFalse(leftRegion.IsAllocated());
            Assert.IsFalse(rightRegion.IsAllocated());
        }

        #endregion

        #region Region Allocation Tests

        [Test]
        public void AllocateRegion_ValidRegion_ReturnsTrue()
        {
            bool result = _manager.AllocateRegion("Robot1", "left_workspace");
            Assert.IsTrue(result);
        }

        [Test]
        public void AllocateRegion_InvalidRegion_ReturnsFalse()
        {
            bool result = _manager.AllocateRegion("Robot1", "nonexistent_region");
            Assert.IsFalse(result);
        }

        [Test]
        public void AllocateRegion_AlreadyAllocated_ReturnsFalse()
        {
            _manager.AllocateRegion("Robot1", "left_workspace");
            bool result = _manager.AllocateRegion("Robot2", "left_workspace");
            Assert.IsFalse(result);
        }

        [Test]
        public void AllocateRegion_SameRobotTwice_ReturnsTrue()
        {
            _manager.AllocateRegion("Robot1", "left_workspace");
            bool result = _manager.AllocateRegion("Robot1", "left_workspace");
            Assert.IsTrue(result); // Same robot can "reallocate" its own region
        }

        [Test]
        public void AllocateRegion_SetsAllocatedRobotId()
        {
            _manager.AllocateRegion("Robot1", "left_workspace");
            var region = _manager.GetRegion("left_workspace");
            Assert.AreEqual("Robot1", region.allocatedRobotId);
        }

        #endregion

        #region Region Release Tests

        [Test]
        public void ReleaseRegion_AllocatedRegion_ReleasesSuccessfully()
        {
            _manager.AllocateRegion("Robot1", "left_workspace");
            _manager.ReleaseRegion("Robot1", "left_workspace");

            var region = _manager.GetRegion("left_workspace");
            Assert.IsFalse(region.IsAllocated());
        }

        [Test]
        public void ReleaseRegion_WrongRobot_DoesNotRelease()
        {
            _manager.AllocateRegion("Robot1", "left_workspace");
            _manager.ReleaseRegion("Robot2", "left_workspace");

            var region = _manager.GetRegion("left_workspace");
            Assert.IsTrue(region.IsAllocated());
            Assert.AreEqual("Robot1", region.allocatedRobotId);
        }

        [Test]
        public void ReleaseAllRegions_MultipleRegions_ReleasesAll()
        {
            _manager.AllocateRegion("Robot1", "left_workspace");
            _manager.AllocateRegion("Robot1", "center");
            _manager.ReleaseAllRegions("Robot1");

            var leftRegion = _manager.GetRegion("left_workspace");
            var centerRegion = _manager.GetRegion("center");

            Assert.IsFalse(leftRegion.IsAllocated());
            Assert.IsFalse(centerRegion.IsAllocated());
        }

        #endregion

        #region Region Availability Tests

        [Test]
        public void IsRegionAvailable_UnallocatedRegion_ReturnsTrue()
        {
            bool available = _manager.IsRegionAvailable("left_workspace");
            Assert.IsTrue(available);
        }

        [Test]
        public void IsRegionAvailable_AllocatedRegion_ReturnsFalse()
        {
            _manager.AllocateRegion("Robot1", "left_workspace");
            bool available = _manager.IsRegionAvailable("left_workspace", "Robot2");
            Assert.IsFalse(available);
        }

        [Test]
        public void IsRegionAvailable_AllocatedToSameRobot_ReturnsTrue()
        {
            _manager.AllocateRegion("Robot1", "left_workspace");
            bool available = _manager.IsRegionAvailable("left_workspace", "Robot1");
            Assert.IsTrue(available);
        }

        [Test]
        public void IsRegionAvailable_InvalidRegion_ReturnsFalse()
        {
            bool available = _manager.IsRegionAvailable("nonexistent_region");
            Assert.IsFalse(available);
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

        [Test]
        public void IsInRobotWorkspace_NoAllocation_ReturnsTrue()
        {
            Vector3 anyPosition = new Vector3(0.5f, 0f, 0.2f);
            bool inWorkspace = _manager.IsInRobotWorkspace("Robot1", anyPosition);

            Assert.IsTrue(inWorkspace); // No allocation, allow any position
        }

        [Test]
        public void IsInRobotWorkspace_WithAllocation_ChecksCorrectly()
        {
            _manager.AllocateRegion("Robot1", "left_workspace");

            Vector3 leftPosition = new Vector3(-0.5f, 0f, 0.2f);
            Vector3 rightPosition = new Vector3(0.5f, 0f, 0.2f);

            Assert.IsTrue(_manager.IsInRobotWorkspace("Robot1", leftPosition));
            Assert.IsFalse(_manager.IsInRobotWorkspace("Robot1", rightPosition));
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

        #region Collision Zone Tests

        [Test]
        public void MarkCollisionZone_ValidRegion_IsMarked()
        {
            _manager.MarkCollisionZone("left_workspace");
            bool isCollisionZone = _manager.IsCollisionZone("left_workspace");

            Assert.IsTrue(isCollisionZone);
        }

        [Test]
        public void ClearCollisionZone_MarkedRegion_IsCleared()
        {
            _manager.MarkCollisionZone("left_workspace");
            _manager.ClearCollisionZone("left_workspace");
            bool isCollisionZone = _manager.IsCollisionZone("left_workspace");

            Assert.IsFalse(isCollisionZone);
        }

        [Test]
        public void IsCollisionZone_UnmarkedRegion_ReturnsFalse()
        {
            bool isCollisionZone = _manager.IsCollisionZone("left_workspace");
            Assert.IsFalse(isCollisionZone);
        }

        [Test]
        public void MarkCollisionZone_MultipleRegions_TrackedIndependently()
        {
            _manager.MarkCollisionZone("left_workspace");
            _manager.MarkCollisionZone("right_workspace");

            Assert.IsTrue(_manager.IsCollisionZone("left_workspace"));
            Assert.IsTrue(_manager.IsCollisionZone("right_workspace"));
            Assert.IsFalse(_manager.IsCollisionZone("center"));
        }

        #endregion

        #region State Management Tests

        [Test]
        public void GetAllocationState_MultipleAllocations_ReturnsCorrectState()
        {
            _manager.AllocateRegion("Robot1", "left_workspace");
            _manager.AllocateRegion("Robot2", "right_workspace");

            var state = _manager.GetAllocationState();

            Assert.AreEqual(2, state.Count);
            // GetAllocationState returns Dictionary<string, HashSet<string>>
            // Each robot can have multiple allocated regions
            Assert.IsTrue(state["Robot1"].Contains("left_workspace"));
            Assert.IsTrue(state["Robot2"].Contains("right_workspace"));
        }

        [Test]
        public void ResetAllocations_ClearsAllState()
        {
            _manager.AllocateRegion("Robot1", "left_workspace");
            _manager.AllocateRegion("Robot2", "right_workspace");
            _manager.MarkCollisionZone("center");

            _manager.ResetAllocations();

            var leftRegion = _manager.GetRegion("left_workspace");
            var rightRegion = _manager.GetRegion("right_workspace");
            var state = _manager.GetAllocationState();

            Assert.IsFalse(leftRegion.IsAllocated());
            Assert.IsFalse(rightRegion.IsAllocated());
            Assert.IsFalse(_manager.IsCollisionZone("center"));
            Assert.AreEqual(0, state.Count);
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

        [Test]
        public void WorkspaceRegion_IsAllocated_ChecksAllocationStatus()
        {
            var region = new WorkspaceRegion("test_region", Vector3.zero, Vector3.one);

            Assert.IsFalse(region.IsAllocated());

            region.allocatedRobotId = "Robot1";
            Assert.IsTrue(region.IsAllocated());

            region.allocatedRobotId = null;
            Assert.IsFalse(region.IsAllocated());
        }

        #endregion

        #region Integration Scenario Tests

        [UnityTest]
        public IEnumerator Scenario_DualRobotCoordination_AllocatesCorrectly()
        {
            // Scenario: Robot1 enters left workspace, Robot2 enters right workspace
            _manager.AllocateRegion("Robot1", "left_workspace");
            yield return null;

            Assert.IsTrue(_manager.IsRegionAvailable("left_workspace", "Robot1"));
            Assert.IsFalse(_manager.IsRegionAvailable("left_workspace", "Robot2"));

            _manager.AllocateRegion("Robot2", "right_workspace");
            yield return null;

            Assert.IsTrue(_manager.IsRegionAvailable("right_workspace", "Robot2"));
            Assert.IsFalse(_manager.IsRegionAvailable("right_workspace", "Robot1"));

            // Both robots should succeed in their own workspaces
            Vector3 leftPos = new Vector3(-0.5f, 0f, 0.2f);
            Vector3 rightPos = new Vector3(0.5f, 0f, 0.2f);

            Assert.IsTrue(_manager.IsInRobotWorkspace("Robot1", leftPos));
            Assert.IsTrue(_manager.IsInRobotWorkspace("Robot2", rightPos));
        }

        [UnityTest]
        public IEnumerator Scenario_SharedZoneAccess_RequiresCoordination()
        {
            // Scenario: Both robots want to access shared zone
            Vector3 sharedPos = new Vector3(0f, 0f, 0.2f);

            _manager.AllocateRegion("Robot1", "shared_zone");
            _manager.MarkCollisionZone("shared_zone");
            yield return null;

            // Robot1 has access
            Assert.IsTrue(_manager.IsRegionAvailable("shared_zone", "Robot1"));

            // Robot2 blocked by collision zone
            Assert.IsFalse(_manager.IsRegionAvailable("shared_zone", "Robot2"));
            Assert.IsTrue(_manager.IsCollisionZone("shared_zone"));

            // Robot1 completes movement
            _manager.ClearCollisionZone("shared_zone");
            _manager.ReleaseRegion("Robot1", "shared_zone");
            yield return null;

            // Now Robot2 can access
            Assert.IsTrue(_manager.IsRegionAvailable("shared_zone", "Robot2"));
        }

        #endregion
    }
}
