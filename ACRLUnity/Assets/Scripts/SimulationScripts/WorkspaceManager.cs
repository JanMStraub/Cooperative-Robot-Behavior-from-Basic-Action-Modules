using System;
using System.Collections.Generic;
using UnityEngine;

namespace Simulation
{
    /// <summary>
    /// Workspace region definition for multi-robot coordination
    /// </summary>
    [Serializable]
    public class WorkspaceRegion
    {
        public string regionName;
        public Vector3 minBounds;
        public Vector3 maxBounds;
        public string allocatedRobotId; // null if unallocated
        public Color debugColor = Color.white;

        public WorkspaceRegion(string name, Vector3 min, Vector3 max)
        {
            regionName = name;
            minBounds = min;
            maxBounds = max;
            allocatedRobotId = null;
        }

        /// <summary>
        /// Check if a position is within this region
        /// </summary>
        public bool ContainsPosition(Vector3 position)
        {
            return position.x >= minBounds.x
                && position.x <= maxBounds.x
                && position.y >= minBounds.y
                && position.y <= maxBounds.y
                && position.z >= minBounds.z
                && position.z <= maxBounds.z;
        }

        /// <summary>
        /// Get center position of region
        /// </summary>
        public Vector3 GetCenter()
        {
            return (minBounds + maxBounds) / 2f;
        }

        /// <summary>
        /// Check if region is currently allocated
        /// </summary>
        public bool IsAllocated()
        {
            return !string.IsNullOrEmpty(allocatedRobotId);
        }
    }

    /// <summary>
    /// Manages workspace allocation and coordination for multi-robot systems.
    ///
    /// This manager tracks workspace regions and their allocation to robots,
    /// syncs with Python WorldState, and prevents workspace conflicts.
    ///
    /// Features:
    /// - Workspace region definition and allocation
    /// - Collision zone tracking
    /// - Python WorldState synchronization
    /// - Debug visualization
    ///
    /// Usage:
    /// 1. Add to scene alongside SimulationManager
    /// 2. Configure workspace regions in inspector or programmatically
    /// 3. Query via WorkspaceManager.Instance for allocation decisions
    /// </summary>
    public class WorkspaceManager : MonoBehaviour
    {
        public static WorkspaceManager Instance { get; private set; }

        [Header("Workspace Configuration")]
        [Tooltip("Predefined workspace regions for robot coordination")]
        [SerializeField]
        private List<WorkspaceRegion> _workspaceRegions = new List<WorkspaceRegion>();

        [Header("Safety Parameters")]
        [Tooltip("Minimum distance to maintain between robot end effectors (meters)")]
        [SerializeField]
        private float _minRobotSeparation = 0.2f;

        // State tracking
        private Dictionary<string, string> _robotWorkspaceAllocation =
            new Dictionary<string, string>(); // robotId -> regionName
        private HashSet<string> _collisionZones = new HashSet<string>(); // Regions currently in use

        // Constants
        private const string LOG_PREFIX = "[WORKSPACE_MANAGER]";

        /// <summary>
        /// Initialize singleton instance
        /// </summary>
        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                InitializeWorkspaces();
            }
            else
            {
                Destroy(gameObject);
            }
        }

        /// <summary>
        /// Initialize default workspace regions if none configured
        /// </summary>
        private void InitializeWorkspaces()
        {
            // If no regions configured, create default dual-arm setup
            if (_workspaceRegions.Count == 0)
            {
                Debug.Log(
                    $"{LOG_PREFIX} No workspace regions configured, creating default dual-arm setup"
                );

                // Left workspace for Robot1
                // IMPORTANT: Must match Python LLMConfig.py WORKSPACE_REGIONS
                var leftRegion = new WorkspaceRegion(
                    "left_workspace",
                    new Vector3(-0.5f, 0.0f, -0.45f),
                    new Vector3(-0.15f, 0.6f, 0.45f)
                );
                leftRegion.debugColor = new Color(1f, 0.5f, 0.5f, 0.3f); // Light red

                // Right workspace for Robot2
                var rightRegion = new WorkspaceRegion(
                    "right_workspace",
                    new Vector3(0.15f, 0.0f, -0.45f),
                    new Vector3(0.5f, 0.6f, 0.45f)
                );
                rightRegion.debugColor = new Color(0.5f, 0.5f, 1f, 0.3f); // Light blue

                // Shared zone (requires coordination)
                var sharedZone = new WorkspaceRegion(
                    "shared_zone",
                    new Vector3(-0.15f, 0.0f, -0.45f),
                    new Vector3(0.15f, 0.6f, 0.45f)
                );
                sharedZone.debugColor = new Color(1f, 1f, 0.5f, 0.3f); // Light yellow

                // Center region
                var centerRegion = new WorkspaceRegion(
                    "center",
                    new Vector3(-0.15f, 0.0f, -0.1f),
                    new Vector3(0.15f, 0.5f, 0.1f)
                );
                centerRegion.debugColor = new Color(0.5f, 1f, 0.5f, 0.3f); // Light green

                _workspaceRegions.Add(leftRegion);
                _workspaceRegions.Add(rightRegion);
                _workspaceRegions.Add(sharedZone);
                _workspaceRegions.Add(centerRegion);

                Debug.Log(
                    $"{LOG_PREFIX} Created {_workspaceRegions.Count} default workspace regions"
                );
            }
        }

        /// <summary>
        /// Allocate a workspace region to a robot
        /// </summary>
        /// <param name="robotId">Robot requesting allocation</param>
        /// <param name="regionName">Name of region to allocate</param>
        /// <returns>True if allocation successful, false if region already allocated</returns>
        public bool AllocateRegion(string robotId, string regionName)
        {
            var region = GetRegion(regionName);
            if (region == null)
            {
                Debug.LogWarning($"{LOG_PREFIX} Region '{regionName}' not found");
                return false;
            }

            if (region.IsAllocated() && region.allocatedRobotId != robotId)
            {
                Debug.LogWarning(
                    $"{LOG_PREFIX} Region '{regionName}' already allocated to {region.allocatedRobotId}"
                );
                return false;
            }

            region.allocatedRobotId = robotId;
            _robotWorkspaceAllocation[robotId] = regionName;
            Debug.Log($"{LOG_PREFIX} Allocated region '{regionName}' to {robotId}");
            return true;
        }

        /// <summary>
        /// Release a workspace region allocation
        /// </summary>
        /// <param name="robotId">Robot releasing the region</param>
        /// <param name="regionName">Name of region to release</param>
        public void ReleaseRegion(string robotId, string regionName)
        {
            var region = GetRegion(regionName);
            if (region == null)
                return;

            if (region.allocatedRobotId == robotId)
            {
                region.allocatedRobotId = null;
                _robotWorkspaceAllocation.Remove(robotId);
                Debug.Log($"{LOG_PREFIX} Released region '{regionName}' from {robotId}");
            }
        }

        /// <summary>
        /// Release all regions allocated to a robot
        /// </summary>
        /// <param name="robotId">Robot to release all regions from</param>
        public void ReleaseAllRegions(string robotId)
        {
            foreach (var region in _workspaceRegions)
            {
                if (region.allocatedRobotId == robotId)
                {
                    region.allocatedRobotId = null;
                }
            }
            _robotWorkspaceAllocation.Remove(robotId);
            Debug.Log($"{LOG_PREFIX} Released all regions from {robotId}");
        }

        /// <summary>
        /// Check if a region is available (not allocated to another robot)
        /// </summary>
        /// <param name="regionName">Region to check</param>
        /// <param name="requestingRobotId">Robot making the request (optional)</param>
        /// <returns>True if available or allocated to requesting robot</returns>
        public bool IsRegionAvailable(string regionName, string requestingRobotId = null)
        {
            var region = GetRegion(regionName);
            if (region == null)
                return false;

            if (!region.IsAllocated())
                return true;
            if (requestingRobotId != null && region.allocatedRobotId == requestingRobotId)
                return true;

            return false;
        }

        /// <summary>
        /// Get the region containing a specific position
        /// </summary>
        /// <param name="position">World position to check</param>
        /// <returns>Region containing position, or null if no region found</returns>
        public WorkspaceRegion GetRegionAtPosition(Vector3 position)
        {
            foreach (var region in _workspaceRegions)
            {
                if (region.ContainsPosition(position))
                {
                    return region;
                }
            }
            return null;
        }

        /// <summary>
        /// Get a workspace region by name
        /// </summary>
        /// <param name="regionName">Name of region</param>
        /// <returns>WorkspaceRegion or null if not found</returns>
        public WorkspaceRegion GetRegion(string regionName)
        {
            return _workspaceRegions.Find(r => r.regionName == regionName);
        }

        /// <summary>
        /// Get all workspace regions
        /// </summary>
        /// <returns>List of all workspace regions</returns>
        public List<WorkspaceRegion> GetAllRegions()
        {
            return new List<WorkspaceRegion>(_workspaceRegions);
        }

        /// <summary>
        /// Check if a position is in a robot's designated workspace
        /// </summary>
        /// <param name="robotId">Robot to check</param>
        /// <param name="position">Target position</param>
        /// <returns>True if position is in robot's workspace</returns>
        public bool IsInRobotWorkspace(string robotId, Vector3 position)
        {
            if (!_robotWorkspaceAllocation.ContainsKey(robotId))
            {
                // Robot has no allocated workspace, allow any position
                return true;
            }

            string allocatedRegionName = _robotWorkspaceAllocation[robotId];
            var region = GetRegion(allocatedRegionName);

            return region != null && region.ContainsPosition(position);
        }

        /// <summary>
        /// Check if two positions maintain minimum safe separation
        /// </summary>
        /// <param name="pos1">First position</param>
        /// <param name="pos2">Second position</param>
        /// <returns>True if positions are safely separated</returns>
        public bool IsSafeSeparation(Vector3 pos1, Vector3 pos2)
        {
            float distance = Vector3.Distance(pos1, pos2);
            return distance >= _minRobotSeparation;
        }

        /// <summary>
        /// Mark a region as a collision zone (currently in use)
        /// </summary>
        /// <param name="regionName">Region to mark</param>
        public void MarkCollisionZone(string regionName)
        {
            _collisionZones.Add(regionName);
        }

        /// <summary>
        /// Clear a region from collision zones
        /// </summary>
        /// <param name="regionName">Region to clear</param>
        public void ClearCollisionZone(string regionName)
        {
            _collisionZones.Remove(regionName);
        }

        /// <summary>
        /// Check if a region is currently a collision zone
        /// </summary>
        /// <param name="regionName">Region to check</param>
        /// <returns>True if region is marked as collision zone</returns>
        public bool IsCollisionZone(string regionName)
        {
            return _collisionZones.Contains(regionName);
        }

        /// <summary>
        /// Get current workspace allocation state
        /// </summary>
        /// <returns>Dictionary mapping robotId to allocated region name</returns>
        public Dictionary<string, string> GetAllocationState()
        {
            return new Dictionary<string, string>(_robotWorkspaceAllocation);
        }

        /// <summary>
        /// Reset all workspace allocations
        /// </summary>
        public void ResetAllocations()
        {
            foreach (var region in _workspaceRegions)
            {
                region.allocatedRobotId = null;
            }
            _robotWorkspaceAllocation.Clear();
            _collisionZones.Clear();
            Debug.Log($"{LOG_PREFIX} Reset all workspace allocations");
        }

#if UNITY_EDITOR
        /// <summary>
        /// Add a workspace region (editor only)
        /// </summary>
        public void AddRegion(string name, Vector3 min, Vector3 max, Color color)
        {
            var region = new WorkspaceRegion(name, min, max);
            region.debugColor = color;
            _workspaceRegions.Add(region);
        }

        /// <summary>
        /// Remove a workspace region (editor only)
        /// </summary>
        public void RemoveRegion(string name)
        {
            _workspaceRegions.RemoveAll(r => r.regionName == name);
        }
#endif
    }
}
