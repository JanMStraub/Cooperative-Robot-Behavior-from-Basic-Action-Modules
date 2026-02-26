using System;
using System.Collections.Generic;
using UnityEngine;

namespace Simulation
{
    /// <summary>
    /// Workspace region definition for multi-robot coordination.
    /// </summary>
    [Serializable]
    public class WorkspaceRegion
    {
        public string regionName;
        public Vector3 minBounds;
        public Vector3 maxBounds;
        public string allocatedRobotId;
        public Color debugColor = Color.white;

        public WorkspaceRegion(string name, Vector3 min, Vector3 max)
        {
            regionName = name;
            minBounds = Vector3.Min(min, max);
            maxBounds = Vector3.Max(min, max);
            allocatedRobotId = null;
        }

        /// <summary>
        /// Check if a position is within this region.
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
        /// Get center position of region.
        /// </summary>
        public Vector3 GetCenter()
        {
            return (minBounds + maxBounds) / 2f;
        }

        /// <summary>
        /// Check if region is currently allocated.
        /// </summary>
        public bool IsAllocated()
        {
            return !string.IsNullOrEmpty(allocatedRobotId);
        }

        /// <summary>
        /// Validate that bounds are properly configured.
        /// </summary>
        /// <returns>True if bounds are valid (min <= max for all axes)</returns>
        public bool ValidateBounds()
        {
            return minBounds.x <= maxBounds.x
                && minBounds.y <= maxBounds.y
                && minBounds.z <= maxBounds.z;
        }

        /// <summary>
        /// Fix inverted bounds by swapping min/max values.
        /// </summary>
        public void FixInvertedBounds()
        {
            Vector3 actualMin = Vector3.Min(minBounds, maxBounds);
            Vector3 actualMax = Vector3.Max(minBounds, maxBounds);
            minBounds = actualMin;
            maxBounds = actualMax;
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

        [Tooltip(
            "Allow robot movement outside defined regions (void space). "
                + "FALSE = strict confinement (industrial safety mode), TRUE = permissive (development mode)"
        )]
        [SerializeField]
        private bool _allowMovementInVoid = true;

        [Header("Debug Visualization")]
        [SerializeField]
        private bool _enableDebugVisualization = true;

        private Dictionary<string, HashSet<string>> _robotWorkspaceAllocation = new();
        private HashSet<string> _collisionZones = new();
        private float _lastValidationTime;
        private const float VALIDATION_INTERVAL = 5f;
        private const string LOG_PREFIX = "[WORKSPACE_MANAGER]";

        // Pre-allocated buffer for the diagnostic out-overload of IsPositionAllowedForRobot
        private readonly List<WorkspaceRegion> _regionQueryBuffer = new List<WorkspaceRegion>();

        /// <summary>
        /// Initialize singleton instance.
        /// </summary>
        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                InitializeWorkspaces();
                _lastValidationTime = Time.time;
            }
            else
            {
                Destroy(gameObject);
            }
        }

#if UNITY_EDITOR
        /// <summary>
        /// Validate region bounds on inspector changes.
        /// </summary>
        private void OnValidate()
        {
            ValidateAllRegionBounds();
        }
#endif

        /// <summary>
        /// Periodic validation of allocation consistency.
        /// </summary>
        private void Update()
        {
            if (Time.time - _lastValidationTime >= VALIDATION_INTERVAL)
            {
                ValidateAndRepairAllocations();
                _lastValidationTime = Time.time;
            }
        }

        /// <summary>
        /// Validate and fix all region bounds.
        /// Detects and repairs inverted bounds (min > max).
        /// </summary>
        private void ValidateAllRegionBounds()
        {
            int fixedCount = 0;

            foreach (var region in _workspaceRegions)
            {
                if (!region.ValidateBounds())
                {
                    Debug.LogWarning(
                        $"{LOG_PREFIX} Region '{region.regionName}' has inverted bounds. "
                            + $"Min: {region.minBounds}, Max: {region.maxBounds}. Fixing."
                    );
                    region.FixInvertedBounds();
                    fixedCount++;
                }
            }

            if (fixedCount > 0)
            {
                Debug.LogWarning($"{LOG_PREFIX} Fixed {fixedCount} region(s) with inverted bounds");
            }
        }

        /// <summary>
        /// Initialize default workspace regions if none configured.
        /// </summary>
        private void InitializeWorkspaces()
        {
            ValidateAllRegionBounds();

            if (_workspaceRegions.Count == 0)
            {
                Debug.Log(
                    $"{LOG_PREFIX} No workspace regions configured, creating default dual-arm setup"
                );

                var leftRegion = new WorkspaceRegion(
                    "left_workspace",
                    new Vector3(-0.65f, 0.0f, -0.5f),
                    new Vector3(-0.1f, 0.7f, 0.5f)
                );
                leftRegion.debugColor = new Color(1f, 0.5f, 0.5f, 0.3f);

                var rightRegion = new WorkspaceRegion(
                    "right_workspace",
                    new Vector3(0.1f, 0.0f, -0.5f),
                    new Vector3(0.65f, 0.7f, 0.5f)
                );
                rightRegion.debugColor = new Color(0.5f, 0.5f, 1f, 0.3f);

                var sharedZone = new WorkspaceRegion(
                    "shared_zone",
                    new Vector3(-0.1f, 0.0f, -0.5f),
                    new Vector3(0.1f, 0.7f, 0.5f)
                );
                sharedZone.debugColor = new Color(1f, 1f, 0.5f, 0.3f);

                var centerRegion = new WorkspaceRegion(
                    "center",
                    new Vector3(-0.3f, 0.0f, -0.3f),
                    new Vector3(0.3f, 0.3f, 0.3f)
                );
                centerRegion.debugColor = new Color(0.5f, 1f, 0.5f, 0.3f);

                _workspaceRegions.Add(leftRegion);
                _workspaceRegions.Add(rightRegion);
                _workspaceRegions.Add(sharedZone);
                _workspaceRegions.Add(centerRegion);

                Debug.Log(
                    $"{LOG_PREFIX} Created {_workspaceRegions.Count} default workspace regions"
                );
            }

            // Sync dictionary from any pre-serialized region allocations (set via Inspector)
            foreach (var region in _workspaceRegions)
            {
                if (region.IsAllocated())
                {
                    string robotId = region.allocatedRobotId;
                    if (!_robotWorkspaceAllocation.ContainsKey(robotId))
                        _robotWorkspaceAllocation[robotId] = new HashSet<string>();
                    _robotWorkspaceAllocation[robotId].Add(region.regionName);
                }
            }
        }

        /// <summary>
        /// Allocate a workspace region to a robot.
        /// </summary>
        /// <param name="robotId">Robot requesting allocation</param>
        /// <param name="regionName">Name of region to allocate</param>
        /// <returns>True if allocation successful</returns>
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

            if (!_robotWorkspaceAllocation.ContainsKey(robotId))
            {
                _robotWorkspaceAllocation[robotId] = new HashSet<string>();
            }

            _robotWorkspaceAllocation[robotId].Add(regionName);
            Debug.Log($"{LOG_PREFIX} Allocated region '{regionName}' to {robotId}");
            return true;
        }

        /// <summary>
        /// Release a specific workspace region allocation.
        /// </summary>
        /// <param name="robotId">Robot releasing the region</param>
        /// <param name="regionName">Name of region to release</param>
        public void ReleaseRegion(string robotId, string regionName)
        {
            var region = GetRegion(regionName);
            if (region == null)
                return;

            bool wasAllocatedInRegion = region.allocatedRobotId == robotId;
            bool wasAllocatedInDict =
                _robotWorkspaceAllocation.ContainsKey(robotId)
                && _robotWorkspaceAllocation[robotId].Contains(regionName);

            if (wasAllocatedInRegion || wasAllocatedInDict)
            {
                region.allocatedRobotId = null;

                if (_robotWorkspaceAllocation.ContainsKey(robotId))
                {
                    _robotWorkspaceAllocation[robotId].Remove(regionName);
                    if (_robotWorkspaceAllocation[robotId].Count == 0)
                    {
                        _robotWorkspaceAllocation.Remove(robotId);
                    }
                }

                Debug.Log($"{LOG_PREFIX} Released region '{regionName}' from {robotId}");

                if (wasAllocatedInRegion != wasAllocatedInDict)
                {
                    Debug.LogWarning(
                        $"{LOG_PREFIX} Detected allocation desync for {robotId} in '{regionName}' "
                            + $"(Region: {wasAllocatedInRegion}, Dict: {wasAllocatedInDict}). Fixed."
                    );
                }
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
        /// Get all regions containing a specific position.
        /// </summary>
        /// <param name="position">World position to check</param>
        /// <returns>New list of all regions containing position</returns>
        public List<WorkspaceRegion> GetRegionsAtPosition(Vector3 position)
        {
            List<WorkspaceRegion> matchingRegions = new List<WorkspaceRegion>();
            foreach (var region in _workspaceRegions)
            {
                if (region.ContainsPosition(position))
                {
                    matchingRegions.Add(region);
                }
            }
            return matchingRegions;
        }

        /// <summary>
        /// Get all regions containing a specific position using buffer pattern.
        /// </summary>
        /// <param name="position">World position to check</param>
        /// <param name="resultBuffer">Pre-allocated list to populate</param>
        /// <returns>Number of matching regions found</returns>
        public int GetRegionsAtPosition(Vector3 position, List<WorkspaceRegion> resultBuffer)
        {
            if (resultBuffer == null)
            {
                Debug.LogError($"{LOG_PREFIX} GetRegionsAtPosition: resultBuffer cannot be null");
                return 0;
            }

            resultBuffer.Clear();
            foreach (var region in _workspaceRegions)
            {
                if (region.ContainsPosition(position))
                {
                    resultBuffer.Add(region);
                }
            }
            return resultBuffer.Count;
        }

        /// <summary>
        /// Get the smallest region containing a specific position.
        /// </summary>
        /// <param name="position">World position to check</param>
        /// <returns>Smallest region containing position, or null</returns>
        public WorkspaceRegion GetRegionAtPosition(Vector3 position)
        {
            WorkspaceRegion smallestRegion = null;
            float smallestVolume = float.MaxValue;

            foreach (var region in _workspaceRegions)
            {
                if (region.ContainsPosition(position))
                {
                    Vector3 size = region.maxBounds - region.minBounds;
                    float volume = size.x * size.y * size.z;

                    if (volume < smallestVolume)
                    {
                        smallestVolume = volume;
                        smallestRegion = region;
                    }
                }
            }

            return smallestRegion;
        }

        /// <summary>
        /// Get a workspace region by name.
        /// Uses manual iteration instead of List.Find to avoid delegate allocation on each call.
        /// </summary>
        /// <param name="regionName">Name of region</param>
        /// <returns>WorkspaceRegion or null</returns>
        public WorkspaceRegion GetRegion(string regionName)
        {
            foreach (var region in _workspaceRegions)
            {
                if (region.regionName == regionName)
                    return region;
            }
            return null;
        }

        /// <summary>
        /// Get all workspace regions.
        /// </summary>
        /// <returns>List of all workspace regions</returns>
        public List<WorkspaceRegion> GetAllRegions()
        {
            return new List<WorkspaceRegion>(_workspaceRegions);
        }

        /// <summary>
        /// Check if position is in any of robot's allocated workspaces.
        /// </summary>
        /// <param name="robotId">Robot to check</param>
        /// <param name="position">Target position</param>
        /// <returns>True if position is in any of robot's allocated regions</returns>
        public bool IsInRobotWorkspace(string robotId, Vector3 position)
        {
            if (!_robotWorkspaceAllocation.ContainsKey(robotId))
            {
                return true;
            }

            foreach (string regionName in _robotWorkspaceAllocation[robotId])
            {
                var region = GetRegion(regionName);
                if (region != null && region.ContainsPosition(position))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Check if position is allowed for robot.
        /// Checks robot permission for allocated regions.
        /// Free regions allowed.
        /// Allocates - use overload for hot paths.
        /// </summary>
        /// <param name="robotId">Robot attempting to move to position</param>
        /// <param name="position">Target position</param>
        /// <param name="violatedRegions">Output list of regions robot doesn't have access to</param>
        /// <returns>True if robot has access to ALL overlapping regions at position</returns>
        public bool IsPositionAllowedForRobot(
            string robotId,
            Vector3 position,
            out List<WorkspaceRegion> violatedRegions
        )
        {
            // Use class-level buffer to avoid allocating a temporary list for the region query
            GetRegionsAtPosition(position, _regionQueryBuffer);

            if (_regionQueryBuffer.Count == 0)
            {
                violatedRegions = new List<WorkspaceRegion>();
                return _allowMovementInVoid;
            }

            // Allocate the output list only when there are regions to inspect (diagnostic path)
            violatedRegions = new List<WorkspaceRegion>();
            foreach (var region in _regionQueryBuffer)
            {
                if (region.IsAllocated() && region.allocatedRobotId != robotId)
                {
                    violatedRegions.Add(region);
                }
            }

            return violatedRegions.Count == 0;
        }

        /// <summary>
        /// Check if position is allowed for robot.
        /// Zero allocations.
        /// Optimized for pathfinding.
        /// Respects void movement setting.
        /// </summary>
        /// <param name="robotId">Robot attempting to move to position</param>
        /// <param name="position">Target position</param>
        /// <returns>True if robot has access to ALL overlapping regions at position</returns>
        public bool IsPositionAllowedForRobot(string robotId, Vector3 position)
        {
            bool foundAnyRegion = false;

            foreach (var region in _workspaceRegions)
            {
                if (region.ContainsPosition(position))
                {
                    foundAnyRegion = true;

                    if (region.IsAllocated() && region.allocatedRobotId != robotId)
                    {
                        return false;
                    }
                }
            }

            return foundAnyRegion || _allowMovementInVoid;
        }

        /// <summary>
        /// Check if two positions maintain minimum safe separation.
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
        /// Mark a region as a collision zone.
        /// </summary>
        /// <param name="regionName">Region to mark</param>
        public void MarkCollisionZone(string regionName)
        {
            _collisionZones.Add(regionName);
        }

        /// <summary>
        /// Clear a region from collision zones.
        /// </summary>
        /// <param name="regionName">Region to clear</param>
        public void ClearCollisionZone(string regionName)
        {
            _collisionZones.Remove(regionName);
        }

        /// <summary>
        /// Check if a region is currently a collision zone.
        /// </summary>
        /// <param name="regionName">Region to check</param>
        /// <returns>True if region is marked as collision zone</returns>
        public bool IsCollisionZone(string regionName)
        {
            return _collisionZones.Contains(regionName);
        }

        /// <summary>
        /// Get current workspace allocation state.
        /// </summary>
        /// <returns>Dictionary mapping robotId to set of allocated region names</returns>
        public Dictionary<string, HashSet<string>> GetAllocationState()
        {
            var stateCopy = new Dictionary<string, HashSet<string>>();
            foreach (var kvp in _robotWorkspaceAllocation)
            {
                stateCopy[kvp.Key] = new HashSet<string>(kvp.Value);
            }
            return stateCopy;
        }

        /// <summary>
        /// Get all regions currently allocated to a robot.
        /// </summary>
        /// <param name="robotId">Robot to query</param>
        /// <returns>Set of region names</returns>
        public HashSet<string> GetRobotAllocations(string robotId)
        {
            if (_robotWorkspaceAllocation.ContainsKey(robotId))
            {
                return new HashSet<string>(_robotWorkspaceAllocation[robotId]);
            }
            return new HashSet<string>();
        }

        /// <summary>
        /// Reset all workspace allocations.
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

        /// <summary>
        /// Validate and fix allocation consistency.
        /// </summary>
        /// <returns>True if allocations were consistent</returns>
        public bool ValidateAndRepairAllocations()
        {
            bool isConsistent = true;
            int repairedCount = 0;

            foreach (var region in _workspaceRegions)
            {
                if (region.IsAllocated())
                {
                    string robotId = region.allocatedRobotId;

                    if (
                        !_robotWorkspaceAllocation.ContainsKey(robotId)
                        || !_robotWorkspaceAllocation[robotId].Contains(region.regionName)
                    )
                    {
                        Debug.LogWarning(
                            $"{LOG_PREFIX} Desync detected: Region '{region.regionName}' allocated to {robotId} "
                                + "but not in allocation dictionary. Adding to dictionary."
                        );

                        if (!_robotWorkspaceAllocation.ContainsKey(robotId))
                        {
                            _robotWorkspaceAllocation[robotId] = new HashSet<string>();
                        }
                        _robotWorkspaceAllocation[robotId].Add(region.regionName);

                        isConsistent = false;
                        repairedCount++;
                    }
                }
            }

            List<string> robotsToClean = new List<string>();
            foreach (var kvp in _robotWorkspaceAllocation)
            {
                string robotId = kvp.Key;
                HashSet<string> allocatedRegions = new HashSet<string>(kvp.Value);

                foreach (string regionName in allocatedRegions)
                {
                    var region = GetRegion(regionName);
                    if (region == null || region.allocatedRobotId != robotId)
                    {
                        Debug.LogWarning(
                            $"{LOG_PREFIX} Desync detected: Robot {robotId} has '{regionName}' in dictionary "
                                + "but region is not allocated to it. Removing from dictionary."
                        );

                        _robotWorkspaceAllocation[robotId].Remove(regionName);
                        isConsistent = false;
                        repairedCount++;
                    }
                }

                if (_robotWorkspaceAllocation[robotId].Count == 0)
                {
                    robotsToClean.Add(robotId);
                }
            }

            foreach (string robotId in robotsToClean)
            {
                _robotWorkspaceAllocation.Remove(robotId);
            }

            if (!isConsistent)
            {
                Debug.LogWarning(
                    $"{LOG_PREFIX} Allocation validation completed: {repairedCount} inconsistencies repaired"
                );
            }

            return isConsistent;
        }

        /// <summary>
        /// Draw workspace regions in Scene view for visual debugging.
        /// </summary>
        private void OnDrawGizmos()
        {
            if (_workspaceRegions == null || _workspaceRegions.Count == 0)
                return;

            if (_enableDebugVisualization)
            {
                foreach (var region in _workspaceRegions)
                {
                    Vector3 center = region.GetCenter();
                    Vector3 size = region.maxBounds - region.minBounds;

                    Color regionColor = region.debugColor;
                    if (region.IsAllocated())
                    {
                        regionColor.a = 0.5f;
                    }
                    else
                    {
                        regionColor.a = 0.2f;
                    }

                    Gizmos.color = regionColor;
                    Gizmos.DrawCube(center, size);

                    Color wireColor = new Color(
                        regionColor.r,
                        regionColor.g,
                        regionColor.b,
                        regionColor.a + 0.3f
                    );
                    Gizmos.color = wireColor;
                    Gizmos.DrawWireCube(center, size);

#if UNITY_EDITOR
                    var labelStyle = new GUIStyle();
                    labelStyle.normal.textColor = Color.white;
                    labelStyle.fontSize = 11;
                    UnityEditor.Handles.Label(
                        center + Vector3.up * (size.y / 2 + 0.05f),
                        $"{region.regionName}\n{(region.IsAllocated() ? $"[{region.allocatedRobotId}]" : "[Free]")}",
                        labelStyle
                    );
#endif
                }
            }
        }

#if UNITY_EDITOR
        /// <summary>
        /// Add a workspace region (editor only).
        /// </summary>
        public void AddRegion(string name, Vector3 min, Vector3 max, Color color)
        {
            var region = new WorkspaceRegion(name, min, max);
            region.debugColor = color;
            _workspaceRegions.Add(region);
        }

        /// <summary>
        /// Remove a workspace region (editor only).
        /// </summary>
        public void RemoveRegion(string name)
        {
            _workspaceRegions.RemoveAll(r => r.regionName == name);
        }
#endif
    }
}
