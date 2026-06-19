using System;
using System.Collections.Generic;
using UnityEngine;

namespace Simulation
{
    /// <summary>
    /// Workspace region definition for multi-robot spatial geometry.
    /// Allocation decisions belong in Python's WorldState - this class is geometry only.
    /// </summary>
    [Serializable]
    public class WorkspaceRegion
    {
        public string regionName;
        public Vector3 minBounds;
        public Vector3 maxBounds;
        public Color debugColor = Color.white;

        public WorkspaceRegion(string name, Vector3 min, Vector3 max)
        {
            regionName = name;
            minBounds = Vector3.Min(min, max);
            maxBounds = Vector3.Max(min, max);
        }

        public bool ContainsPosition(Vector3 position)
        {
            return position.x >= minBounds.x
                && position.x <= maxBounds.x
                && position.y >= minBounds.y
                && position.y <= maxBounds.y
                && position.z >= minBounds.z
                && position.z <= maxBounds.z;
        }

        public Vector3 GetCenter()
        {
            return (minBounds + maxBounds) / 2f;
        }

        public bool ValidateBounds()
        {
            return minBounds.x <= maxBounds.x
                && minBounds.y <= maxBounds.y
                && minBounds.z <= maxBounds.z;
        }

        public void FixInvertedBounds()
        {
            Vector3 actualMin = Vector3.Min(minBounds, maxBounds);
            Vector3 actualMax = Vector3.Max(minBounds, maxBounds);
            minBounds = actualMin;
            maxBounds = actualMax;
        }
    }

    /// <summary>
    /// Manages workspace geometry for multi-robot systems.
    ///
    /// Provides pure read-only geometry queries (region lookup, safe separation check)
    /// and debug visualization. All coordination decisions (who may move where) are
    /// made by Python via signal/wait operations.
    ///
    /// Usage:
    /// 1. Add to scene alongside SimulationManager
    /// 2. Configure workspace regions in inspector or programmatically
    /// 3. Query via WorkspaceManager.Instance for geometry data
    /// </summary>
    public class WorkspaceManager : MonoBehaviour
    {
        public static WorkspaceManager Instance { get; private set; }

        [Header("Workspace Configuration")]
        [Tooltip("Predefined workspace regions for robot coordination")]
        [SerializeField]
        private List<WorkspaceRegion> _workspaceRegions = new List<WorkspaceRegion>();

        [Header("Safety Parameters")]
        [Tooltip(
            "Minimum distance to maintain between robot end effectors (meters). Mirrors MIN_ROBOT_SEPARATION in config/Robot.py."
        )]
        [SerializeField]
        private float _minRobotSeparation = 0.2f;

        [Header("Robot Base Positions")]
        [Tooltip(
            "Base position of Robot1 in world coordinates. Mirrors ROBOT_BASE_POSITIONS[Robot1] in config/Robot.py."
        )]
        [SerializeField]
        private Vector3 _robot1BasePosition = new Vector3(-0.475f, 0f, 0f);

        [Tooltip(
            "Base position of Robot2 in world coordinates. Mirrors ROBOT_BASE_POSITIONS[Robot2] in config/Robot.py."
        )]
        [SerializeField]
        private Vector3 _robot2BasePosition = new Vector3(0.475f, 0f, 0f);

        [Header("Debug Visualization")]
        [SerializeField]
        private bool _enableDebugVisualization = true;

        private const string LOG_PREFIX = "[WORKSPACE_MANAGER]";

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

#if UNITY_EDITOR
        private void OnValidate()
        {
            ValidateAllRegionBounds();
        }
#endif

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
        }

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

        public List<WorkspaceRegion> GetAllRegions()
        {
            return new List<WorkspaceRegion>(_workspaceRegions);
        }

        public bool IsSafeSeparation(Vector3 pos1, Vector3 pos2)
        {
            float distance = Vector3.Distance(pos1, pos2);
            return distance >= _minRobotSeparation;
        }

        /// <summary>
        /// Get the base position of a robot by ID.
        /// Mirrors ROBOT_BASE_POSITIONS in ACRLPython/config/Robot.py.
        /// </summary>
        /// <param name="robotId">Robot identifier ("Robot1" or "Robot2")</param>
        /// <returns>Base position in world coordinates, or Vector3.zero if unknown</returns>
        public Vector3 GetRobotBasePosition(string robotId)
        {
            if (robotId == "Robot1")
                return _robot1BasePosition;
            if (robotId == "Robot2")
                return _robot2BasePosition;
            Debug.LogWarning($"{LOG_PREFIX} Unknown robotId '{robotId}' in GetRobotBasePosition");
            return Vector3.zero;
        }

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
                    regionColor.a = 0.2f;

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
                        region.regionName,
                        labelStyle
                    );
#endif
                }
            }
        }

#if UNITY_EDITOR
        public void AddRegion(string name, Vector3 min, Vector3 max, Color color)
        {
            var region = new WorkspaceRegion(name, min, max);
            region.debugColor = color;
            _workspaceRegions.Add(region);
        }

        public void RemoveRegion(string name)
        {
            _workspaceRegions.RemoveAll(r => r.regionName == name);
        }
#endif
    }
}
