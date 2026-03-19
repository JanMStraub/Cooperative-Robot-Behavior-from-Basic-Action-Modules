using System;
using System.Collections.Generic;
using Core;
using Robotics;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Data structure for world state update messages sent to Python WorldState
    /// </summary>
    [System.Serializable]
    public class WorldStateUpdate
    {
        public string type = "world_state_update";
        public List<RobotStateData> robots;
        public List<ObjectStateData> objects;
        public float timestamp;
    }

    [System.Serializable]
    public class RobotStateData
    {
        public string robot_id;
        public PositionData position;
        public RotationData rotation;
        public PositionData target_position;
        public RotationData target_rotation;
        public string gripper_state; // "open", "closed", "unknown"
        public bool is_moving;
        public bool is_initialized;
        public float[] joint_angles;
        public float[] start_joint_angles; // Saved at registration time; used by ROS return-to-start
        public string control_mode; // "unity", "ros", "hybrid" (null if no ROSControlModeManager)
    }

    [System.Serializable]
    public class PositionData
    {
        public float x;
        public float y;
        public float z;

        public PositionData(Vector3 v)
        {
            x = v.x;
            y = v.y;
            z = v.z;
        }
    }

    [System.Serializable]
    public class RotationData
    {
        public float x;
        public float y;
        public float z;
        public float w;

        public RotationData(Quaternion q)
        {
            x = q.x;
            y = q.y;
            z = q.z;
            w = q.w;
        }
    }

    [System.Serializable]
    public class ObjectStateData
    {
        public string object_id;
        public PositionData position;
        public PositionData dimensions; // Object size (x=width, y=height, z=depth) in meters
        public RotationData rotation;
        public string color;
        public string object_type;
        public float confidence;
    }

    /// <summary>
    /// Periodically publishes robot and object states to Python's WorldState system.
    ///
    /// This component enables spatial reasoning operations on the Python side by providing:
    /// - Robot positions and targets
    /// - Gripper states
    /// - Movement status
    /// - Detected object positions
    ///
    /// Usage:
    /// 1. Attach to a GameObject in your scene
    /// 2. Configure update rate (default 2Hz - every 0.5s)
    /// 3. Optionally track specific objects in the scene
    /// </summary>
    public class WorldStatePublisher : MonoBehaviour
    {
        public static WorldStatePublisher Instance { get; private set; }

        [Header("Configuration")]
        [Tooltip("Enable world state publishing")]
        [SerializeField]
        private bool _enablePublishing = true;

        [Tooltip("Update rate in Hz (updates per second). Mirrors 1.0/WORLD_STATE_UPDATE_INTERVAL in ACRLPython/config/Robot.py (default: 10 Hz = 0.1s interval).")]
        [SerializeField]
        [Range(0.1f, 10f)]
        private float _updateRate = 10.0f;

        [Tooltip("Publish robot states")]
        [SerializeField]
        private bool _publishRobots = true;

        [Tooltip("Publish object states")]
        [SerializeField]
        private bool _publishObjects = true;

        [Tooltip("Enable verbose logging")]
        [SerializeField]
        private bool _verboseLogging = false;

        [Header("Object Tracking")]
        [Tooltip("GameObjects to track as objects (cubes, spheres, etc.)")]
        [SerializeField]
        private List<GameObject> _trackedObjects = new List<GameObject>();

        [Header("Runtime Info")]
        [SerializeField]
        private int _updatesSent = 0;

        [SerializeField]
        private float _lastUpdateTime = 0f;

        private RobotManager _robotManager;
        private float _updateInterval;
        private float _timeSinceLastUpdate = 0f;

        // Helper variable
        private const string _logPrefix = "[WORLD_STATE_PUBLISHER]";

        // Cache detected objects from vision system
        private Dictionary<string, ObjectStateData> _detectedObjects =
            new Dictionary<string, ObjectStateData>();

        // Pre-allocated collections to avoid per-frame GC allocations
        private readonly List<RobotStateData> _robotStates = new List<RobotStateData>();
        private readonly List<ObjectStateData> _objectStates = new List<ObjectStateData>();

        // Pre-allocated joint angle buffer sized for AR4 (6 joints); grows if needed
        private float[] _jointAnglesCache = new float[6];

        #region Unity Lifecycle

        /// <summary>
        /// Initialize singleton
        /// </summary>
        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
            }
            else
            {
                Destroy(gameObject);
                return;
            }
        }

        /// <summary>
        /// Initialize components and calculate update interval
        /// </summary>
        private void Start()
        {
            _robotManager = RobotManager.Instance;
            if (_robotManager == null)
            {
                Debug.LogError(
                    $"{_logPrefix} RobotManager.Instance is null! "
                        + "Ensure RobotManager GameObject is in the scene."
                );
                enabled = false;
                return;
            }

            _updateInterval = 1.0f / _updateRate;

            if (_enablePublishing)
            {
                Debug.Log(
                    $"{_logPrefix} Initialized. Publishing at {_updateRate:F1} Hz (every {_updateInterval:F2}s)"
                );
            }
            else
            {
                Debug.Log($"{_logPrefix} Initialized but publishing is disabled");
            }
        }

        /// <summary>
        /// Periodic update - publish world state at configured rate
        /// </summary>
        private void Update()
        {
            if (!_enablePublishing)
                return;

            _timeSinceLastUpdate += Time.deltaTime;

            if (_timeSinceLastUpdate >= _updateInterval)
            {
                PublishWorldState();
                _timeSinceLastUpdate = 0f;
            }
        }

        #endregion

        #region World State Publishing

        /// <summary>
        /// Gather and publish current world state to Python.
        /// Uses dedicated WorldStateClient (port 5014) to avoid message correlation
        /// conflicts with command/response traffic on port 5010.
        /// </summary>
        private void PublishWorldState()
        {
            try
            {
                // Reuse pre-allocated lists — clear in-place instead of allocating new ones
                _robotStates.Clear();
                _objectStates.Clear();

                if (_publishRobots)
                    GatherRobotStates(_robotStates);
                if (_publishObjects)
                    GatherObjectStates(_objectStates);

                var update = new WorldStateUpdate
                {
                    robots = _robotStates,
                    objects = _objectStates,
                    timestamp = Time.time,
                };

                // Convert to JSON and send via dedicated WorldStateClient
                string json = JsonUtility.ToJson(update);

                // Use dedicated WorldStateClient (port 5014) instead of ResultsClient (port 5010)
                // This prevents unsolicited broadcasts from interfering with command request/response correlation
                if (WorldStateClient.Instance != null)
                {
                    bool sent = WorldStateClient.Instance.PublishWorldState(json);

                    if (sent)
                    {
                        _updatesSent++;
                        _lastUpdateTime = Time.time;

                        if (_verboseLogging)
                        {
                            Debug.Log(
                                $"{_logPrefix} Published world state: {update.robots.Count} robots, "
                                    + $"{update.objects.Count} objects"
                            );
                        }
                    }
                }
                else
                {
                    if (_verboseLogging)
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} WorldStateClient not available (Python WorldStateServer may not be running)"
                        );
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error publishing world state: {ex.Message}\n{ex.StackTrace}"
                );
            }
        }

        /// <summary>
        /// Gather current state of all robots into the supplied list (cleared by caller).
        /// </summary>
        private void GatherRobotStates(List<RobotStateData> robotStates)
        {
            foreach (var kvp in _robotManager.RobotInstances)
            {
                string robotId = kvp.Key;
                RobotInstance robotInstance = kvp.Value;
                RobotController controller = robotInstance.controller;

                if (controller == null)
                    continue;

                // Gather robot state
                Vector3 position =
                    controller.endEffectorBase != null
                        ? controller.endEffectorBase.position
                        : Vector3.zero;

                Quaternion rotation =
                    controller.endEffectorBase != null
                        ? controller.endEffectorBase.rotation
                        : Quaternion.identity;

                Vector3 targetPosition = controller.GetCurrentTarget() ?? Vector3.zero;
                Quaternion? targetRotation = controller.GetCurrentTargetRotation();
                float distanceToTarget = controller.GetDistanceToTarget();
                bool isMoving = distanceToTarget > RobotConstants.MOVEMENT_THRESHOLD;

                // Get gripper state
                string gripperState = "unknown";
                var gripperController =
                    robotInstance.robotGameObject.GetComponentInChildren<GripperController>();
                if (gripperController != null)
                {
                    gripperState = gripperController.targetPosition > 0.9f ? "open" : "closed";
                }

                // Get joint angles into pre-allocated cache (grows if robot has more than 6 joints)
                float[] jointAngles = GatherJointAngles(controller);

                // Get ROS control mode if available — ToString().ToLower() only called when mode is present
                string controlMode = null;
                var rosControlMode = controller.GetComponent<ROSControlModeManager>();
                if (rosControlMode != null)
                {
                    controlMode = rosControlMode.CurrentMode.ToString().ToLower();
                }

                // Convert startJointTargets (degrees, Unity) → radians for ROS consumers
                float[] startJointAnglesRad = null;
                if (robotInstance.startJointTargets != null && robotInstance.startJointTargets.Length > 0)
                {
                    startJointAnglesRad = new float[robotInstance.startJointTargets.Length];
                    for (int i = 0; i < robotInstance.startJointTargets.Length; i++)
                        startJointAnglesRad[i] = robotInstance.startJointTargets[i] * Mathf.Deg2Rad;
                }

                var robotState = new RobotStateData
                {
                    robot_id = robotId,
                    position = new PositionData(position),
                    rotation = new RotationData(rotation),
                    target_position = new PositionData(targetPosition),
                    target_rotation = targetRotation.HasValue ? new RotationData(targetRotation.Value) : null,
                    gripper_state = gripperState,
                    is_moving = isMoving,
                    is_initialized = true,
                    joint_angles = jointAngles,
                    start_joint_angles = startJointAnglesRad,
                    control_mode = controlMode,
                };

                robotStates.Add(robotState);
            }
        }

        /// <summary>
        /// Gather joint angles from robot controller into the pre-allocated cache array.
        /// Returns the cached array (resized only when joint count exceeds current capacity).
        /// </summary>
        private float[] GatherJointAngles(RobotController controller)
        {
            if (controller.robotJoints == null)
                return new float[0];

            int count = controller.robotJoints.Length;

            // Grow the cache only when needed — avoids allocation in the common case
            if (_jointAnglesCache.Length < count)
                _jointAnglesCache = new float[count];

            for (int i = 0; i < count; i++)
            {
                var joint = controller.robotJoints[i];
                _jointAnglesCache[i] = (joint != null && joint.jointType == ArticulationJointType.RevoluteJoint)
                    ? joint.jointPosition[0]
                    : 0f;
            }

            return _jointAnglesCache;
        }

        /// <summary>
        /// Gather current state of all tracked objects into the supplied list (cleared by caller).
        /// </summary>
        private void GatherObjectStates(List<ObjectStateData> objectStates)
        {
            // Add tracked GameObjects from scene
            foreach (var obj in _trackedObjects)
            {
                if (obj == null)
                    continue;

                // Infer dimensions from collider bounds; fall back to uniform 1m cube
                var col = obj.GetComponent<Collider>();
                Vector3 size = col != null ? col.bounds.size : Vector3.one;

                var objectState = new ObjectStateData
                {
                    object_id = obj.name,
                    position = new PositionData(obj.transform.position),
                    dimensions = new PositionData(size),
                    rotation = new RotationData(obj.transform.rotation),
                    color = InferColorFromName(obj.name),
                    object_type = InferTypeFromName(obj.name),
                    confidence = 1.0f,
                };

                objectStates.Add(objectState);
            }

            // Add detected objects from vision system
            foreach (var kvp in _detectedObjects)
            {
                objectStates.Add(kvp.Value);
            }
        }

        /// <summary>
        /// Infer object color from GameObject name.
        /// Uses IndexOf with OrdinalIgnoreCase to avoid the ToLower() string allocation.
        /// </summary>
        private string InferColorFromName(string name)
        {
            string lower = name.ToLowerInvariant();
            
            // Prevent false positives from words containing "red"
            lower = lower.Replace("shared", "")
                         .Replace("desired", "")
                         .Replace("colored", "")
                         .Replace("ignored", "")
                         .Replace("measured", "");

            if (lower.Contains("red")) return "red";
            if (lower.Contains("blue")) return "blue";
            if (lower.Contains("green")) return "green";
            if (lower.Contains("yellow")) return "yellow";
            if (lower.Contains("orange")) return "orange";
            if (lower.Contains("purple")) return "purple";
            if (lower.Contains("black")) return "black";
            if (lower.Contains("white")) return "white";
            
            return "unknown";
        }

        /// <summary>
        /// Infer object type from GameObject name.
        /// Uses IndexOf with OrdinalIgnoreCase to avoid the ToLower() string allocation.
        /// </summary>
        private string InferTypeFromName(string name)
        {
            if (name.IndexOf("cube", StringComparison.OrdinalIgnoreCase) >= 0)
                return "cube";
            if (name.IndexOf("sphere", StringComparison.OrdinalIgnoreCase) >= 0)
                return "sphere";
            if (name.IndexOf("cylinder", StringComparison.OrdinalIgnoreCase) >= 0)
                return "cylinder";
            return "unknown";
        }

        #endregion

        #region Public API

        /// <summary>
        /// Register a detected object from vision system
        /// </summary>
        /// <param name="objectId">Unique object identifier</param>
        /// <param name="position">Object position in world coordinates</param>
        /// <param name="color">Object color</param>
        /// <param name="objectType">Object type</param>
        /// <param name="confidence">Detection confidence (0-1)</param>
        /// <param name="rotation">Optional object rotation (null if unknown)</param>
        /// <param name="dimensions">Optional object size (x=width, y=height, z=depth) in meters</param>
        public void RegisterDetectedObject(
            string objectId,
            Vector3 position,
            string color = "unknown",
            string objectType = "cube",
            float confidence = 1.0f,
            Quaternion? rotation = null,
            Vector3? dimensions = null
        )
        {
            var objectState = new ObjectStateData
            {
                object_id = objectId,
                position = new PositionData(position),
                dimensions = dimensions.HasValue ? new PositionData(dimensions.Value) : null,
                rotation = rotation.HasValue ? new RotationData(rotation.Value) : null,
                color = color,
                object_type = objectType,
                confidence = confidence,
            };

            _detectedObjects[objectId] = objectState;

            if (_verboseLogging)
            {
                Debug.Log(
                    $"{_logPrefix} Registered detected object: {objectId} at ({position.x:F3}, {position.y:F3}, {position.z:F3})"
                );
            }
        }

        /// <summary>
        /// Clear detected objects cache
        /// </summary>
        public void ClearDetectedObjects()
        {
            _detectedObjects.Clear();
            if (_verboseLogging)
            {
                Debug.Log($"{_logPrefix} Cleared detected objects cache");
            }
        }

        /// <summary>
        /// Add GameObject to tracked objects list
        /// </summary>
        public void AddTrackedObject(GameObject obj)
        {
            if (obj != null && !_trackedObjects.Contains(obj))
            {
                _trackedObjects.Add(obj);
                if (_verboseLogging)
                {
                    Debug.Log($"{_logPrefix} Added tracked object: {obj.name}");
                }
            }
        }

        /// <summary>
        /// Remove GameObject from tracked objects list
        /// </summary>
        public void RemoveTrackedObject(GameObject obj)
        {
            if (_trackedObjects.Remove(obj))
            {
                if (_verboseLogging)
                {
                    Debug.Log($"{_logPrefix} Removed tracked object: {obj.name}");
                }
            }
        }

        /// <summary>
        /// Force immediate world state publish (outside normal update cycle)
        /// </summary>
        public void PublishNow()
        {
            if (_enablePublishing)
            {
                PublishWorldState();
            }
        }

        /// <summary>
        /// Enable or disable publishing
        /// </summary>
        public void SetPublishingEnabled(bool enabled)
        {
            _enablePublishing = enabled;
            Debug.Log($"{_logPrefix} Publishing {(enabled ? "enabled" : "disabled")}");
        }

        /// <summary>
        /// Get publishing statistics
        /// </summary>
        public (int updatesSent, float lastUpdateTime) GetStats()
        {
            return (_updatesSent, _lastUpdateTime);
        }

        #endregion
    }
}
