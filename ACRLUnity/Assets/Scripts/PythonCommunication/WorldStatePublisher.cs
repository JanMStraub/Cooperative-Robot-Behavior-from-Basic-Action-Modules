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
        public string gripper_state; // "open", "closed", "unknown"
        public bool is_moving;
        public bool is_initialized;
        public float[] joint_angles;
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

        [Tooltip("Update rate in Hz (updates per second)")]
        [SerializeField]
        [Range(0.1f, 10f)]
        private float _updateRate = 2.0f;

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
                var update = new WorldStateUpdate
                {
                    robots = _publishRobots ? GatherRobotStates() : new List<RobotStateData>(),
                    objects = _publishObjects ? GatherObjectStates() : new List<ObjectStateData>(),
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
        /// Gather current state of all robots
        /// </summary>
        private List<RobotStateData> GatherRobotStates()
        {
            var robotStates = new List<RobotStateData>();

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
                float distanceToTarget = controller.GetDistanceToTarget();
                bool isMoving = distanceToTarget > RobotConstants.MOVEMENT_THRESHOLD;

                // Get gripper state
                string gripperState = "unknown";
                var gripperController =
                    robotInstance.robotGameObject.GetComponentInChildren<GripperController>();
                if (gripperController != null)
                {
                    if (gripperController.targetPosition > 0.9f)
                    {
                        gripperState = "open";
                    }
                    else
                    {
                        gripperState = "closed";
                    }
                }

                // Get joint angles
                float[] jointAngles = GatherJointAngles(controller);

                // Get ROS control mode if available
                string controlMode = null;
                var rosControlMode = controller.GetComponent<ROSControlModeManager>();
                if (rosControlMode != null)
                {
                    controlMode = rosControlMode.CurrentMode.ToString().ToLower();
                }

                var robotState = new RobotStateData
                {
                    robot_id = robotId,
                    position = new PositionData(position),
                    rotation = new RotationData(rotation),
                    target_position = new PositionData(targetPosition),
                    gripper_state = gripperState,
                    is_moving = isMoving,
                    is_initialized = true,
                    joint_angles = jointAngles,
                    control_mode = controlMode,
                };

                robotStates.Add(robotState);
            }

            return robotStates;
        }

        /// <summary>
        /// Gather joint angles from robot controller
        /// </summary>
        private float[] GatherJointAngles(RobotController controller)
        {
            if (controller.robotJoints == null)
                return new float[0];

            float[] jointAngles = new float[controller.robotJoints.Length];
            for (int i = 0; i < controller.robotJoints.Length; i++)
            {
                var joint = controller.robotJoints[i];
                if (joint != null && joint.jointType == ArticulationJointType.RevoluteJoint)
                {
                    jointAngles[i] = joint.jointPosition[0];
                }
            }

            return jointAngles;
        }

        /// <summary>
        /// Gather current state of all tracked objects
        /// </summary>
        private List<ObjectStateData> GatherObjectStates()
        {
            var objectStates = new List<ObjectStateData>();

            // Add tracked GameObjects from scene
            foreach (var obj in _trackedObjects)
            {
                if (obj == null)
                    continue;

                var objectState = new ObjectStateData
                {
                    object_id = obj.name,
                    position = new PositionData(obj.transform.position),
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

            return objectStates;
        }

        /// <summary>
        /// Infer object color from GameObject name
        /// </summary>
        private string InferColorFromName(string name)
        {
            string nameLower = name.ToLower();
            if (nameLower.Contains("red"))
                return "red";
            if (nameLower.Contains("blue"))
                return "blue";
            if (nameLower.Contains("green"))
                return "green";
            if (nameLower.Contains("yellow"))
                return "yellow";
            return "unknown";
        }

        /// <summary>
        /// Infer object type from GameObject name
        /// </summary>
        private string InferTypeFromName(string name)
        {
            string nameLower = name.ToLower();
            if (nameLower.Contains("cube"))
                return "cube";
            if (nameLower.Contains("sphere"))
                return "sphere";
            if (nameLower.Contains("cylinder"))
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
        public void RegisterDetectedObject(
            string objectId,
            Vector3 position,
            string color = "unknown",
            string objectType = "cube",
            float confidence = 1.0f
        )
        {
            var objectState = new ObjectStateData
            {
                object_id = objectId,
                position = new PositionData(position),
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
