using System;
using System.Collections.Generic;
using System.Linq;
using Configuration;
using Logging;
using UnityEngine;

namespace Robotics
{
    [System.Serializable]
    public class RobotInstance
    {
        public string robotId;
        public GameObject robotGameObject;
        public GameObject targetGameObject;
        public RobotController controller;
        public RobotConfig profile;
        public bool isActive;
        public float lastTargetChangeTime;
        public Vector3 lastTargetPosition;
        public float[] startJointTargets;
    }

    public class RobotManager : MonoBehaviour
    {
        public static RobotManager Instance { get; private set; }

        [Header("Robot Profiles")]
        [SerializeField]
        public RobotConfig robotProfile;

        // Configuration values for an AR4 robotic arm

        [Header("Global Settings")]
        [SerializeField, Range(0.1f, 5f)]
        public float globalSpeedMultiplier = 1.0f;

        [SerializeField]
        private bool _enableTargetChangeDetection = true;

        [SerializeField]
        private float _targetChangeCheckInterval = 0.1f;

        [Header("Debug Settings")]
        [SerializeField]
        private bool _logConfigurationChanges = true;

        // Core components
        private MainLogger _logger;

        // Robot management
        private Dictionary<string, RobotInstance> _robotInstances =
            new Dictionary<string, RobotInstance>();
        private float _nextTargetCheckTime;
        private int _nextRobotIdCounter = 1;
        private HashSet<string> _usedRobotIds = new HashSet<string>();

        // Events
        public event System.Action<string, GameObject> OnTargetChanged;

        // Properties
        public IReadOnlyDictionary<string, RobotInstance> RobotInstances => _robotInstances;
        public int ActiveRobotCount => _robotInstances.Values.Count(r => r.isActive);
        public RobotConfig RobotProfile => robotProfile;

        // Helper variables
        private const string _logPrefix = "[ROBOT_MANAGER]";

        /// <summary>
        /// Gets a list of all registered robot IDs.
        /// </summary>
        public List<string> AllRobotIds => new List<string>(_robotInstances.Keys);

        /// <summary>
        /// Gets an array of tuples containing robot IDs and their parent GameObjects.
        /// </summary>
        public (string id, GameObject parentGameObject)[] Robots
        {
            get
            {
                var robotArray = new (string, GameObject)[_robotInstances.Count];
                int index = 0;
                foreach (var kvp in _robotInstances)
                {
                    robotArray[index] = (kvp.Key, kvp.Value.robotGameObject);
                    index++;
                }
                return robotArray;
            }
        }

        /// <summary>
        /// Unity Awake callback - initializes singleton instance.
        /// </summary>
        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                InitializeManager();
            }
            else
            {
                Destroy(gameObject);
            }
        }

        /// <summary>
        /// Initializes the RobotManager and creates default robot profile if needed.
        /// </summary>
        private void InitializeManager()
        {
            try
            {
                // Create default profile if not assigned in inspector
                if (robotProfile == null)
                {
                    robotProfile = ScriptableObject.CreateInstance<RobotConfig>();
                }

                // Ensure default profile is properly initialized
                if (robotProfile.joints == null || robotProfile.joints.Length == 0)
                {
                    robotProfile.InitializeDefaultAR4Profile();
                }

                _nextTargetCheckTime = Time.time + _targetChangeCheckInterval;
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Failed to initialize: {ex.Message}");
            }
        }

        /// <summary>
        /// Unity Start callback - initializes component references and discovers robots in the scene.
        /// </summary>
        private void Start()
        {
            try
            {
                // Get component references
                _logger = MainLogger.Instance;

                // Auto-discover robots
                DiscoverRobots();

                // Log initialization
                Debug.Log($"{_logPrefix} Initialized with {_robotInstances.Count} robots");

                LogConfigurationSummary();
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Failed to start: {ex.Message}");
            }
        }

        /// <summary>
        /// Automatically discovers all RobotController components in the scene and registers them.
        /// </summary>
        private void DiscoverRobots()
        {
            RobotController[] controllers = FindObjectsByType<RobotController>(
                FindObjectsInactive.Exclude,
                FindObjectsSortMode.None
            );

            foreach (var controller in controllers)
            {
                // Generate unique ID for this robot
                string robotId = GenerateUniqueRobotId(controller.gameObject.name);

                if (!_robotInstances.ContainsKey(robotId))
                {
                    RegisterRobot(robotId, controller.gameObject);
                    Debug.Log($"{_logPrefix} Auto-discovered robot: {robotId} with no target.");
                }
            }
        }

        /// <summary>
        /// Unity Update callback - checks for target position changes at regular intervals.
        /// </summary>
        private void Update()
        {
            if (_enableTargetChangeDetection && Time.time >= _nextTargetCheckTime)
            {
                CheckForTargetChanges();
                _nextTargetCheckTime = Time.time + _targetChangeCheckInterval;
            }
        }

        /// <summary>
        /// Checks all registered robots for target position changes and updates controllers accordingly.
        /// </summary>
        private void CheckForTargetChanges()
        {
            foreach (var robotEntry in _robotInstances)
            {
                var robot = robotEntry.Value;
                if (robot.targetGameObject == null || !robot.isActive)
                    continue;

                Vector3 currentPos = robot.targetGameObject.transform.position;
                if (Vector3.Distance(currentPos, robot.lastTargetPosition) > 0.001f)
                {
                    robot.lastTargetPosition = currentPos;
                    robot.lastTargetChangeTime = Time.time;

                    // Update robot target
                    if (robot.controller != null)
                    {
                        robot.controller.SetTarget(robot.targetGameObject);
                        OnTargetChanged?.Invoke(robot.robotId, robot.targetGameObject);

                        if (_logConfigurationChanges && _logger != null)
                        {
                            string actionId = _logger.StartAction(
                                actionName: "target_updated",
                                type: Logging.ActionType.Movement,
                                robotIds: new[] { robot.robotId },
                                targetPos: currentPos,
                                objectIds: new[] { robot.targetGameObject.name },
                                description: $"Target updated to {robot.targetGameObject.name}"
                            );
                            _logger.CompleteAction(actionId, success: true, qualityScore: 1f);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Generates a unique robot ID based on a base name. If the ID already exists, appends a counter.
        /// </summary>
        /// <param name="baseName">The base name to use for ID generation (typically GameObject name)</param>
        /// <returns>A unique robot ID</returns>
        private string GenerateUniqueRobotId(string baseName)
        {
            // Clean the base name (remove any existing counters)
            string cleanBaseName = baseName;

            // If ID is unique, use it directly
            if (!_usedRobotIds.Contains(cleanBaseName))
            {
                _usedRobotIds.Add(cleanBaseName);
                return cleanBaseName;
            }

            // Otherwise, append a counter
            string uniqueId;
            do
            {
                uniqueId = $"{cleanBaseName}_{_nextRobotIdCounter}";
                _nextRobotIdCounter++;
            } while (_usedRobotIds.Contains(uniqueId));

            _usedRobotIds.Add(uniqueId);
            return uniqueId;
        }

        /// <summary>
        /// Registers a robot with the RobotManager and applies its configuration profile.
        /// If robotId is null or empty, a unique ID will be generated automatically.
        /// </summary>
        /// <param name="robotId">Unique identifier for the robot (auto-generated if null/empty)</param>
        /// <param name="robotObject">The robot's GameObject containing RobotController</param>
        /// <param name="targetObject">Optional target GameObject for the robot to track</param>
        /// <param name="profile">Optional custom RobotConfig profile, uses default if null</param>
        public void RegisterRobot(
            string robotId,
            GameObject robotObject,
            GameObject targetObject = null,
            RobotConfig profile = null
        )
        {
            try
            {
                // Generate unique ID if not provided
                if (string.IsNullOrEmpty(robotId))
                {
                    robotId = GenerateUniqueRobotId(robotObject.name);
                    Debug.Log($"{_logPrefix} Auto-generated robot ID: {robotId}");
                }
                else if (!_usedRobotIds.Contains(robotId))
                {
                    // Mark provided ID as used
                    _usedRobotIds.Add(robotId);
                }

                if (_robotInstances.ContainsKey(robotId))
                {
                    Debug.LogWarning(
                        $"{_logPrefix} Robot {robotId} already registered. Updating configuration."
                    );
                }

                var controller = robotObject.GetComponent<RobotController>();
                if (controller == null)
                {
                    Debug.LogError(
                        $"{_logPrefix} Robot {robotId} missing RobotController component"
                    );
                    return;
                }

                var instance = new RobotInstance
                {
                    robotId = robotId,
                    robotGameObject = robotObject,
                    targetGameObject = targetObject,
                    controller = controller,
                    profile = profile ?? robotProfile,
                    isActive = true,
                    lastTargetChangeTime = Time.time,
                    lastTargetPosition = targetObject?.transform.position ?? Vector3.zero,
                    startJointTargets = (float[])controller.jointDriveTargets.Clone(),
                };

                _robotInstances[robotId] = instance;

                // Apply configuration to robot
                ApplyProfileToRobot(robotId);

                // Set target on the controller if available
                if (targetObject != null && controller != null)
                {
                    controller.SetTarget(targetObject);
                    Debug.Log(
                        $"{_logPrefix} Assigned target '{targetObject.name}' to robot '{robotId}'"
                    );
                }

                string profileName =
                    instance.profile != null ? instance.profile.profileName : "default";

                Debug.Log($"{_logPrefix} Registered robot: {robotId} with profile: {profileName}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Failed to register robot {robotId}: {ex.Message}");
            }
        }

        /// <summary>
        /// Checks if a robot is already registered.
        /// </summary>
        /// <param name="robotId">The robot identifier to check</param>
        /// <returns>True if the robot is registered, false otherwise</returns>
        public bool IsRobotRegistered(string robotId)
        {
            return _robotInstances.ContainsKey(robotId);
        }

        /// <summary>
        /// Unregisters a robot from the RobotManager.
        /// </summary>
        /// <param name="robotId">The robot identifier to unregister</param>
        public void UnregisterRobot(string robotId)
        {
            if (_robotInstances.Remove(robotId))
            {
                _usedRobotIds.Remove(robotId);
                Debug.Log($"{_logPrefix} Robot {robotId} unregistered");
            }
        }

        /// <summary>
        /// Sets the target GameObject for a specific robot.
        /// </summary>
        /// <param name="robotId">The robot identifier</param>
        /// <param name="target">The target GameObject to assign</param>
        public void SetRobotTarget(string robotId, GameObject target)
        {
            if (_robotInstances.TryGetValue(robotId, out RobotInstance robot))
            {
                robot.targetGameObject = target;
                robot.lastTargetPosition = target?.transform.position ?? Vector3.zero;
                robot.lastTargetChangeTime = Time.time;

                if (robot.controller != null && target != null)
                {
                    robot.controller.SetTarget(target);
                    OnTargetChanged?.Invoke(robotId, target);
                }
            }
        }

        /// <summary>
        /// Sets the target for a robot using a position coordinate.
        /// Creates a temporary target object at the specified position.
        /// </summary>
        /// <param name="robotId">The robot identifier</param>
        /// <param name="position">The target position in world coordinates</param>
        public void SetRobotTarget(string robotId, Vector3 position)
        {
            if (_robotInstances.TryGetValue(robotId, out RobotInstance robot))
            {
                if (robot.controller != null)
                {
                    robot.controller.SetTarget(position);

                    // Update robot instance tracking
                    robot.targetGameObject = GameObject.Find($"{robotId}_TempTarget");
                    robot.lastTargetPosition = position;
                    robot.lastTargetChangeTime = Time.time;

                    if (robot.targetGameObject != null)
                    {
                        OnTargetChanged?.Invoke(robotId, robot.targetGameObject);
                    }
                }
            }
        }

        /// <summary>
        /// Sets the target for a robot using position and rotation coordinates.
        /// Creates a temporary target object at the specified position and rotation.
        /// </summary>
        /// <param name="robotId">The robot identifier</param>
        /// <param name="position">The target position in world coordinates</param>
        /// <param name="rotation">The target rotation in world coordinates</param>
        public void SetRobotTarget(string robotId, Vector3 position, Quaternion rotation)
        {
            if (_robotInstances.TryGetValue(robotId, out RobotInstance robot))
            {
                if (robot.controller != null)
                {
                    robot.controller.SetTarget(position, rotation);

                    // Update robot instance tracking
                    robot.targetGameObject = GameObject.Find($"{robotId}_TempTarget");
                    robot.lastTargetPosition = position;
                    robot.lastTargetChangeTime = Time.time;

                    if (robot.targetGameObject != null)
                    {
                        OnTargetChanged?.Invoke(robotId, robot.targetGameObject);
                    }
                }
            }
        }

        /// <summary>
        /// Applies the robot's configuration profile to all its joints.
        /// </summary>
        /// <param name="robotId">The robot identifier</param>
        private void ApplyProfileToRobot(string robotId)
        {
            if (!_robotInstances.TryGetValue(robotId, out RobotInstance robot))
                return;

            try
            {
                var controller = robot.controller;
                if (
                    controller == null
                    || controller.robotJoints == null
                    || robot.profile == null
                    || robot.profile.joints == null
                )
                    return;

                // Validate joint count matches
                int jointCount = Mathf.Min(
                    controller.robotJoints.Length,
                    robot.profile.joints.Length
                );
                if (controller.robotJoints.Length != robot.profile.joints.Length)
                {
                    Debug.LogWarning(
                        $"{_logPrefix} Joint count mismatch for {robotId}: Controller has {controller.robotJoints.Length}, "
                            + $"Profile has {robot.profile.joints.Length}. Applying to first {jointCount} joints."
                    );
                }

                // Apply joint configurations
                for (int i = 0; i < jointCount; i++)
                {
                    var joint = controller.robotJoints[i];
                    var config = robot.profile.joints[i];
                    var drive = joint.xDrive;

                    drive.stiffness = config.stiffness;
                    drive.damping = config.damping;
                    drive.forceLimit = config.forceLimit;
                    drive.upperLimit = config.upperLimit;
                    drive.lowerLimit = config.lowerLimit;

                    joint.xDrive = drive;
                }

                if (_logConfigurationChanges && _logger != null)
                {
                    string actionId = _logger.StartAction(
                        actionName: "configuration_applied",
                        type: Logging.ActionType.Task,
                        robotIds: new[] { robotId },
                        objectIds: new[] { robot.profile.profileName },
                        description: $"Applied configuration {robot.profile.profileName} to {robotId}"
                    );
                    _logger.CompleteAction(actionId, success: true, qualityScore: 1f);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Failed to apply profile to robot {robotId}: {ex.Message}"
                );
            }
        }

        /// <summary>
        /// Gets the profile for a specific robot by robotId
        /// </summary>
        /// <param name="robotId">The robot identifier</param>
        /// <returns>The robot's profile or null if robot not found</returns>
        public RobotConfig GetRobotProfile(string robotId)
        {
            return _robotInstances.TryGetValue(robotId, out RobotInstance robot)
                ? robot.profile
                : null;
        }

        /// <summary>
        /// Logs a summary of all robot configurations to the file logger and console.
        /// </summary>
        private void LogConfigurationSummary()
        {
            if (!_logConfigurationChanges)
                return;

            string summary = $"RobotManager Configuration Summary:\n";
            summary += $"- Total Robots: {_robotInstances.Count}\n";
            summary += $"- Global Speed Multiplier: {globalSpeedMultiplier}\n";
            summary += $"- Default Profile: {robotProfile.profileName}\n";

            foreach (var robot in _robotInstances.Values)
            {
                summary +=
                    $"- {robot.robotId}: Profile '{robot.profile.profileName}', Active: {robot.isActive}\n";
            }

            Debug.Log($"{_logPrefix} Configuration summary:\n{summary}");
        }

        /// <summary>
        /// Unity OnDestroy callback - cleans up singleton instance.
        /// </summary>
        private void OnDestroy()
        {
            if (Instance == this)
            {
                Debug.Log($"{_logPrefix} destroyed");
                Instance = null;
            }
        }
    }
}
