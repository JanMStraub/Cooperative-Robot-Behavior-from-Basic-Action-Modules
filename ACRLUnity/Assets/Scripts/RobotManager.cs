using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

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
    private FileLogger _fileLogger;
    private RobotActionLogger _robotActionLogger;

    // Robot management
    private Dictionary<string, RobotInstance> _robotInstances =
        new Dictionary<string, RobotInstance>();
    private float _nextTargetCheckTime;

    // Events
    public event System.Action<string, GameObject> OnTargetChanged;

    // Properties
    public IReadOnlyDictionary<string, RobotInstance> RobotInstances => _robotInstances;
    public int ActiveRobotCount => _robotInstances.Values.Count(r => r.isActive);
    public RobotConfig RobotProfile => robotProfile;

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
            Debug.LogError($"Failed to initialize RobotManager: {ex.Message}");
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
            _fileLogger = FileLogger.Instance;
            _robotActionLogger = RobotActionLogger.Instance;

            // Auto-discover robots
            DiscoverRobots();

            // Log initialization
            _fileLogger?.LogSimulationEvent(
                "robot_manager_initialized",
                $"Initialized with {_robotInstances.Count} robots"
            );

            LogConfigurationSummary();
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to start RobotManager: {ex.Message}");
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
            string robotId = controller.gameObject.name;

            if (!_robotInstances.ContainsKey(robotId))
            {
                // Try to find a target for this robot
                GameObject target = FindTargetForRobot(robotId);

                RegisterRobot(robotId, controller.gameObject, target);
                Debug.Log(
                    $"Auto-discovered robot: {robotId}"
                        + (target != null ? $" with target: {target.name}" : " (no target found)")
                );
            }
        }
    }

    /// <summary>
    /// Attempts to find a target GameObject for a specific robot.
    /// Searches for objects named "{RobotId}Target" or "Target_{RobotId}" or just "Target"
    /// </summary>
    /// <param name="robotId">The robot identifier to find a target for</param>
    /// <returns>The target GameObject if found, null otherwise</returns>
    private GameObject FindTargetForRobot(string robotId)
    {
        // Search patterns: "AR4LeftTarget", "Target_AR4Left", "AR4Left_Target"
        string[] patterns = new string[]
        {
            $"{robotId}Target",
            $"Target_{robotId}",
            $"{robotId}_Target",
        };

        // Add pattern without "AR4" prefix if it exists
        if (robotId.Contains("AR4"))
        {
            string nameWithoutAR4 = robotId.Replace("AR4", "");
            patterns = patterns.Append($"{nameWithoutAR4}Target").ToArray();
        }

        foreach (var pattern in patterns)
        {
            GameObject found = GameObject.Find(pattern);
            if (found != null)
            {
                Debug.Log($"Found target '{found.name}' for robot '{robotId}'");
                return found;
            }
        }

        // Fallback: search for any object with "Target" in name near the robot
        GameObject[] allObjects = FindObjectsByType<GameObject>(FindObjectsSortMode.None);
        foreach (var obj in allObjects)
        {
            if (
                obj.name.ToLower().Contains("target")
                && obj.name.Contains(robotId.Replace("AR4", ""))
            )
            {
                Debug.Log($"Found target '{obj.name}' for robot '{robotId}' (fallback search)");
                return obj;
            }
        }

        Debug.LogWarning(
            $"No target found for robot '{robotId}'. Robot will not move until target is assigned."
        );
        return null;
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

                    if (_logConfigurationChanges)
                    {
                        _robotActionLogger?.LogAction(
                            "target_updated",
                            robot.robotId,
                            robot.targetGameObject.name,
                            currentPos,
                            null,
                            0f,
                            true
                        );
                    }
                }
            }
        }
    }

    /// <summary>
    /// Registers a robot with the RobotManager and applies its configuration profile.
    /// </summary>
    /// <param name="robotId">Unique identifier for the robot</param>
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
            if (_robotInstances.ContainsKey(robotId))
            {
                Debug.LogWarning($"Robot {robotId} already registered. Updating configuration.");
            }

            var controller = robotObject.GetComponent<RobotController>();
            if (controller == null)
            {
                Debug.LogError($"Robot {robotId} missing RobotController component");
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
            };

            _robotInstances[robotId] = instance;

            // Apply configuration to robot
            ApplyProfileToRobot(robotId);

            // Set target on the controller if available
            if (targetObject != null && controller != null)
            {
                controller.SetTarget(targetObject);
                Debug.Log($"Assigned target '{targetObject.name}' to robot '{robotId}'");
            }

            string profileName =
                instance.profile != null ? instance.profile.profileName : "default";

            _fileLogger?.LogSimulationEvent(
                "robot_registered",
                $"Robot {robotId} registered with profile {profileName}"
            );

            Debug.Log($"Registered robot: {robotId} with profile: {profileName}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to register robot {robotId}: {ex.Message}");
        }
    }

    /// <summary>
    /// Unregisters a robot from the RobotManager.
    /// </summary>
    /// <param name="robotId">The robot identifier to unregister</param>
    public void UnregisterRobot(string robotId)
    {
        if (_robotInstances.Remove(robotId))
        {
            if (_fileLogger != null)
            {
                _fileLogger.LogSimulationEvent(
                    "robot_unregistered",
                    $"Robot {robotId} unregistered"
                );
            }
            Debug.Log($"Unregistered robot: {robotId}");
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
            int jointCount = Mathf.Min(controller.robotJoints.Length, robot.profile.joints.Length);
            if (controller.robotJoints.Length != robot.profile.joints.Length)
            {
                Debug.LogWarning(
                    $"Joint count mismatch for {robotId}: Controller has {controller.robotJoints.Length}, "
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

            if (_logConfigurationChanges && _robotActionLogger != null)
            {
                _robotActionLogger.LogAction(
                    "configuration_applied",
                    robotId,
                    robot.profile.profileName,
                    null,
                    null,
                    0f,
                    true
                );
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to apply profile to robot {robotId}: {ex.Message}");
        }
    }

    /// <summary>
    /// Gets the profile for a specific robot by robotId
    /// </summary>
    /// <param name="robotId">The robot identifier</param>
    /// <returns>The robot's profile or null if robot not found</returns>
    public RobotConfig GetRobotProfile(string robotId)
    {
        return _robotInstances.TryGetValue(robotId, out RobotInstance robot) ? robot.profile : null;
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

        _fileLogger?.LogSimulationEvent("robot_configuration_summary", summary);
        Debug.Log(summary);
    }

    /// <summary>
    /// Unity OnDestroy callback - cleans up singleton instance.
    /// </summary>
    private void OnDestroy()
    {
        if (Instance == this)
        {
            if (_fileLogger != null)
            {
                _fileLogger.LogSimulationEvent("robot_manager_destroyed", "RobotManager destroyed");
            }
            Instance = null;
        }
    }
}
