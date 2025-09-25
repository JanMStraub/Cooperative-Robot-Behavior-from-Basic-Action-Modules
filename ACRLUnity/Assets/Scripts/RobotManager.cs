using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

[System.Serializable]
public class JointConfiguration
{
    [Header("Joint Parameters")]
    public float stiffness = 800f;
    public float damping = 250f;
    public float forceLimit = 1000f;
    public float upperLimit = 170f;
    public float lowerLimit = -170f;

    [Header("Performance Settings")]
    public float maxVelocity = 180f; // degrees per second
    public float acceleration = 360f; // degrees per second squared

    public JointConfiguration() { }

    public JointConfiguration(float stiff, float damp, float force, float upper, float lower)
    {
        stiffness = stiff;
        damping = damp;
        forceLimit = force;
        upperLimit = upper;
        lowerLimit = lower;
    }
}

[System.Serializable]
public class RobotProfile
{
    [Header("Robot Identity")]
    public string profileName = "AR4_Default";
    public string description = "Standard AR4 robotic arm configuration";

    [Header("Joint Configurations")]
    public JointConfiguration[] joints = new JointConfiguration[6];

    [Header("IK Settings")]
    [Range(0.1f, 5f)]
    public float adjustmentSpeed = 1.0f;

    [Range(0.001f, 0.5f)]
    public float convergenceThreshold = 0.1f;

    [Range(0.01f, 0.5f)]
    public float maxJointStepRad = 0.1f;

    [Header("Performance Limits")]
    public float maxReachDistance = 0.8f;
    public float minReachDistance = 0.1f;
    public int maxIKIterations = 100;
    public float ikTimeout = 5f;

    public RobotProfile()
    {
        InitializeDefaultAR4Profile();
    }

    public void InitializeDefaultAR4Profile()
    {
        joints = new JointConfiguration[6]
        {
            new JointConfiguration(800, 250, 1000, 170, -170), // Base
            new JointConfiguration(700, 200, 1500, 90, -90), // Shoulder
            new JointConfiguration(600, 150, 1000, 65, -70), // Elbow
            new JointConfiguration(300, 100, 800, 135, -135), // Wrist 1
            new JointConfiguration(200, 80, 500, 100, -100), // Wrist 2
            new JointConfiguration(100, 50, 300, 180, -180), // Wrist 3
        };
    }
}

[System.Serializable]
public class RobotInstance
{
    public string robotId;
    public GameObject robotGameObject;
    public GameObject targetGameObject;
    public RobotController controller;
    public RobotProfile profile;
    public bool isActive;
    public float lastTargetChangeTime;
    public Vector3 lastTargetPosition;
}

public class RobotManager : MonoBehaviour
{
    public static RobotManager Instance { get; private set; }

    [Header("Robot Profiles")]
    [SerializeField]
    private RobotProfile defaultProfile = new RobotProfile();

    [SerializeField]
    private List<RobotProfile> robotProfiles = new List<RobotProfile>();

    [Header("Global Settings")]
    [SerializeField, Range(0.1f, 5f)]
    private float globalSpeedMultiplier = 1.0f;

    [SerializeField]
    private bool enableTargetChangeDetection = true;

    [SerializeField]
    private float targetChangeCheckInterval = 0.1f;

    [Header("Legacy Support")]
    [SerializeField]
    private GameObject leftTarget;

    [SerializeField]
    private GameObject rightTarget;

    [SerializeField]
    private GameObject leftRobot;

    [SerializeField]
    private GameObject rightRobot;

    [Header("Debug Settings")]
    [SerializeField]
    private bool logConfigurationChanges = true;

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
    public RobotProfile DefaultProfile => defaultProfile;
    public IReadOnlyDictionary<string, RobotInstance> RobotInstances => _robotInstances;
    public int ActiveRobotCount => _robotInstances.Values.Count(r => r.isActive);

    // Legacy properties for backward compatibility
    public float robotAdjustmentSpeed => defaultProfile.adjustmentSpeed * globalSpeedMultiplier;
    public float convergenceThreshold => defaultProfile.convergenceThreshold;
    public float maxRawJointStepRad => defaultProfile.maxJointStepRad;

    // Singleton pattern initialization
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

    private void InitializeManager()
    {
        try
        {
            // Ensure default profile is properly initialized
            if (defaultProfile.joints == null || defaultProfile.joints.Length == 0)
            {
                defaultProfile.InitializeDefaultAR4Profile();
            }

            _nextTargetCheckTime = Time.time + targetChangeCheckInterval;

            Debug.Log($"RobotManager initialized with {robotProfiles.Count + 1} profiles");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to initialize RobotManager: {ex.Message}");
        }
    }

    private void Start()
    {
        try
        {
            // Get component references
            _fileLogger = FileLogger.Instance;
            _robotActionLogger = RobotActionLogger.Instance;

            // Auto-discover robots
            DiscoverRobots();

            // Setup legacy robots if specified
            SetupLegacyRobots();

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

    private void DiscoverRobots()
    {
        RobotController[] controllers = FindObjectsByType<RobotController>(
            FindObjectsSortMode.None
        );

        foreach (var controller in controllers)
        {
            string robotId = controller.gameObject.name;

            if (!_robotInstances.ContainsKey(robotId))
            {
                RegisterRobot(robotId, controller.gameObject, null, GetProfileForRobot(robotId));
                Debug.Log($"Auto-discovered robot: {robotId}");
            }
        }
    }

    private void SetupLegacyRobots()
    {
        // Setup legacy left robot
        if (leftRobot != null && leftTarget != null)
        {
            string leftId = "AR4Left";
            if (!_robotInstances.ContainsKey(leftId))
            {
                RegisterRobot(leftId, leftRobot, leftTarget, defaultProfile);
            }
            else
            {
                _robotInstances[leftId].targetGameObject = leftTarget;
            }
        }

        // Setup legacy right robot
        if (rightRobot != null && rightTarget != null)
        {
            string rightId = "AR4Right";
            if (!_robotInstances.ContainsKey(rightId))
            {
                RegisterRobot(rightId, rightRobot, rightTarget, defaultProfile);
            }
            else
            {
                _robotInstances[rightId].targetGameObject = rightTarget;
            }
        }
    }

    private void Update()
    {
        if (enableTargetChangeDetection && Time.time >= _nextTargetCheckTime)
        {
            CheckForTargetChanges();
            _nextTargetCheckTime = Time.time + targetChangeCheckInterval;
        }
    }

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

                    if (logConfigurationChanges)
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

    // Public API
    public void RegisterRobot(
        string robotId,
        GameObject robotObject,
        GameObject targetObject = null,
        RobotProfile profile = null
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
                profile = profile ?? defaultProfile,
                isActive = true,
                lastTargetChangeTime = Time.time,
                lastTargetPosition = targetObject?.transform.position ?? Vector3.zero,
            };

            _robotInstances[robotId] = instance;

            // Apply configuration to robot
            ApplyProfileToRobot(robotId);

            _fileLogger?.LogSimulationEvent(
                "robot_registered",
                $"Robot {robotId} registered with profile {profile?.profileName ?? "default"}"
            );

            Debug.Log(
                $"Registered robot: {robotId} with profile: {profile?.profileName ?? "default"}"
            );
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to register robot {robotId}: {ex.Message}");
        }
    }

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

    public void SetRobotProfile(string robotId, RobotProfile profile)
    {
        if (_robotInstances.TryGetValue(robotId, out RobotInstance robot))
        {
            robot.profile = profile;
            ApplyProfileToRobot(robotId);

            if (logConfigurationChanges)
            {
                _fileLogger?.LogSimulationEvent(
                    "robot_profile_changed",
                    $"Robot {robotId} profile changed to {profile.profileName}"
                );
            }
        }
    }

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

    public void SetRobotActive(string robotId, bool active)
    {
        if (_robotInstances.TryGetValue(robotId, out RobotInstance robot))
        {
            robot.isActive = active;
            _fileLogger?.LogSimulationEvent(
                "robot_activation_changed",
                $"Robot {robotId} active: {active}"
            );
        }
    }

    private void ApplyProfileToRobot(string robotId)
    {
        if (!_robotInstances.TryGetValue(robotId, out RobotInstance robot))
            return;

        try
        {
            var controller = robot.controller;
            if (controller?.robotJoints == null)
                return;

            // Apply joint configurations
            for (
                int i = 0;
                i < controller.robotJoints.Length && i < robot.profile.joints.Length;
                i++
            )
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

            if (logConfigurationChanges && _robotActionLogger != null)
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

    private RobotProfile GetProfileForRobot(string robotId)
    {
        // Try to find specific profile for robot
        var profile = robotProfiles.FirstOrDefault(p => p.profileName.Contains(robotId));
        return profile ?? defaultProfile;
    }

    private void LogConfigurationSummary()
    {
        if (!logConfigurationChanges)
            return;

        string summary = $"RobotManager Configuration Summary:\n";
        summary += $"- Total Robots: {_robotInstances.Count}\n";
        summary += $"- Global Speed Multiplier: {globalSpeedMultiplier}\n";
        summary += $"- Default Profile: {defaultProfile.profileName}\n";
        summary += $"- Available Profiles: {robotProfiles.Count + 1}\n";

        foreach (var robot in _robotInstances.Values)
        {
            summary +=
                $"- {robot.robotId}: Profile '{robot.profile.profileName}', Active: {robot.isActive}\n";
        }

        _fileLogger?.LogSimulationEvent("robot_configuration_summary", summary);
        Debug.Log(summary);
    }

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
