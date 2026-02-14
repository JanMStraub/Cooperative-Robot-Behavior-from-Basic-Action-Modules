using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Robotics
{
    /// <summary>
    /// Control mode for robot motion: Unity IK, ROS MoveIt, or Hybrid.
    /// </summary>
    public enum ControlMode
    {
        /// <summary>Unity-native IK via RobotController (default, existing behavior).</summary>
        Unity,

        /// <summary>ROS MoveIt controls motion via trajectory topics. Unity IK is disabled.</summary>
        ROS,

        /// <summary>ROS has priority when executing a trajectory; falls back to Unity IK otherwise.</summary>
        Hybrid,
    }

#if UNITY_EDITOR
    [CustomEditor(typeof(ROSControlModeManager))]
    public class ROSControlModeManagerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var manager = (ROSControlModeManager)target;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Runtime Controls", EditorStyles.boldLabel);
            EditorGUILayout.LabelField($"Current Mode: {manager.CurrentMode}");
            EditorGUILayout.LabelField($"ROS Trajectory Active: {manager.IsROSTrajectoryActive}");

            EditorGUILayout.Space();
            EditorGUILayout.BeginHorizontal();

            if (GUILayout.Button("Unity Mode"))
                manager.SetControlMode(ControlMode.Unity);

            if (GUILayout.Button("ROS Mode"))
                manager.SetControlMode(ControlMode.ROS);

            if (GUILayout.Button("Hybrid Mode"))
                manager.SetControlMode(ControlMode.Hybrid);

            EditorGUILayout.EndHorizontal();
        }
    }
#endif

    /// <summary>
    /// Manages the control mode for a robot, switching between Unity IK and ROS MoveIt.
    /// Attach to the robot root GameObject alongside RobotController.
    ///
    /// In Unity mode: RobotController IK runs normally, ROS subscribers are disabled.
    /// In ROS mode: RobotController IK is bypassed, ROS subscribers control joints.
    /// In Hybrid mode: ROS has priority during trajectory execution, otherwise Unity IK.
    ///
    /// Joint state publishing is always active regardless of mode to keep ROS informed.
    /// </summary>
    public class ROSControlModeManager : MonoBehaviour
    {
        [Header("Control Mode")]
        [Tooltip("Initial control mode")]
        [SerializeField]
        private ControlMode _initialMode = ControlMode.Unity;

        [Header("References")]
        [SerializeField]
        private RobotController _robotController;

        [SerializeField]
        private ROSJointStatePublisher _jointStatePublisher;

        [SerializeField]
        private ROSTrajectorySubscriber _trajectorySubscriber;

        [SerializeField]
        private ROSGripperSubscriber _gripperSubscriber;

        private ControlMode _currentMode;
        private const string _logPrefix = "[ROS_CONTROL_MODE_MANAGER]";

        /// <summary>
        /// The current active control mode.
        /// </summary>
        public ControlMode CurrentMode => _currentMode;

        /// <summary>
        /// Whether a ROS trajectory is currently being executed.
        /// </summary>
        public bool IsROSTrajectoryActive =>
            _trajectorySubscriber != null && _trajectorySubscriber.IsExecutingTrajectory;

        /// <summary>
        /// Whether Unity IK should be active based on current mode and state.
        /// In Hybrid mode, Unity IK is active only when ROS is not executing a trajectory.
        /// </summary>
        public bool ShouldUnityIKBeActive
        {
            get
            {
                return _currentMode switch
                {
                    ControlMode.Unity => true,
                    ControlMode.ROS => false,
                    ControlMode.Hybrid => !IsROSTrajectoryActive,
                    _ => true,
                };
            }
        }

        private void Start()
        {
            // Auto-find components if not assigned
            if (_robotController == null)
                _robotController = GetComponent<RobotController>();

            if (_jointStatePublisher == null)
                _jointStatePublisher = GetComponentInChildren<ROSJointStatePublisher>();

            if (_trajectorySubscriber == null)
                _trajectorySubscriber = GetComponentInChildren<ROSTrajectorySubscriber>();

            if (_gripperSubscriber == null)
                _gripperSubscriber = GetComponentInChildren<ROSGripperSubscriber>();

            if (_robotController == null)
            {
                Debug.LogError($"{_logPrefix} No RobotController found. Disabling.");
                enabled = false;
                return;
            }

            // Apply initial mode
            SetControlMode(_initialMode);

            string robotId = _robotController.robotId;
            Debug.Log($"{_logPrefix} Initialized for {robotId}. Mode: {_currentMode}");
        }

        private void Update()
        {
            // In Hybrid mode, dynamically toggle IsManuallyDriven based on ROS activity
            if (_currentMode == ControlMode.Hybrid && _robotController != null)
            {
                bool rosActive = IsROSTrajectoryActive;
                if (_robotController.IsManuallyDriven != rosActive)
                {
                    _robotController.IsManuallyDriven = rosActive;
                }
            }
        }

        /// <summary>
        /// Switch control mode at runtime.
        /// </summary>
        public void SetControlMode(ControlMode mode)
        {
            ControlMode previousMode = _currentMode;
            _currentMode = mode;

            ApplyMode(mode);

            if (previousMode != mode)
            {
                string robotId = _robotController != null ? _robotController.robotId : "unknown";
                Debug.Log($"{_logPrefix} [{robotId}] Mode changed: {previousMode} -> {mode}");
            }
        }

        /// <summary>
        /// Apply the control mode settings to all components.
        /// </summary>
        private void ApplyMode(ControlMode mode)
        {
            switch (mode)
            {
                case ControlMode.Unity:
                    // Unity IK active, ROS subscribers disabled
                    if (_robotController != null)
                        _robotController.IsManuallyDriven = false;
                    SetROSSubscribersEnabled(false);
                    break;

                case ControlMode.ROS:
                    // Unity IK disabled, ROS subscribers active
                    if (_robotController != null)
                        _robotController.IsManuallyDriven = true;
                    SetROSSubscribersEnabled(true);
                    break;

                case ControlMode.Hybrid:
                    // Both available; ROS takes priority when executing trajectory
                    if (_robotController != null)
                        _robotController.IsManuallyDriven = IsROSTrajectoryActive;
                    SetROSSubscribersEnabled(true);
                    break;
            }

            // Joint state publisher is always active to keep ROS informed
            if (_jointStatePublisher != null)
                _jointStatePublisher.SetPublishing(true);
        }

        /// <summary>
        /// Enable or disable ROS subscriber components.
        /// </summary>
        private void SetROSSubscribersEnabled(bool enable)
        {
            if (_trajectorySubscriber != null)
                _trajectorySubscriber.enabled = enable;

            if (_gripperSubscriber != null)
                _gripperSubscriber.SetActive(enable);
        }
    }
}
