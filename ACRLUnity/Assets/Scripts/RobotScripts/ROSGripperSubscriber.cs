using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using Simulation;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Subscribes to ROS 2 gripper commands and bridges them to the existing GripperController.
    /// Receives JointState messages on /gripper/command with position field controlling
    /// the normalized gripper opening (0=closed, 1=open).
    /// Publishes gripper feedback state on /gripper/state.
    /// </summary>
    public class ROSGripperSubscriber : MonoBehaviour
    {
        [Header("ROS Configuration")]
        [Tooltip("ROS topic for incoming gripper commands (use {robot_id} for per-robot topics)")]
        [SerializeField]
        private string _commandTopic = "/{robot_id}/gripper/command";

        [Tooltip("ROS topic for gripper state feedback (use {robot_id} for per-robot topics)")]
        [SerializeField]
        private string _stateTopic = "/{robot_id}/gripper/state";

        [Tooltip("State feedback publish rate in Hz")]
        [SerializeField]
        [Range(1f, 50f)]
        private float _statePublishRate = 10f;

        [Header("References")]
        [SerializeField]
        private GripperController _gripperController;

        [SerializeField]
        private RobotController _robotController;

        private ROSConnection _ros;
        private float _statePublishInterval;
        private float _timeSinceLastStatePublish;
        private JointStateMsg _stateMsg;
        private string _resolvedCommandTopic;
        private string _resolvedStateTopic;

        private const string _logPrefix = "[ROS_GRIPPER_SUBSCRIBER]";

        /// <summary>
        /// Whether this subscriber is active.
        /// </summary>
        public bool IsActive { get; private set; }

        private void Start()
        {
            _ros = ROSConnection.GetOrCreateInstance();

            if (_gripperController == null)
                _gripperController = GetComponentInChildren<GripperController>();

            if (_robotController == null)
                _robotController = GetComponentInParent<RobotController>();

            if (_gripperController == null)
            {
                Debug.LogError($"{_logPrefix} No GripperController found. Disabling.");
                enabled = false;
                return;
            }

            _statePublishInterval = 1f / _statePublishRate;
            _timeSinceLastStatePublish = 0f;

            // Initialize state feedback message
            _stateMsg = new JointStateMsg
            {
                header = new HeaderMsg(),
                name = new[] { "gripper_jaw1_joint", "gripper_jaw2_joint" },
                position = new double[2],
                velocity = new double[2],
                effort = new double[2],
            };

            // Resolve topic names with robot ID - ensure per-robot namespacing
            string robotId = _robotController != null ? _robotController.robotId : "unknown";
            _resolvedCommandTopic = ResolveTopicName(_commandTopic, robotId);
            _resolvedStateTopic = ResolveTopicName(_stateTopic, robotId);

            _ros.Subscribe<JointStateMsg>(_resolvedCommandTopic, OnGripperCommandReceived);
            _ros.RegisterPublisher<JointStateMsg>(_resolvedStateTopic);

            IsActive = true;

            Debug.Log(
                $"{_logPrefix} Initialized for {robotId}. "
                    + $"Listening on {_resolvedCommandTopic}, publishing state on {_resolvedStateTopic}"
            );
        }

        /// <summary>
        /// Handle incoming gripper command from ROS.
        /// Position[0] = normalized gripper position (0=closed, 1=open).
        /// Effort[0] = optional max force limit.
        /// </summary>
        private void OnGripperCommandReceived(JointStateMsg msg)
        {
            if (!IsActive || _gripperController == null)
                return;

            if (msg.position != null && msg.position.Length > 0)
            {
                float normalizedPosition = Mathf.Clamp01((float)msg.position[0]);

                Debug.Log(
                    $"{_logPrefix} Gripper command received: position={normalizedPosition:F2}"
                );

                _gripperController.SetGripperPosition(normalizedPosition);
            }
            else if (msg.name != null && msg.name.Length > 0)
            {
                // Support string-based commands: "open" or "close"
                string command = msg.name[0].ToLower();
                if (command == "open")
                {
                    _gripperController.OpenGrippers();
                }
                else if (command == "close")
                {
                    _gripperController.CloseGrippers();
                }
            }
        }

        private void Update()
        {
            if (!IsActive || _ros == null || _gripperController == null)
                return;

            // Don't queue messages before the ROS connection is established
            if (ROSConnectionInitializer.Instance != null && !ROSConnectionInitializer.Instance.IsConnected)
                return;

            _timeSinceLastStatePublish += Time.deltaTime;

            if (_timeSinceLastStatePublish >= _statePublishInterval)
            {
                PublishGripperState();
                _timeSinceLastStatePublish = 0f;
            }
        }

        /// <summary>
        /// Publish current gripper state as JointState feedback.
        /// </summary>
        private void PublishGripperState()
        {
            // Use system clock (Unix epoch) for ROS 2 time synchronization
            System.DateTime epoch = new System.DateTime(1970, 1, 1, 0, 0, 0, System.DateTimeKind.Utc);
            System.TimeSpan timeSinceEpoch = System.DateTime.UtcNow - epoch;
            double t = timeSinceEpoch.TotalSeconds;
            int sec = (int)t;
            uint nsec = (uint)((t - sec) * 1e9);

            _stateMsg.header.stamp = new TimeMsg { sec = sec, nanosec = nsec };

            // Read actual joint positions from ArticulationBody
            if (_gripperController.leftGripper != null)
            {
                var lg = _gripperController.leftGripper;
                _stateMsg.position[0] = lg.jointPosition.dofCount > 0 ? lg.jointPosition[0] : 0.0;
                _stateMsg.velocity[0] = lg.jointVelocity.dofCount > 0 ? lg.jointVelocity[0] : 0.0;
                _stateMsg.effort[0] = lg.jointForce.dofCount > 0 ? lg.jointForce[0] : 0.0;
            }

            if (_gripperController.rightGripper != null)
            {
                var rg = _gripperController.rightGripper;
                _stateMsg.position[1] = rg.jointPosition.dofCount > 0 ? rg.jointPosition[0] : 0.0;
                _stateMsg.velocity[1] = rg.jointVelocity.dofCount > 0 ? rg.jointVelocity[0] : 0.0;
                _stateMsg.effort[1] = rg.jointForce.dofCount > 0 ? rg.jointForce[0] : 0.0;
            }

            _ros.Publish(_resolvedStateTopic, _stateMsg);
        }

        /// <summary>
        /// Resolve topic name, ensuring it includes the robot ID namespace.
        /// Handles both new format (/{robot_id}/topic) and legacy format (/topic).
        /// </summary>
        private static string ResolveTopicName(string topicTemplate, string robotId)
        {
            if (topicTemplate.Contains("{robot_id}"))
                return topicTemplate.Replace("{robot_id}", robotId);

            // Legacy topic without placeholder - prepend robot ID namespace
            if (topicTemplate.StartsWith("/"))
                return $"/{robotId}{topicTemplate}";

            return $"/{robotId}/{topicTemplate}";
        }

        /// <summary>
        /// Enable or disable the gripper subscriber.
        /// </summary>
        public void SetActive(bool active)
        {
            IsActive = active;
            Debug.Log($"{_logPrefix} {(active ? "Enabled" : "Disabled")}");
        }
    }
}
