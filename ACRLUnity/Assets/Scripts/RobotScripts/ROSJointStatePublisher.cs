using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using RosMessageTypes.BuiltinInterfaces;
using Simulation;
using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Publishes robot joint states to ROS 2 as sensor_msgs/JointState messages.
    /// Reads positions, velocities, and efforts from ArticulationBody joints and publishes
    /// at a configurable rate (default 50Hz) on the /joint_states topic.
    /// </summary>
    public class ROSJointStatePublisher : MonoBehaviour
    {
        [Header("ROS Configuration")]
        [Tooltip("ROS topic name for joint states (use {robot_id} for per-robot topics)")]
        [SerializeField]
        private string _topicName = "/{robot_id}/joint_states";

        [Tooltip("Publishing rate in Hz")]
        [SerializeField]
        [Range(1f, 100f)]
        private float _publishRate = 50f;

        [Tooltip("Include gripper joints in published state")]
        [SerializeField]
        private bool _includeGripperJoints = true;

        [Header("References")]
        [Tooltip("RobotController to read joint data from")]
        [SerializeField]
        private RobotController _robotController;

        [Tooltip("GripperController for gripper joint data")]
        [SerializeField]
        private GripperController _gripperController;

        private ROSConnection _ros;
        private float _publishInterval;
        private float _timeSinceLastPublish;
        private JointStateMsg _jointStateMsg;
        private string _resolvedTopicName;

        /// <summary>
        /// URDF joint names for the 6-DOF AR4 arm.
        /// </summary>
        private static readonly string[] ArmJointNames =
        {
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        };

        /// <summary>
        /// URDF joint names for the gripper.
        /// </summary>
        private static readonly string[] GripperJointNames =
        {
            "gripper_jaw1_joint", "gripper_jaw2_joint"
        };

        private const string _logPrefix = "[ROS_JOINT_STATE_PUBLISHER]";

        /// <summary>
        /// Whether this publisher is actively publishing.
        /// </summary>
        public bool IsPublishing { get; private set; }

        private void Start()
        {
            _ros = ROSConnection.GetOrCreateInstance();

            if (_robotController == null)
                _robotController = GetComponentInParent<RobotController>();

            if (_gripperController == null)
                _gripperController = GetComponentInChildren<GripperController>();

            if (_robotController == null)
            {
                Debug.LogError($"{_logPrefix} No RobotController found. Disabling publisher.");
                enabled = false;
                return;
            }

            _publishInterval = 1f / _publishRate;
            _timeSinceLastPublish = 0f;

            int jointCount = ArmJointNames.Length;
            if (_includeGripperJoints && _gripperController != null)
                jointCount += GripperJointNames.Length;

            InitializeMessage(jointCount);

            // Resolve topic name with robot ID - ensure per-robot namespacing
            _resolvedTopicName = ResolveTopicName(_topicName, _robotController.robotId);

            _ros.RegisterPublisher<JointStateMsg>(_resolvedTopicName);
            IsPublishing = true;

            Debug.Log(
                $"{_logPrefix} Initialized for {_robotController.robotId}. "
                + $"Publishing {jointCount} joints at {_publishRate}Hz on {_resolvedTopicName}"
            );
        }

        /// <summary>
        /// Pre-allocate the JointState message with joint names to avoid GC.
        /// </summary>
        private void InitializeMessage(int jointCount)
        {
            _jointStateMsg = new JointStateMsg
            {
                header = new HeaderMsg(),
                name = new string[jointCount],
                position = new double[jointCount],
                velocity = new double[jointCount],
                effort = new double[jointCount],
            };

            for (int i = 0; i < ArmJointNames.Length; i++)
                _jointStateMsg.name[i] = ArmJointNames[i];

            if (_includeGripperJoints && _gripperController != null)
            {
                for (int i = 0; i < GripperJointNames.Length; i++)
                    _jointStateMsg.name[ArmJointNames.Length + i] = GripperJointNames[i];
            }
        }

        private void FixedUpdate()
        {
            if (!IsPublishing || _ros == null)
                return;

            // Don't queue messages before the ROS connection is established
            if (ROSConnectionInitializer.Instance != null && !ROSConnectionInitializer.Instance.IsConnected)
                return;

            _timeSinceLastPublish += Time.fixedDeltaTime;

            if (_timeSinceLastPublish >= _publishInterval)
            {
                PublishJointState();
                _timeSinceLastPublish = 0f;
            }
        }

        /// <summary>
        /// Gather current joint data and publish to ROS.
        /// </summary>
        private void PublishJointState()
        {
            ArticulationBody[] joints = _robotController.robotJoints;
            if (joints == null || joints.Length == 0)
                return;

            // Update header timestamp
            _jointStateMsg.header.stamp = RosTimestamp();

            // Read arm joint data
            int armCount = Mathf.Min(joints.Length, ArmJointNames.Length);
            for (int i = 0; i < armCount; i++)
            {
                ArticulationBody joint = joints[i];
                if (joint == null)
                    continue;

                _jointStateMsg.position[i] = joint.jointPosition.dofCount > 0
                    ? joint.jointPosition[0]
                    : 0.0;

                _jointStateMsg.velocity[i] = joint.jointVelocity.dofCount > 0
                    ? joint.jointVelocity[0]
                    : 0.0;

                _jointStateMsg.effort[i] = joint.jointForce.dofCount > 0
                    ? joint.jointForce[0]
                    : 0.0;
            }

            // Read gripper joint data
            if (_includeGripperJoints && _gripperController != null)
            {
                int offset = ArmJointNames.Length;

                if (_gripperController.leftGripper != null)
                {
                    var lg = _gripperController.leftGripper;
                    _jointStateMsg.position[offset] = lg.jointPosition.dofCount > 0
                        ? lg.jointPosition[0] : 0.0;
                    _jointStateMsg.velocity[offset] = lg.jointVelocity.dofCount > 0
                        ? lg.jointVelocity[0] : 0.0;
                    _jointStateMsg.effort[offset] = lg.jointForce.dofCount > 0
                        ? lg.jointForce[0] : 0.0;
                }

                if (_gripperController.rightGripper != null)
                {
                    var rg = _gripperController.rightGripper;
                    _jointStateMsg.position[offset + 1] = rg.jointPosition.dofCount > 0
                        ? rg.jointPosition[0] : 0.0;
                    _jointStateMsg.velocity[offset + 1] = rg.jointVelocity.dofCount > 0
                        ? rg.jointVelocity[0] : 0.0;
                    _jointStateMsg.effort[offset + 1] = rg.jointForce.dofCount > 0
                        ? rg.jointForce[0] : 0.0;
                }
            }

            _ros.Publish(_resolvedTopicName, _jointStateMsg);
        }

        /// <summary>
        /// Create a ROS timestamp from system clock (Unix epoch time).
        /// CRITICAL: Must use system time, NOT Unity simulation time,
        /// for compatibility with ROS 2 time synchronization.
        /// </summary>
        private static TimeMsg RosTimestamp()
        {
            // Use system clock (Unix epoch) instead of Unity simulation time
            // ROS expects seconds since 1970-01-01 00:00:00 UTC
            System.DateTime epoch = new System.DateTime(1970, 1, 1, 0, 0, 0, System.DateTimeKind.Utc);
            System.TimeSpan timeSinceEpoch = System.DateTime.UtcNow - epoch;
            double t = timeSinceEpoch.TotalSeconds;
            int sec = (int)t;
            uint nsec = (uint)((t - sec) * 1e9);
            return new TimeMsg { sec = sec, nanosec = nsec };
        }

        /// <summary>
        /// Resolve topic name, ensuring it includes the robot ID namespace.
        /// Handles both new format (/{robot_id}/topic) and legacy format (/topic).
        /// </summary>
        private static string ResolveTopicName(string topicTemplate, string robotId)
        {
            // If template contains placeholder, replace it
            if (topicTemplate.Contains("{robot_id}"))
                return topicTemplate.Replace("{robot_id}", robotId);

            // Legacy topic without placeholder - prepend robot ID namespace
            // e.g., "/joint_states" -> "/Robot1/joint_states"
            if (topicTemplate.StartsWith("/"))
                return $"/{robotId}{topicTemplate}";

            return $"/{robotId}/{topicTemplate}";
        }

        /// <summary>
        /// Enable or disable publishing at runtime.
        /// </summary>
        public void SetPublishing(bool enable)
        {
            IsPublishing = enable;
            Debug.Log($"{_logPrefix} Publishing {(enable ? "enabled" : "disabled")}");
        }

        /// <summary>
        /// Change the publish rate at runtime.
        /// </summary>
        public void SetPublishRate(float hz)
        {
            _publishRate = Mathf.Clamp(hz, 1f, 100f);
            _publishInterval = 1f / _publishRate;
        }
    }
}
