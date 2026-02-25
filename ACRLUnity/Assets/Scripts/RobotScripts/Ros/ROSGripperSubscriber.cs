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
        /// Position[0] = gripper jaw position in meters (0=closed, 0.014=fully open).
        /// This matches ROSMotionClient.control_gripper() which sends raw meter values.
        /// Effort[0] = optional max force limit.
        ///
        /// When closing (position &lt; 0.002m, i.e. ~2mm = effectively closed), automatically
        /// finds and sets the nearest graspable object so GripperController.AttachObject
        /// fires on close completion. This mirrors what RobotController.ExecuteThreeWaypointGrasp
        /// does for the Unity IK path; without it the gripper closes but never captures the object.
        /// </summary>
        private void OnGripperCommandReceived(JointStateMsg msg)
        {
            if (!IsActive || _gripperController == null)
                return;

            if (msg.position != null && msg.position.Length > 0)
            {
                // Value is in meters (0=closed, 0.014=fully open) — clamp to valid jaw range.
                float positionMeters = Mathf.Clamp((float)msg.position[0], 0f, 0.014f);

                Debug.Log(
                    $"{_logPrefix} Gripper command received: position={positionMeters:F4}m"
                );

                // When closing, arm the attachment by finding the nearest Target-tagged object
                // within the gripper's reach so GripperController.AttachObject fires.
                // Threshold of 0.002m (~2mm) treats near-zero positions as a close command.
                if (positionMeters < 0.002f)
                {
                    GameObject nearestTarget = FindNearestGraspableObject();
                    if (nearestTarget != null)
                    {
                        _gripperController.SetTargetObject(nearestTarget);
                        Debug.Log(
                            $"{_logPrefix} Arming grasp attachment for '{nearestTarget.name}'"
                        );
                    }
                    else
                    {
                        Debug.LogWarning($"{_logPrefix} Close command received but no graspable object found nearby");
                    }
                    _gripperController.CloseGrippers();
                }
                else
                {
                    // Opening: use OpenGrippers() so _detachmentPending is set and the held
                    // object is properly released. SetGripperPosition() alone only moves the
                    // fingers without triggering the deferred-detach state machine.
                    _gripperController.OpenGrippers();
                }
            }
            else if (msg.name != null && msg.name.Length > 0)
            {
                // Support string-based commands: "open" or "close"
                string command = msg.name[0].ToLower();
                if (command == "open")
                {
                    _gripperController.ClearTargetObject();
                    _gripperController.OpenGrippers();
                }
                else if (command == "close")
                {
                    GameObject nearestTarget = FindNearestGraspableObject();
                    if (nearestTarget != null)
                        _gripperController.SetTargetObject(nearestTarget);
                    _gripperController.CloseGrippers();
                }
            }
        }

        /// <summary>
        /// Find the nearest GameObject tagged "Target" within gripper reach.
        /// Searches from the finger positions (below ee_link) with a generous radius
        /// to account for the offset between the attachment point and the cube center.
        /// Falls back to the attachment point or transform if fingers are unavailable.
        /// </summary>
        private GameObject FindNearestGraspableObject()
        {
            // Prefer finger positions — they sit closer to the grasped object than
            // the attachment point (ee_link / gripper_focus) which is above them.
            Vector3 searchOrigin;
            if (_gripperController.leftGripper != null)
            {
                // Average of both finger positions gives the centre of the gripper mouth
                Vector3 left = _gripperController.leftGripper.transform.position;
                Vector3 right = _gripperController.rightGripper != null
                    ? _gripperController.rightGripper.transform.position
                    : left;
                searchOrigin = (left + right) * 0.5f;
            }
            else if (_gripperController.attachmentPoint != null)
            {
                searchOrigin = _gripperController.attachmentPoint.position;
            }
            else
            {
                searchOrigin = _gripperController.transform.position;
            }

            // 20cm radius: finger tips are ~5cm below ee_link, cube half-height ~3cm,
            // so worst-case cube centre is ~8cm from ee_link. 20cm gives ample margin.
            const float searchRadius = 0.20f;
            Collider[] hits = Physics.OverlapSphere(searchOrigin, searchRadius);

            GameObject nearest = null;
            float nearestDist = float.MaxValue;
            int candidateCount = 0;

            foreach (var hit in hits)
            {
                if (!hit.CompareTag("Target"))
                    continue;

                candidateCount++;
                float dist = Vector3.Distance(searchOrigin, hit.transform.position);
                if (dist < nearestDist)
                {
                    nearestDist = dist;
                    nearest = hit.gameObject;
                }
            }

            if (nearest != null)
            {
                if (candidateCount > 1)
                    Debug.LogWarning(
                        $"{_logPrefix} {candidateCount} Target-tagged objects within {searchRadius*100f:F0}cm — "
                        + $"attaching to nearest ('{nearest.name}' at {nearestDist*100f:F1}cm). "
                        + "Wrong object may be grasped in dense scenes."
                    );
                else
                    Debug.Log($"{_logPrefix} Found '{nearest.name}' at {nearestDist*100f:F1}cm from gripper centre");
            }
            else
                Debug.LogWarning($"{_logPrefix} No Target-tagged object within {searchRadius*100f:F0}cm of {searchOrigin}");

            return nearest;
        }

        private void Update()
        {
            if (!IsActive || _ros == null || _gripperController == null)
                return;

            // Don't queue messages before the ROS connection is established
            if (
                ROSConnectionInitializer.Instance != null
                && !ROSConnectionInitializer.Instance.IsConnected
            )
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
            System.DateTime epoch = new System.DateTime(
                1970,
                1,
                1,
                0,
                0,
                0,
                System.DateTimeKind.Utc
            );
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
