using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using Simulation;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

namespace Robotics
{
    /// <summary>Bridges ROS 2 gripper commands to GripperController and publishes gripper state feedback.</summary>
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

        private readonly Collider[] _overlapBuffer = new Collider[32];
        private readonly TimeMsg _rosTimestamp = new TimeMsg();
        private static readonly System.DateTime _unixEpoch = new System.DateTime(
            1970,
            1,
            1,
            0,
            0,
            0,
            System.DateTimeKind.Utc
        );

        private const string _logPrefix = "[ROS_GRIPPER_SUBSCRIBER]";

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

            _stateMsg = new JointStateMsg
            {
                header = new HeaderMsg(),
                name = new[] { "gripper_jaw1_joint", "gripper_jaw2_joint" },
                position = new double[2],
                velocity = new double[2],
                effort = new double[2],
            };

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

        // position[0] in meters (0=closed, 0.014=fully open). On close (<0.002m), arms attachment to nearest Target-tagged object.
        private void OnGripperCommandReceived(JointStateMsg msg)
        {
            if (!IsActive || _gripperController == null)
                return;

            if (msg.position != null && msg.position.Length > 0)
            {
                float positionMeters = Mathf.Clamp((float)msg.position[0], 0f, 0.014f);

                Debug.Log($"{_logPrefix} Gripper command received: position={positionMeters:F4}m");

                if (positionMeters < 0.002f) // ~2mm threshold = close command
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
                        Debug.LogWarning(
                            $"{_logPrefix} Close command received but no graspable object found nearby"
                        );
                    }
                    _gripperController.CloseGrippers();
                }
                else
                {
                    // OpenGrippers() sets _detachmentPending; SetGripperPosition() alone skips the deferred-detach state machine.
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

        // Two passes: (1) free Target-tagged objects via OverlapSphere, (2) objects held by another gripper
        // (handoff case - AttachObject clears the "Target" tag so pass 1 misses them).
        private GameObject FindNearestGraspableObject()
        {
            // Finger tips sit closer to the object than ee_link/gripper_focus.
            Vector3 searchOrigin;
            if (_gripperController.leftGripper != null)
            {
                Vector3 left = _gripperController.leftGripper.transform.position;
                Vector3 right =
                    _gripperController.rightGripper != null
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

            // 10cm: finger tips ~5cm below ee_link, cube half-height ~3cm → worst-case 8cm. Tight enough to exclude adjacent cubes.
            const float searchRadius = 0.10f;

            int hitCount = Physics.OverlapSphereNonAlloc(
                searchOrigin,
                searchRadius,
                _overlapBuffer
            );

            GameObject nearest = null;
            float nearestDist = float.MaxValue;
            int candidateCount = 0;

            for (int i = 0; i < hitCount; i++)
            {
                var hit = _overlapBuffer[i];
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

            if (nearest == null) // Pass 2: handoff - scan other grippers for held objects
            {
                GripperController[] allGrippers = FindObjectsByType<GripperController>(
                    FindObjectsSortMode.None
                );
                foreach (var other in allGrippers)
                {
                    if (other == _gripperController)
                        continue;

                    if (!other.IsHoldingObject || other.GraspedObject == null)
                        continue;

                    float dist = Vector3.Distance(
                        searchOrigin,
                        other.GraspedObject.transform.position
                    );
                    if (dist <= searchRadius && dist < nearestDist)
                    {
                        nearestDist = dist;
                        nearest = other.GraspedObject;
                        candidateCount++;
                        Debug.Log(
                            $"{_logPrefix} Pass 2: found handoff object '{nearest.name}' "
                                + $"held by another gripper at {dist * 100f:F1}cm from gripper centre"
                        );
                    }
                }
            }

            if (nearest != null)
            {
                if (candidateCount > 1)
                    Debug.LogWarning(
                        $"{_logPrefix} {candidateCount} graspable objects within {searchRadius * 100f:F0}cm - "
                            + $"attaching to nearest ('{nearest.name}' at {nearestDist * 100f:F1}cm). "
                            + "Wrong object may be grasped in dense scenes."
                    );
                else
                    Debug.Log(
                        $"{_logPrefix} Found '{nearest.name}' at {nearestDist * 100f:F1}cm from gripper centre"
                    );
            }
            else
                Debug.LogWarning(
                    $"{_logPrefix} No graspable object within {searchRadius * 100f:F0}cm of {searchOrigin}"
                );

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

        private void PublishGripperState()
        {
            double t = (System.DateTime.UtcNow - _unixEpoch).TotalSeconds;
            int sec = (int)t;
            _rosTimestamp.sec = sec;
            _rosTimestamp.nanosec = (uint)((t - sec) * 1e9);

            _stateMsg.header.stamp = _rosTimestamp;

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

        private static string ResolveTopicName(string topicTemplate, string robotId)
        {
            if (topicTemplate.Contains("{robot_id}"))
                return topicTemplate.Replace("{robot_id}", robotId);

            if (topicTemplate.StartsWith("/"))
                return $"/{robotId}{topicTemplate}";

            return $"/{robotId}/{topicTemplate}";
        }

        public void SetActive(bool active)
        {
            IsActive = active;
            Debug.Log($"{_logPrefix} {(active ? "Enabled" : "Disabled")}");
        }
    }
}
