using System;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Trajectory;
using RosMessageTypes.Std;
using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Subscribes to ROS 2 JointTrajectory messages and executes them on the robot.
    /// When a trajectory is received, sets RobotController.IsManuallyDriven = true
    /// to bypass Unity IK, then interpolates between waypoints by setting
    /// ArticulationBody xDrive targets directly.
    ///
    /// Fires OnTrajectoryComplete when execution finishes, which can be used
    /// to notify SimulationManager or send completion feedback to Python.
    /// </summary>
    public class ROSTrajectorySubscriber : MonoBehaviour
    {
        [Header("ROS Configuration")]
        [Tooltip("ROS topic for incoming joint trajectories (use {robot_id} for per-robot topics)")]
        [SerializeField]
        private string _trajectoryTopic = "/{robot_id}/arm_controller/joint_trajectory";

        [Tooltip("ROS topic for trajectory execution feedback (use {robot_id} for per-robot topics)")]
        [SerializeField]
        private string _feedbackTopic = "/{robot_id}/arm_controller/feedback";

        [Header("Safety")]
        [Tooltip("Maximum allowed joint velocity in rad/s")]
        [SerializeField]
        private float _maxJointVelocity = 1.05f; // ~60 deg/s, matching URDF

        [Tooltip("Reject trajectories with points further apart than this (seconds)")]
        [SerializeField]
        private float _maxPointGap = 5f;

        [Header("Speed Control")]
        [Tooltip("Speed scaling factor for trajectory execution (1.0 = normal speed, 0.5 = half speed, 2.0 = double speed)")]
        [SerializeField]
        [Range(0.1f, 2.0f)]
        private float _speedScaling = 0.5f; // Default to 50% speed for slower, more visible motion

        [Header("References")]
        [SerializeField]
        private RobotController _robotController;

        private ROSConnection _ros;
        private Coroutine _executionCoroutine;
        private ArticulationBody[] _joints;
        private int[] _jointIndexMap; // Maps trajectory joint names to robotJoints indices
        private string _resolvedTrajectoryTopic;
        private string _resolvedFeedbackTopic;

        private const string _logPrefix = "[ROS_TRAJECTORY_SUBSCRIBER]";

        /// <summary>
        /// True while a trajectory is being executed.
        /// </summary>
        public bool IsExecutingTrajectory { get; private set; }

        /// <summary>
        /// Fired when a trajectory completes execution (success or abort).
        /// Bool parameter indicates success.
        /// </summary>
        public event Action<bool> OnTrajectoryComplete;

        /// <summary>
        /// URDF joint names in the same order as RobotController.robotJoints.
        /// </summary>
        private static readonly string[] ExpectedJointNames =
        {
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        };

        private void Start()
        {
            _ros = ROSConnection.GetOrCreateInstance();

            if (_robotController == null)
                _robotController = GetComponentInParent<RobotController>();

            if (_robotController == null)
            {
                Debug.LogError($"{_logPrefix} No RobotController found. Disabling subscriber.");
                enabled = false;
                return;
            }

            _joints = _robotController.robotJoints;

            // Resolve topic names with robot ID - ensure per-robot namespacing
            _resolvedTrajectoryTopic = ResolveTopicName(_trajectoryTopic, _robotController.robotId);
            _resolvedFeedbackTopic = ResolveTopicName(_feedbackTopic, _robotController.robotId);

            _ros.Subscribe<JointTrajectoryMsg>(_resolvedTrajectoryTopic, OnTrajectoryReceived);
            _ros.RegisterPublisher<StringMsg>(_resolvedFeedbackTopic);

            Debug.Log(
                $"{_logPrefix} Subscribed to {_resolvedTrajectoryTopic} for {_robotController.robotId}"
            );
        }

        /// <summary>
        /// Callback when a JointTrajectory message arrives from ROS.
        /// </summary>
        private void OnTrajectoryReceived(JointTrajectoryMsg msg)
        {
            if (msg.points == null || msg.points.Length == 0)
            {
                Debug.LogWarning($"{_logPrefix} Received empty trajectory. Ignoring.");
                return;
            }

            if (IsExecutingTrajectory)
            {
                Debug.LogWarning($"{_logPrefix} Already executing trajectory. Aborting current.");
                AbortExecution();
            }

            // Build joint name -> index mapping
            _jointIndexMap = BuildJointIndexMap(msg.joint_names);
            if (_jointIndexMap == null)
            {
                Debug.LogError($"{_logPrefix} Failed to map joint names. Rejecting trajectory.");
                PublishFeedback("rejected", "Joint name mapping failed");
                return;
            }

            // Validate trajectory safety
            if (!ValidateTrajectory(msg))
            {
                Debug.LogError($"{_logPrefix} Trajectory validation failed. Rejecting.");
                PublishFeedback("rejected", "Trajectory validation failed");
                return;
            }

            Debug.Log(
                $"{_logPrefix} Executing trajectory with {msg.points.Length} points "
                + $"for {_robotController.robotId}"
            );

            _executionCoroutine = StartCoroutine(ExecuteTrajectory(msg));
        }

        /// <summary>
        /// Map incoming trajectory joint names to ArticulationBody indices.
        /// Returns null if any joint name is unrecognized.
        /// </summary>
        private int[] BuildJointIndexMap(string[] trajectoryJointNames)
        {
            if (trajectoryJointNames == null || trajectoryJointNames.Length == 0)
                return null;

            int[] map = new int[trajectoryJointNames.Length];

            for (int i = 0; i < trajectoryJointNames.Length; i++)
            {
                int found = -1;
                for (int j = 0; j < ExpectedJointNames.Length; j++)
                {
                    if (trajectoryJointNames[i] == ExpectedJointNames[j])
                    {
                        found = j;
                        break;
                    }
                }

                if (found < 0 || found >= _joints.Length)
                {
                    Debug.LogError(
                        $"{_logPrefix} Unknown joint name: {trajectoryJointNames[i]}"
                    );
                    return null;
                }

                map[i] = found;
            }

            return map;
        }

        /// <summary>
        /// Validate trajectory for safety: check velocities, time ordering, joint limits.
        /// </summary>
        private bool ValidateTrajectory(JointTrajectoryMsg msg)
        {
            double prevTime = 0;

            for (int p = 0; p < msg.points.Length; p++)
            {
                var point = msg.points[p];
                double pointTime = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9;

                // Check time ordering - allow equal timestamps (MoveIt may duplicate
                // the start state at time 0). Execution handles this gracefully.
                if (p > 0 && pointTime < prevTime)
                {
                    Debug.LogError(
                        $"{_logPrefix} Non-monotonic time at point {p}: "
                        + $"{pointTime:F4}s < prev {prevTime:F4}s"
                    );
                    return false;
                }

                // Check max gap
                if (p > 0 && (pointTime - prevTime) > _maxPointGap)
                {
                    Debug.LogError($"{_logPrefix} Time gap too large at point {p}");
                    return false;
                }

                // Check joint positions against limits
                if (point.positions != null)
                {
                    for (int j = 0; j < point.positions.Length && j < _jointIndexMap.Length; j++)
                    {
                        int idx = _jointIndexMap[j];
                        ArticulationBody joint = _joints[idx];
                        float posDeg = (float)point.positions[j] * Mathf.Rad2Deg;

                        if (posDeg < joint.xDrive.lowerLimit || posDeg > joint.xDrive.upperLimit)
                        {
                            Debug.LogError(
                                $"{_logPrefix} Joint {idx} position {posDeg:F1} deg exceeds limits "
                                + $"[{joint.xDrive.lowerLimit:F1}, {joint.xDrive.upperLimit:F1}]"
                            );
                            return false;
                        }
                    }
                }

                // Check velocities
                if (point.velocities != null)
                {
                    for (int j = 0; j < point.velocities.Length; j++)
                    {
                        if (Mathf.Abs((float)point.velocities[j]) > _maxJointVelocity)
                        {
                            Debug.LogError(
                                $"{_logPrefix} Velocity {point.velocities[j]:F2} rad/s "
                                + $"exceeds max {_maxJointVelocity} at point {p}, joint {j}"
                            );
                            return false;
                        }
                    }
                }

                prevTime = pointTime;
            }

            return true;
        }

        /// <summary>
        /// Execute a trajectory by interpolating between waypoints.
        /// Sets IsManuallyDriven on the RobotController to bypass Unity IK.
        /// </summary>
        private IEnumerator ExecuteTrajectory(JointTrajectoryMsg msg)
        {
            IsExecutingTrajectory = true;
            _robotController.IsManuallyDriven = true;

            PublishFeedback("executing", $"Starting {msg.points.Length}-point trajectory");
            float startTime = Time.time;

            // Get starting joint positions (in radians)
            double[] startPositions = new double[_jointIndexMap.Length];
            for (int j = 0; j < _jointIndexMap.Length; j++)
            {
                int idx = _jointIndexMap[j];
                startPositions[j] = _joints[idx].jointPosition.dofCount > 0
                    ? _joints[idx].jointPosition[0]
                    : 0.0;
            }

            JointTrajectoryPointMsg prevPoint = null;
            double prevPointTime = 0;

            for (int p = 0; p < msg.points.Length; p++)
            {
                var targetPoint = msg.points[p];
                double targetTime = targetPoint.time_from_start.sec
                    + targetPoint.time_from_start.nanosec * 1e-9;

                // Determine interpolation source
                double[] fromPositions;
                if (prevPoint != null && prevPoint.positions != null)
                    fromPositions = prevPoint.positions;
                else
                    fromPositions = startPositions;

                double segmentDuration = targetTime - prevPointTime;
                if (segmentDuration <= 0)
                    segmentDuration = Time.fixedDeltaTime;

                // Apply speed scaling to slow down trajectory execution in simulation
                segmentDuration *= (1.0 / _speedScaling);

                // Interpolate from previous point to current
                float segmentStart = Time.time;
                while (true)
                {
                    float elapsed = Time.time - segmentStart;
                    float t = Mathf.Clamp01((float)(elapsed / segmentDuration));

                    // Set joint drive targets
                    if (targetPoint.positions != null)
                    {
                        for (int j = 0; j < _jointIndexMap.Length; j++)
                        {
                            if (j >= targetPoint.positions.Length || j >= fromPositions.Length)
                                break;

                            int idx = _jointIndexMap[j];
                            double interpRad = fromPositions[j]
                                + (targetPoint.positions[j] - fromPositions[j]) * t;

                            // Convert radians to degrees for ArticulationBody drive targets
                            float targetDeg = (float)interpRad * Mathf.Rad2Deg;

                            ArticulationDrive drive = _joints[idx].xDrive;
                            drive.target = Mathf.Clamp(targetDeg, drive.lowerLimit, drive.upperLimit);
                            _joints[idx].xDrive = drive;

                            if (idx < _robotController.jointDriveTargets.Length)
                                _robotController.jointDriveTargets[idx] = drive.target;
                        }
                    }

                    if (t >= 1f)
                        break;

                    yield return new WaitForFixedUpdate();
                }

                prevPoint = targetPoint;
                prevPointTime = targetTime;
            }

            float totalDuration = Time.time - startTime;
            Debug.Log(
                $"{_logPrefix} Trajectory completed in {totalDuration:F2}s "
                + $"for {_robotController.robotId}"
            );

            IsExecutingTrajectory = false;
            _robotController.IsManuallyDriven = false;
            _executionCoroutine = null;

            PublishFeedback("completed", $"Finished in {totalDuration:F2}s");

            // Notify listeners
            _robotController.SetTargetReached(true);
            OnTrajectoryComplete?.Invoke(true);
        }

        /// <summary>
        /// Abort the currently executing trajectory.
        /// </summary>
        public void AbortExecution()
        {
            if (_executionCoroutine != null)
            {
                StopCoroutine(_executionCoroutine);
                _executionCoroutine = null;
            }

            IsExecutingTrajectory = false;
            _robotController.IsManuallyDriven = false;

            PublishFeedback("aborted", "Trajectory execution aborted");
            OnTrajectoryComplete?.Invoke(false);

            Debug.Log($"{_logPrefix} Trajectory execution aborted for {_robotController.robotId}");
        }

        /// <summary>
        /// Publish execution feedback to ROS.
        /// </summary>
        private void PublishFeedback(string status, string message)
        {
            if (_ros == null)
                return;

            var feedback = new StringMsg
            {
                data = $"{{\"robot_id\":\"{_robotController.robotId}\","
                    + $"\"status\":\"{status}\",\"message\":\"{message}\","
                    + $"\"timestamp\":{Time.time}}}"
            };

            _ros.Publish(_resolvedFeedbackTopic, feedback);
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

        private void OnDisable()
        {
            if (IsExecutingTrajectory)
                AbortExecution();
        }
    }
}
