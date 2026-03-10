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

        [Header("Settle Detection")]
        [Tooltip("Max joint velocity (deg/s) considered 'settled'. Feedback fires only after all joints drop below this.")]
        [SerializeField]
        private float _settleVelocityThresholdDegPerSec = 5.0f;

        [Tooltip("Base time (seconds) to wait for joints to settle before firing 'completed' feedback anyway. "
               + "Actual timeout = this value / _speedScaling so slower trajectories get proportionally more settle time.")]
        [SerializeField]
        private float _settleTimeoutSeconds = 2.0f;

        [Header("References")]
        [SerializeField]
        private RobotController _robotController;

        [Tooltip("Used to determine whether ROS or Unity IK owns the arm after trajectory completion. "
               + "Auto-found on the same GameObject if not assigned.")]
        [SerializeField]
        private ROSControlModeManager _controlModeManager;

        private ROSConnection _ros;
        private Coroutine _executionCoroutine;
        private ArticulationBody[] _joints;
        private int[] _jointIndexMap; // Maps trajectory joint names to robotJoints indices
        private string _resolvedTrajectoryTopic;
        private string _resolvedFeedbackTopic;
        private bool _abortingForPreempt; // Suppresses OnTrajectoryComplete(false) on preempt

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

            if (_controlModeManager == null)
                _controlModeManager = GetComponentInParent<ROSControlModeManager>();

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
                _abortingForPreempt = true;
                AbortExecution();
                _abortingForPreempt = false;
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

            // Set IsManuallyDriven immediately (before the coroutine's first frame) so
            // the RobotController IK never fires between consecutive trajectories.
            _robotController.IsManuallyDriven = true;
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

                // Check velocities — MoveIt plans at full speed; Unity executes at
                // _speedScaling, so the physical velocity = planned * _speedScaling.
                // Warn once per joint (not per point) to avoid log spam.
                if (point.velocities != null && p == 0)
                {
                    float effectiveLimit = _maxJointVelocity / _speedScaling;
                    for (int j = 0; j < point.velocities.Length; j++)
                    {
                        if (Mathf.Abs((float)point.velocities[j]) > effectiveLimit)
                        {
                            Debug.LogWarning(
                                $"{_logPrefix} Joint {j} planned velocity {point.velocities[j]:F2} rad/s "
                                + $"exceeds scaled limit {effectiveLimit:F2} — will be clamped during execution."
                            );
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

            // Read actual joint positions as the trajectory start to avoid a jump
            // when the arm hasn't fully settled at the end of a previous trajectory.
            // Using planned positions as fromPositions would cause a discontinuity
            // if physics lag left the joints a few degrees off.
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

                // Always interpolate from actual physics state for the first segment.
                // For subsequent segments use the previous planned point so the overall
                // timing matches MoveIt's plan; the first-segment correction absorbs any
                // residual physics lag from the previous trajectory.
                double[] fromPositions;
                if (prevPoint != null && prevPoint.positions != null)
                    fromPositions = prevPoint.positions;
                else
                    fromPositions = startPositions;

                double rawSegmentDuration = targetTime - prevPointTime;
                if (rawSegmentDuration <= 0)
                    rawSegmentDuration = Time.fixedDeltaTime;

                // Wall-clock duration after applying speed scaling (e.g. 0.5x → 2x wall time).
                double segmentDuration = rawSegmentDuration / _speedScaling;

                // Collect from/to velocities for cubic Hermite interpolation.
                // MoveIt plans velocity-continuous trajectories; using these velocities
                // gives C1-continuous motion (no velocity jumps at waypoints).
                double[] fromVelocities = prevPoint?.velocities;
                double[] toVelocities = targetPoint.velocities;

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

                            double p0 = fromPositions[j];
                            double p1 = targetPoint.positions[j];
                            double interpRad;

                            // Use cubic Hermite when velocity data is available on both ends.
                            // This matches MoveIt's trajectory intent and eliminates velocity
                            // discontinuities at waypoints (the main source of visible jerks).
                            if (fromVelocities != null && toVelocities != null
                                && j < fromVelocities.Length && j < toVelocities.Length)
                            {
                                // Convert MoveIt rad/s velocities to normalised-time tangents.
                                // The Hermite curve is parameterised over t ∈ [0,1] mapping to
                                // rawSegmentDuration seconds of MoveIt time. The tangent at each
                                // endpoint must equal velocity_rad_s * rawSegmentDuration (the
                                // positional change at that velocity over the MoveIt segment).
                                // Speed scaling stretches wall-clock time but does NOT change the
                                // shape of the position curve — only when t reaches 1.0. Using
                                // rawSegmentDuration (not the scaled wall-clock value) keeps the
                                // curve shape identical to MoveIt's intent.
                                double v0 = fromVelocities[j] * rawSegmentDuration;
                                double v1 = toVelocities[j]   * rawSegmentDuration;
                                double t2 = t * t;
                                double t3 = t2 * t;
                                // Standard cubic Hermite basis functions
                                interpRad = (2*t3 - 3*t2 + 1) * p0
                                          + (t3 - 2*t2 + t)   * v0
                                          + (-2*t3 + 3*t2)    * p1
                                          + (t3 - t2)         * v1;
                            }
                            else
                            {
                                // Fallback: linear interpolation
                                interpRad = p0 + (p1 - p0) * t;
                            }

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

            // Wait for all joint velocities to physically settle below the threshold.
            // "Completed" feedback must NOT fire until the arm has actually stopped moving —
            // otherwise the Python side begins planning the next trajectory (grasp descent)
            // while the arm is still oscillating, causing a jerk at every waypoint handoff.
            // Scale the timeout inversely with speed scaling: at 0.5x speed the arm covers
            // the same distance more slowly and needs proportionally longer to damp out.
            float effectiveSettleTimeout = _settleTimeoutSeconds / Mathf.Max(0.1f, _speedScaling);
            float settleStartTime = Time.time;
            bool settled = false;
            while (Time.time - settleStartTime < effectiveSettleTimeout)
            {
                settled = true;
                for (int j = 0; j < _jointIndexMap.Length; j++)
                {
                    int idx = _jointIndexMap[j];
                    float velDegPerSec = _joints[idx].jointVelocity.dofCount > 0
                        ? Mathf.Abs(_joints[idx].jointVelocity[0]) * Mathf.Rad2Deg
                        : 0f;
                    if (velDegPerSec > _settleVelocityThresholdDegPerSec)
                    {
                        settled = false;
                        break;
                    }
                }
                if (settled)
                    break;
                yield return new WaitForFixedUpdate();
            }

            float totalDuration = Time.time - startTime;
            string settleStatus = settled ? "OK" : "timeout";
            string logMessage =
                $"{_logPrefix} Trajectory completed in {totalDuration:F2}s "
                + $"(settle: {settleStatus}) for {_robotController.robotId}";
            if (settled)
                Debug.Log(logMessage);
            else
                Debug.LogWarning(logMessage);

            // Only sync the IK target when the control mode is ROS or Hybrid.
            // In those modes Unity IK is normally suppressed; when it re-engages it must
            // start from the current pose (zero error) or it will fight the ROS position.
            // In Unity mode the IK owns motion end-to-end and manages its own target, so
            // we must NOT overwrite it here.
            bool isROSControlled = _controlModeManager == null
                || _controlModeManager.CurrentMode != ControlMode.Unity;

            if (isROSControlled)
            {
                // Sync the IK solver's target to the current end-effector position BEFORE
                // re-enabling Unity IK (IsManuallyDriven = false). Without this, the IK wakes
                // up with a stale _targetTransform pointing to wherever the arm was going before
                // the ROS trajectory started, and immediately drives the arm away from the
                // ROS-delivered pose — the main cause of post-grasp jitter.
                // Also clears moving-target tracking so a kinematic held object doesn't
                // create a feedback loop (object moves → tracker re-arms IK → arm moves → repeat).
                _robotController.SyncIKTargetToCurrentPose();
            }

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

            Debug.Log($"{_logPrefix} Trajectory execution aborted for {_robotController.robotId}");

            if (!_abortingForPreempt)
            {
                PublishFeedback("aborted", "Trajectory execution aborted");
                OnTrajectoryComplete?.Invoke(false);
            }
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
