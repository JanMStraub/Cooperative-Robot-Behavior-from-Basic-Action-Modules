using System.Collections;
using System.Collections.Generic;
using Configuration;
using Core;
using Robotics.Grasp;
using RobotScripts;
using Simulation;
using UnityEngine;
using Utilities;

namespace Robotics
{
    /// <summary>
    /// Controls a robotic arm using inverse kinematics (IK) with a damped least squares approach.
    /// Highly optimized for low Garbage Collection (GC) and minimal physics overhead.
    /// </summary>
    public class RobotController : MonoBehaviour
    {
        private SimulationManager _simulationManager;
        private RobotManager _robotManager;

        [Header("Robot Identity")]
        public string robotId = "Robot1";

        // --- Performance Caching ---
        private Transform _ikFrame;
        private JointInfo[] _cachedJointInfos;
        private ArticulationDrive[] _cachedDriveUpdates;
        private bool _cachedIsRobotActive = true;
        private float _lastActiveCheckTime = 0f;
        private const float ACTIVE_CHECK_CACHE_INTERVAL = 0.1f;

        // ---------------------------

        [Header("IK Parameters")]
        [SerializeField]
        private float _dampingFactorLambda = RobotConstants.DEFAULT_DAMPING_FACTOR;
        private float _minStepSpeedNearTarget = RobotConstants.MIN_STEP_SPEED_NEAR_TARGET;
        private float _maxStepSpeed = RobotConstants.MAX_STEP_SPEED;

        private IKSolver _ikSolver;
        private TrajectoryController _trajectoryController;

        // Trajectory following state
        private CartesianPath _currentPath;
        private bool _followingPath = false;
        private float _pathStartTime;
        private VelocityProfile _velocityProfile;

        // GC optimization: Cached waypoint list to prevent allocations
        private List<CartesianWaypoint> _cachedWaypointList = new List<CartesianWaypoint>(50);

        // State variables
        private float _sqrDistanceToTarget; // Optimization: Use Squared Distance
        private float _distanceToTarget; // Cached for when actual distance is needed
        private bool _hasReachedTarget = true;
        private bool _isGraspingTarget = false;
        private Transform _targetTransform;

        // IK State cache (reusable structs)
        private Vector3 _endEffectorLocalPosition;
        private Quaternion _endEffectorLocalRotation;
        private Vector3 _targetLocalPosition;
        private Quaternion _targetLocalRotation;

        // Target Objects
        private GameObject _targetObject;
        public GameObject debugTarget;

        // Reusable Temporary Targets
        private GameObject _cachedTempTargetMove;
        private GameObject _cachedTempTargetGrasp;
        private GameObject _cachedTempTargetPre;
        private GameObject _cachedTempTargetRetreat;

        private Coroutine _activeGraspCoroutine;

        [Header("Robot Components")]
        public ArticulationBody[] robotJoints;
        public Transform endEffectorBase;
        public Transform IKReferenceFrame;
        public float[] jointDriveTargets;

        [Header("Gripper Integration")]
        [SerializeField]
        private GripperController _gripperController;
        public bool _closeGripperAfterReach = false;
        private GraspApproach _currentGraspApproach;

        [Header("Target Tracking")]
        [SerializeField]
        private bool _enableMovingTargetTracking = true;

        [SerializeField]
        private float _targetMovementThreshold = 0.01f;
        private float _sqrTargetMovementThreshold;
        private Vector3 _lastTrackedTargetPosition;
        private bool _isTrackingMovingTarget = false;

        // Moving target smoothing
        private Vector3 _smoothedTargetPosition;
        private Vector3 _targetPositionVelocity;
        private const float TARGET_SMOOTHING_TIME = 0.2f;

        [Header("Advanced Grasp Planning")]
        [SerializeField]
        private GraspConfig _graspConfig;
        private GraspPlanningPipeline _graspPipeline;

        [Header("Debug Visualization")]
        [SerializeField]
        private bool enableDebugVisualization = true;

        private const string _logPrefix = "[ROBOT_CONTROLLER]";

        public event System.Action OnTargetReached;
        public event System.Action<bool> OnCoordinationStateChanged;

        private void Start()
        {
            _simulationManager = SimulationManager.Instance;
            _robotManager = RobotManager.Instance;

            if (string.IsNullOrEmpty(robotId))
                robotId = gameObject.name;

            _ikFrame = IKReferenceFrame != null ? IKReferenceFrame : endEffectorBase.root;

            if (_gripperController == null)
                _gripperController = GetComponentInChildren<GripperController>();

            if (_robotManager != null && !_robotManager.IsRobotRegistered(robotId))
                _robotManager.RegisterRobot(robotId, gameObject);

            InitializeRobot();
            InitializeGraspPipeline();

            if (debugTarget != null)
                SetTarget(debugTarget, GraspOptions.Advanced);
        }

        private void InitializeRobot()
        {
            if (robotJoints == null || robotJoints.Length == 0)
                return;

            int jointCount = robotJoints.Length;

            jointDriveTargets = new float[jointCount];
            _cachedJointInfos = new JointInfo[jointCount];
            _cachedDriveUpdates = new ArticulationDrive[jointCount];

            for (int i = 0; i < jointCount; i++)
            {
                var drive = robotJoints[i].xDrive;

                if (
                    _robotManager != null
                    && _robotManager.RobotInstances.TryGetValue(robotId, out var robotInstance)
                    && robotInstance.profile?.joints != null
                    && i < robotInstance.profile.joints.Length
                )
                {
                    var jointConfig = robotInstance.profile.joints[i];
                    drive.stiffness = jointConfig.stiffness;
                    drive.damping = jointConfig.damping;
                    drive.forceLimit = jointConfig.forceLimit;
                    drive.upperLimit = jointConfig.upperLimit;
                    drive.lowerLimit = jointConfig.lowerLimit;
                }

                robotJoints[i].xDrive = drive;
                jointDriveTargets[i] = drive.target;
            }

            _ikSolver = new IKSolver(jointCount, _dampingFactorLambda);
            _trajectoryController = new TrajectoryController(
                positionGains: new Vector3(10f, 10f, 10f), // Kp for position
                velocityGains: new Vector3(2f, 2f, 2f)      // Kd for velocity damping
            );
            _gripperController?.ResetGrippers();
            _sqrTargetMovementThreshold = _targetMovementThreshold * _targetMovementThreshold;
        }

        private void InitializeGraspPipeline()
        {
            if (
                _graspConfig != null
                && robotJoints != null
                && _ikFrame != null
                && endEffectorBase != null
            )
            {
                _graspPipeline = new GraspPlanningPipeline(
                    _graspConfig,
                    robotJoints,
                    _ikFrame,
                    endEffectorBase,
                    _dampingFactorLambda
                );
            }
        }

        // --- Core Helpers ---

        private GameObject GetCachedTempObject(ref GameObject cacheField, string suffix)
        {
            if (cacheField == null)
                cacheField = new GameObject($"{robotId}{suffix}");
            return cacheField;
        }

        private void StopActiveGraspCoroutine()
        {
            if (_activeGraspCoroutine != null)
            {
                StopCoroutine(_activeGraspCoroutine);
                _activeGraspCoroutine = null;
            }
        }

        public void SetTargetReached(bool setting)
        {
            if (_hasReachedTarget != setting)
            {
                _hasReachedTarget = setting;
                _simulationManager?.NotifyTargetReached(robotId, setting);

                if (setting)
                {
                    // Stop following trajectory when target is reached
                    _followingPath = false;

                    OnTargetReached?.Invoke();
                    if (_closeGripperAfterReach && _gripperController != null)
                        _gripperController.CloseGrippers();
                }
            }
        }

        // --- Optimized Core Loop ---

        private void FixedUpdate()
        {
            if (_simulationManager != null && _simulationManager.ShouldStopRobots)
                return;
            if (_simulationManager != null && !GetCachedIsRobotActive())
                return;

            if (!_hasReachedTarget)
            {
                PerformInverseKinematicsStep();
            }

            DrawDebugVisualization();
        }

        private void UpdateJointInfoCache()
        {
            Quaternion ikFrameInverseRot = Quaternion.Inverse(_ikFrame.rotation);

            for (int i = 0; i < robotJoints.Length; i++)
            {
                var joint = robotJoints[i];
                Vector3 jointLocalPos = _ikFrame.InverseTransformPoint(joint.transform.position);

                // Optimized axis rotation
                Vector3 axisWorld = joint.transform.rotation * joint.anchorRotation * Vector3.right;
                Vector3 axisLocal = ikFrameInverseRot * axisWorld;

                _cachedJointInfos[i] = new JointInfo(jointLocalPos, axisLocal.normalized);
            }
        }

        /// <summary>
        /// Get end effector velocity in IK frame coordinates.
        /// Computes velocity from ArticulationBody joint velocities using forward kinematics.
        /// </summary>
        private Vector3 GetEndEffectorVelocity()
        {
            if (robotJoints == null || robotJoints.Length == 0 || endEffectorBase == null)
                return Vector3.zero;

            Vector3 endEffectorWorldVelocity = Vector3.zero;

            // Compute velocity contribution from each joint: v = omega x r
            for (int i = 0; i < robotJoints.Length; i++)
            {
                var joint = robotJoints[i];

                // Get joint angular velocity (rad/sec)
                float jointAngularVel = joint.jointVelocity[0];

                // Get joint axis in world space
                Vector3 axisWorld = joint.transform.rotation * joint.anchorRotation * Vector3.right;

                // Get vector from joint to end effector
                Vector3 jointToEE = endEffectorBase.position - joint.transform.position;

                // Velocity contribution: v = omega x r
                Vector3 velContribution = Vector3.Cross(axisWorld * jointAngularVel, jointToEE);
                endEffectorWorldVelocity += velContribution;
            }

            // Transform to IK frame local coordinates
            return Quaternion.Inverse(_ikFrame.rotation) * endEffectorWorldVelocity;
        }

        private void UpdateEndEffectorState()
        {
            if (_targetTransform == null)
            {
                _sqrDistanceToTarget = 0f;
                _distanceToTarget = 0f;
                return;
            }

            bool hasGrippedTarget = _hasReachedTarget && _closeGripperAfterReach;
            if (
                _enableMovingTargetTracking
                && _isTrackingMovingTarget
                && _targetObject != null
                && !hasGrippedTarget
            )
            {
                Vector3 currentTargetPos = _targetObject.transform.position;

                _smoothedTargetPosition = Vector3.SmoothDamp(
                    _smoothedTargetPosition,
                    currentTargetPos,
                    ref _targetPositionVelocity,
                    TARGET_SMOOTHING_TIME
                );

                if (
                    Vector3.SqrMagnitude(_smoothedTargetPosition - _lastTrackedTargetPosition)
                    > _sqrTargetMovementThreshold
                ) // 0.02^2
                {
                    if (_targetTransform != _targetObject.transform)
                    {
                        _targetTransform.position = _smoothedTargetPosition;
                        _targetTransform.rotation = _targetObject.transform.rotation;
                    }
                    _lastTrackedTargetPosition = _smoothedTargetPosition;
                    if (_hasReachedTarget)
                        SetTargetReached(false);
                }
            }

            // Calculation optimizations
            Vector3 eeWorldPos = endEffectorBase.position;
            Quaternion eeWorldRot = endEffectorBase.rotation;
            Vector3 targetWorldPos = _targetTransform.position;
            Quaternion targetWorldRot = _targetTransform.rotation;

            _endEffectorLocalPosition = _ikFrame.InverseTransformPoint(eeWorldPos);
            _targetLocalPosition = _ikFrame.InverseTransformPoint(targetWorldPos);

            Quaternion invFrameRot = Quaternion.Inverse(_ikFrame.rotation);
            _endEffectorLocalRotation = invFrameRot * eeWorldRot;
            _targetLocalRotation = invFrameRot * targetWorldRot;

            Vector3 distVec = _targetLocalPosition - _endEffectorLocalPosition;
            _sqrDistanceToTarget = distVec.sqrMagnitude;
            _distanceToTarget = Mathf.Sqrt(_sqrDistanceToTarget); // Only calc sqrt once per frame
        }

        public void PerformInverseKinematicsStep()
        {
            if (robotJoints.Length == 0 || endEffectorBase == null || _targetTransform == null)
                return;

            UpdateEndEffectorState();

            // Thresholds
            float posThreshold = _isGraspingTarget
                ? RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD
                    * RobotConstants.GRASP_CONVERGENCE_MULTIPLIER
                : RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD;

            // Optimization: Compare Squares to avoid SQRT in logic
            float sqrPosThreshold = posThreshold * posThreshold;

            float angleError = Quaternion.Angle(_endEffectorLocalRotation, _targetLocalRotation);
            bool isOrientationCorrect =
                angleError < RobotConstants.DEFAULT_ORIENTATION_THRESHOLD_DEGREES;

            // Get current velocity for convergence check
            Vector3 currentVelocity = GetEndEffectorVelocity();

            // Adaptive velocity threshold: relax during grasp operations
            // Grasping requires position precision but can tolerate higher velocity
            float velocityThreshold = _isGraspingTarget ? 0.15f : 0.05f; // 15cm/s for grasp, 5cm/s normal
            bool isVelocityLow = currentVelocity.magnitude < velocityThreshold;

            // Stop Condition - require position, orientation, AND velocity convergence
            // For very precise grasps (< 2cm), prioritize position over velocity
            bool isPreciseEnough = _sqrDistanceToTarget <= sqrPosThreshold && isOrientationCorrect;
            bool isFullySettled = isPreciseEnough && isVelocityLow;

            // Allow convergence if position is precise enough, even if velocity is still settling
            // This prevents hovering during grasp operations while maintaining stability for normal moves
            if (isFullySettled || (isPreciseEnough && _isGraspingTarget && _distanceToTarget < posThreshold * 0.5f))
            {
                if (!_hasReachedTarget)
                {
                    Debug.Log($"{_logPrefix} [{robotId}] ✅ TARGET REACHED! Dist: {_distanceToTarget:F4}m, Vel: {currentVelocity.magnitude:F3}m/s, Grasping: {_isGraspingTarget}");
                    SetTargetReached(true);
                    OnTargetReached?.Invoke();
                }
                return;
            }
            else if (enableDebugVisualization && !_hasReachedTarget && _distanceToTarget < posThreshold * 2f && Time.frameCount % 60 == 0)
            {
                // Debug: Log why we're NOT converging when close to target (throttled to once per second)
                Debug.Log($"{_logPrefix} [{robotId}] Near target but not converged - Dist: {_distanceToTarget:F4}m (thresh: {posThreshold:F4}m), " +
                         $"Vel: {currentVelocity.magnitude:F3}m/s (thresh: {velocityThreshold:F3}m/s), " +
                         $"Angle: {angleError:F1}°, Grasping: {_isGraspingTarget}, " +
                         $"PreciseEnough: {isPreciseEnough}, VelocityLow: {isVelocityLow}");
            }
            else if (
                _hasReachedTarget && (_distanceToTarget > posThreshold * 1.5f || angleError > 10f)
            )
            {
                SetTargetReached(false);
            }

            UpdateJointInfoCache();

            // --- Trajectory Following or Carrot Logic ---
            Vector3 constrainedTargetPos;
            Quaternion constrainedTargetRot;
            Vector3 targetVelocity = Vector3.zero;

            if (_followingPath && _currentPath != null && _velocityProfile != null)
            {
                // Use TrajectoryController for smooth motion with velocity profiles
                float currentTime = Time.fixedTime - _pathStartTime;

                // Get trajectory state (position, velocity, acceleration)
                var (trajPos, trajVel, trajAccel) = _trajectoryController.GetTrajectoryState(
                    currentTime,
                    _currentPath,
                    _velocityProfile
                );

                // Transform trajectory target to local IK frame
                constrainedTargetPos = _ikFrame.InverseTransformPoint(trajPos);
                targetVelocity = Quaternion.Inverse(_ikFrame.rotation) * trajVel;

                // For rotation, use final target (smooth interpolation handled by trajectory)
                constrainedTargetRot = _targetLocalRotation;

                // Stop following path when close to target (within 3cm) OR trajectory time exceeded
                // Earlier termination allows IK to take over for final precision approach
                float trajectoryDuration = _currentPath.totalDistance / (_velocityProfile.cruiseVelocity > 0 ? _velocityProfile.cruiseVelocity : 0.1f);
                bool trajectoryTimeExceeded = currentTime > trajectoryDuration * 1.2f;
                bool closeToTarget = _distanceToTarget < 0.03f; // 3cm threshold (was 5cm)

                if (trajectoryTimeExceeded || closeToTarget)
                {
                    Debug.Log($"{_logPrefix} [{robotId}] Stopping trajectory following - Time: {currentTime:F2}s, " +
                             $"Duration: {trajectoryDuration:F2}s, Dist: {_distanceToTarget:F3}m, Exceeded: {trajectoryTimeExceeded}, Close: {closeToTarget}");
                    _followingPath = false; // Switch to final convergence mode
                }
            }
            else
            {
                // Fallback: Direct targeting for final convergence (no carrot)
                // When close to target (<3cm), go directly for better precision and speed
                // The IK solver with velocity damping prevents overshoot
                constrainedTargetPos = _targetLocalPosition;

                // --- KEY FIX: ALWAYS target the final rotation, but vary the weight ---
                // This allows orientation ramping to work correctly
                constrainedTargetRot = _targetLocalRotation;
            }

            // --- Orientation Ramping ---
            float baseWeight = GetOrientationWeight(_currentGraspApproach);
            float orientationRamp = 0f;

            // Start ramping at 30cm instead of 20cm with linear curve for smoother transition
            if (_distanceToTarget < 0.30f)
            {
                // Between 30cm and 10cm - gradually ramp up orientation
                // At 30cm: ramp = 0 (ignore rotation)
                // At 10cm: ramp = 1 (full rotation control)
                float t = Mathf.Clamp01((_distanceToTarget - 0.10f) / 0.20f);
                orientationRamp = 1.0f - t; // Linear ramp from 0.3m to 0.1m (was quadratic: 1.0f - (t * t))
            }
            // else: > 30cm, orientationRamp stays 0 (ignore rotation completely)

            float finalWeight = baseWeight * orientationRamp;

            // --- Dynamic Damping (Precision) ---
            float dynamicLambda = _dampingFactorLambda;
            if (_distanceToTarget < 0.1f)
            {
                // Increase damping when closer for stability (not decrease)
                dynamicLambda = Mathf.Lerp(
                    _dampingFactorLambda,
                    _dampingFactorLambda * 1.5f,
                    1f - (_distanceToTarget / 0.1f)
                );
            }

            // Input States
            IKState currentState = new IKState(
                _endEffectorLocalPosition,
                _endEffectorLocalRotation
            );
            IKState targetState = new IKState(constrainedTargetPos, constrainedTargetRot);

            // Use velocity-level IK with PD control for better stability
            // PD gains: Kp (position gain) and Kd (velocity damping)
            // Higher Kp for faster convergence during final approach
            // When following trajectory, use trajectory velocity; otherwise zero for stationary target
            var jointDeltas = _ikSolver.ComputeJointDeltasWithVelocity(
                currentState,
                targetState,
                currentVelocity,
                targetVelocity, // From trajectory or zero for stationary target
                _cachedJointInfos,
                posThreshold,
                Kp: 2.0f, // Position gain (increased from 1.0 for faster convergence)
                Kd: 0.8f, // Velocity damping gain (increased proportionally for stability)
                orientationWeight: finalWeight,
                orientationConvergenceThreshold: RobotConstants.DEFAULT_ORIENTATION_THRESHOLD_DEGREES * Mathf.Deg2Rad,
                overrideDamping: dynamicLambda
            );

            // IKSolver should not return null anymore (convergence check disabled)
            // If it does, it means position/orientation are within threshold
            if (jointDeltas == null)
            {
                Debug.LogWarning($"{_logPrefix} [{robotId}] IKSolver returned null unexpectedly - skipping motor updates");
                return; // Skip motor updates, let main convergence check at top handle target reached
            }

            // Speed Calculation
            RobotConfig robotProfile =
                _robotManager?.GetRobotProfile(robotId) ?? _robotManager?.RobotProfile;
            float maxStep =
                robotProfile != null
                    ? robotProfile.maxJointStepRad
                    : RobotConstants.DEFAULT_MAX_JOINT_STEP_RAD;
            float adjSpeed = robotProfile != null ? robotProfile.adjustmentSpeed : 1.0f;
            float globalSpeed = _robotManager != null ? _robotManager.globalSpeedMultiplier : 1.0f;

            float adaptiveGain = _maxStepSpeed;
            if (_distanceToTarget < 0.1f)
            {
                adaptiveGain = Mathf.Lerp(
                    _minStepSpeedNearTarget,
                    _maxStepSpeed,
                    _distanceToTarget / 0.1f
                );
            }

            float stepScale = adjSpeed * globalSpeed * adaptiveGain * Mathf.Rad2Deg;

            // Apply Updates
            bool hasUpdates = false;

            for (int i = 0; i < robotJoints.Length; i++)
            {
                double rawDelta = System.Math.Clamp(jointDeltas[i], -maxStep, maxStep);
                ArticulationDrive drive = robotJoints[i].xDrive;
                float deltaAngleDegree = (float)rawDelta * stepScale;

                float newTarget = Mathf.Clamp(
                    drive.target + deltaAngleDegree,
                    drive.lowerLimit,
                    drive.upperLimit
                );

                // Epsilon check to minimize PhysX overhead
                if (Mathf.Abs(newTarget - drive.target) > Mathf.Epsilon)
                {
                    drive.target = newTarget;
                    _cachedDriveUpdates[i] = drive;
                    hasUpdates = true;
                }
                else
                {
                    _cachedDriveUpdates[i] = drive;
                }
            }

            if (hasUpdates)
            {
                for (int i = 0; i < robotJoints.Length; i++)
                {
                    if (
                        Mathf.Abs(robotJoints[i].xDrive.target - _cachedDriveUpdates[i].target)
                        > Mathf.Epsilon
                    )
                    {
                        robotJoints[i].xDrive = _cachedDriveUpdates[i];
                        if (i < jointDriveTargets.Length)
                            jointDriveTargets[i] = _cachedDriveUpdates[i].target;
                    }
                }
            }

            // Velocity Damping - REMOVED: Let IK solver control motion
            // The physics velocity damping was conflicting with IK commands
            // causing poor convergence. IK solver now handles all motion control.
        }

        // --- Target Setting (Refactored) ---

        private void SetTargetInternal(
            Transform targetTransform,
            GameObject originalObject,
            GraspOptions options
        )
        {
            _targetObject = originalObject;
            _targetTransform = targetTransform;
            _closeGripperAfterReach = options.closeGripperOnReach;
            SetTargetReached(false);

            _isTrackingMovingTarget = (originalObject != null);
            if (_isTrackingMovingTarget)
            {
                _lastTrackedTargetPosition = originalObject.transform.position;
            }

            _ikSolver?.ResetIterationCount();

            // Generate trajectory for smooth motion
            GenerateTrajectoryToTarget();
        }

        /// <summary>
        /// Generate a CartesianPath and VelocityProfile for smooth motion to target.
        /// Uses TrajectoryController for PD control with feedforward terms.
        /// </summary>
        private void GenerateTrajectoryToTarget()
        {
            if (_targetTransform == null || endEffectorBase == null)
                return;

            // Get start and end positions in world space
            Vector3 startPos = endEffectorBase.position;
            Vector3 endPos = _targetTransform.position;

            // Create simple straight-line path with waypoints
            float distance = Vector3.Distance(startPos, endPos);
            const float waypointSpacing = 0.05f; // 5cm waypoints
            int numWaypoints = Mathf.Max(2, Mathf.CeilToInt(distance / waypointSpacing));

            // GC OPTIMIZATION: Clear and reuse cached list instead of allocating new
            _cachedWaypointList.Clear();
            for (int i = 0; i < numWaypoints; i++)
            {
                float t = i / (float)(numWaypoints - 1);
                Vector3 pos = Vector3.Lerp(startPos, endPos, t);
                Quaternion rot = Quaternion.Slerp(endEffectorBase.rotation, _targetTransform.rotation, t);
                float dist = t * distance;

                _cachedWaypointList.Add(new CartesianWaypoint
                {
                    position = pos,
                    rotation = rot,
                    distanceFromStart = dist,
                    timeFromStart = 0f // Will be calculated by velocity profile
                });
            }

            _currentPath = new CartesianPath
            {
                waypoints = _cachedWaypointList, // Reference cached list - no allocation!
                totalDistance = distance,
                maxVelocity = 0.5f,
                acceleration = 1.0f
            };

            // Generate trapezoidal velocity profile
            _velocityProfile = VelocityProfile.CreateTrapezoidal(
                distance,
                maxVelocity: 0.5f, // 0.5 m/s max
                acceleration: 1.0f  // 1.0 m/s^2
            );

            // Reset trajectory tracking
            _followingPath = true;
            _pathStartTime = Time.fixedTime;
            _trajectoryController?.Reset();
        }

        public void SetTarget(GameObject target, GraspOptions options = default)
        {
            if (target == null)
                return;
            StopActiveGraspCoroutine();

            if (options.Equals(default(GraspOptions)))
                options = GraspOptions.Default;

            // Check Advanced/Simple Planning
            if (_graspPipeline != null)
            {
                GraspCandidate? candidate = null;

                if (options.useAdvancedPlanning)
                {
                    if (options.graspConfig == null)
                        options.graspConfig = _graspConfig;
                    candidate = _graspPipeline.PlanGrasp(target, endEffectorBase.position, options);
                    if (candidate.HasValue)
                    {
                        _activeGraspCoroutine = StartCoroutine(
                            ExecuteThreeWaypointGrasp(candidate.Value, target, options)
                        );
                        return;
                    }
                }
                else if (options.useGraspPlanning)
                {
                    // Create minimal options for simple planner
                    var simpleOpts = new GraspOptions
                    {
                        useGraspPlanning = false,
                        approach = options.approach,
                        graspConfig = options.graspConfig,
                    };
                    candidate = _graspPipeline.PlanGrasp(
                        target,
                        endEffectorBase.position,
                        simpleOpts
                    );
                    if (candidate.HasValue)
                    {
                        _activeGraspCoroutine = StartCoroutine(
                            ExecuteTwoWaypointGrasp(candidate.Value, target, options)
                        );
                        return;
                    }
                }
            }

            // Fallback: Direct Move
            if (options.openGripperOnSet)
                _gripperController?.OpenGrippers();
            _isGraspingTarget = false;
            SetTargetInternal(target.transform, target, options);
        }

        public void SetTarget(Vector3 position, GraspOptions options = default)
        {
            StopActiveGraspCoroutine();
            if (options.Equals(default(GraspOptions)))
                options = GraspOptions.MoveOnly;

            // Object Finder Logic
            if (ObjectFinder.Instance != null)
            {
                GameObject realObject = ObjectFinder.Instance.FindClosestObject(
                    position,
                    RobotConstants.OBJECT_FINDING_RADIUS
                );
                if (realObject != null)
                {
                    if (
                        Vector3.SqrMagnitude(position - realObject.transform.position)
                        < RobotConstants.OBJECT_DISTANCE_THRESHOLD
                            * RobotConstants.OBJECT_DISTANCE_THRESHOLD
                    )
                    {
                        SetTarget(realObject, options);
                        return;
                    }
                }
            }

            GameObject temp = GetCachedTempObject(
                ref _cachedTempTargetMove,
                RobotConstants.TEMP_TARGET_SUFFIX
            );
            temp.transform.position = position;

            options.useGraspPlanning = false;
            if (options.openGripperOnSet)
                _gripperController?.OpenGrippers();

            _isGraspingTarget = false;
            SetTargetInternal(temp.transform, null, options);
        }

        public void SetTarget(Vector3 position, Quaternion rotation, GraspOptions options = default)
        {
            StopActiveGraspCoroutine();
            if (options.Equals(default(GraspOptions)))
                options = GraspOptions.MoveOnly;

            GameObject temp = GetCachedTempObject(
                ref _cachedTempTargetMove,
                RobotConstants.TEMP_TARGET_SUFFIX
            );
            temp.transform.SetPositionAndRotation(position, rotation);

            options.useGraspPlanning = false;
            if (options.openGripperOnSet)
                _gripperController?.OpenGrippers();

            _isGraspingTarget = false;
            SetTargetInternal(temp.transform, null, options);
        }

        // --- Grasp Coroutines ---

        /// <summary>
        /// Helper coroutine that waits for target to be reached with timeout protection.
        /// Prevents infinite waiting when target is unreachable.
        /// </summary>
        /// <param name="timeoutSeconds">Maximum time to wait before giving up</param>
        /// <param name="onTimeout">Optional callback to invoke on timeout</param>
        private IEnumerator WaitForTargetWithTimeout(
            float timeoutSeconds,
            System.Action onTimeout = null
        )
        {
            float startTime = Time.time;
            while (!_hasReachedTarget)
            {
                if (Time.time - startTime > timeoutSeconds)
                {
                    Debug.LogWarning(
                        $"{_logPrefix} {robotId} reached target timeout after {timeoutSeconds}s at distance {_distanceToTarget:F3}m"
                    );
                    onTimeout?.Invoke();
                    yield break;
                }
                yield return null;
            }
        }

        private IEnumerator ExecuteTwoWaypointGrasp(
            GraspCandidate candidate,
            GameObject targetObject,
            GraspOptions options
        )
        {
            _currentGraspApproach = candidate.approachType;
            if (_gripperController != null)
                _gripperController.SetGripperPosition(candidate.preGraspGripperWidth);

            // 1. Pre-Grasp
            GameObject pre = GetCachedTempObject(ref _cachedTempTargetPre, "_pre");
            pre.transform.SetPositionAndRotation(
                candidate.preGraspPosition,
                candidate.preGraspRotation
            );

            _isGraspingTarget = false;
            SetTargetInternal(
                pre.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return StartCoroutine(
                WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS)
            );
            if (!_hasReachedTarget)
                yield break; // Abort if timed out
            yield return new WaitForSeconds(0.2f);

            // 2. Grasp
            GameObject main = GetCachedTempObject(
                ref _cachedTempTargetGrasp,
                RobotConstants.GRASP_TARGET_SUFFIX
            );
            main.transform.SetPositionAndRotation(candidate.graspPosition, candidate.graspRotation);

            _isGraspingTarget = true;
            SetTargetInternal(
                main.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return StartCoroutine(
                WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS)
            );
            if (!_hasReachedTarget)
                yield break; // Abort if timed out

            if (options.closeGripperOnReach && _gripperController != null)
            {
                yield return new WaitForSeconds(0.3f);
                _gripperController.SetGripperPosition(candidate.graspGripperWidth);
            }

            _activeGraspCoroutine = null;
            OnTargetReached?.Invoke();
        }

        private IEnumerator ExecuteThreeWaypointGrasp(
            GraspCandidate candidate,
            GameObject targetObject,
            GraspOptions options
        )
        {
            _currentGraspApproach = candidate.approachType;
            if (_gripperController != null)
                _gripperController.SetGripperPosition(candidate.preGraspGripperWidth);

            // 1. Pre
            GameObject pre = GetCachedTempObject(ref _cachedTempTargetPre, "_pre");
            pre.transform.SetPositionAndRotation(
                candidate.preGraspPosition,
                candidate.preGraspRotation
            );

            _isGraspingTarget = false;
            SetTargetInternal(
                pre.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return StartCoroutine(
                WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS)
            );
            if (!_hasReachedTarget)
                yield break; // Abort if timed out
            yield return new WaitForSeconds(0.1f);

            // 2. Grasp
            GameObject main = GetCachedTempObject(
                ref _cachedTempTargetGrasp,
                RobotConstants.GRASP_TARGET_SUFFIX
            );
            main.transform.SetPositionAndRotation(candidate.graspPosition, candidate.graspRotation);

            _isGraspingTarget = true;
            SetTargetInternal(
                main.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return StartCoroutine(
                WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS)
            );
            if (!_hasReachedTarget)
                yield break; // Abort if timed out

            if (options.closeGripperOnReach && _gripperController != null)
            {
                yield return new WaitForSeconds(0.1f);
                _gripperController.SetGripperPosition(candidate.graspGripperWidth);
                yield return new WaitWhile(() => _gripperController.IsMoving);
                yield return new WaitForSeconds(1.1f);
            }

            // 3. Retreat
            if (options.graspConfig != null && options.graspConfig.enableRetreat)
            {
                GameObject retreat = GetCachedTempObject(ref _cachedTempTargetRetreat, "_retreat");
                retreat.transform.SetPositionAndRotation(
                    candidate.retreatPosition,
                    candidate.retreatRotation
                );

                _isGraspingTarget = false;
                SetTargetInternal(
                    retreat.transform,
                    targetObject,
                    new GraspOptions { closeGripperOnReach = false }
                );
                yield return StartCoroutine(
                    WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS)
                );
                if (!_hasReachedTarget)
                    yield break; // Abort if timed out
            }

            _activeGraspCoroutine = null;
            OnTargetReached?.Invoke();
        }

        // --- Utilities ---

        public float GetDistanceToTarget()
        {
            if (_targetTransform == null)
                return 0f;
            // Return the cached value if this frame already updated, otherwise quick calc
            return _distanceToTarget;
        }

        public Vector3? GetCurrentTarget() => _targetTransform?.position;

        public Quaternion? GetCurrentTargetRotation() => _targetTransform?.rotation;

        public bool HasTarget => _targetTransform != null;

        public GameObject GetTargetObject() => _targetObject;

        public void SetMovingTargetTracking(bool enable) => _enableMovingTargetTracking = enable;

        public void SetTargetMovementThreshold(float threshold)
        {
            _targetMovementThreshold = Mathf.Max(0.001f, threshold);
            _sqrTargetMovementThreshold = _targetMovementThreshold * _targetMovementThreshold;
        }

        public bool IsTargetTrackingEnabled() => _enableMovingTargetTracking;

        public Vector3 GetCurrentEndEffectorPosition() =>
            endEffectorBase == null ? Vector3.zero : endEffectorBase.position;

        public void ResetJointTargets()
        {
            for (int i = 0; i < robotJoints.Length; i++)
            {
                var drive = robotJoints[i].xDrive;
                drive.target = 0;
                robotJoints[i].xDrive = drive;
                robotJoints[i].jointPosition = new ArticulationReducedSpace(0f);
                robotJoints[i].jointForce = new ArticulationReducedSpace(0f);
                robotJoints[i].jointVelocity = new ArticulationReducedSpace(0f);
                if (i < jointDriveTargets.Length)
                    jointDriveTargets[i] = 0;
            }
        }

        private bool GetCachedIsRobotActive()
        {
            float t = Time.time;
            if (t - _lastActiveCheckTime >= ACTIVE_CHECK_CACHE_INTERVAL)
            {
                bool newState = _simulationManager.IsRobotActive(robotId);
                if (newState != _cachedIsRobotActive)
                    OnCoordinationStateChanged?.Invoke(newState);
                _cachedIsRobotActive = newState;
                _lastActiveCheckTime = t;
            }
            return _cachedIsRobotActive;
        }

        private float GetOrientationWeight(GraspApproach approach)
        {
            switch (approach)
            {
                case GraspApproach.Top:
                    return 1.5f;
                case GraspApproach.Side:
                    return 0.7f;
                default:
                    return 1.0f;
            }
        }

        private void DrawDebugVisualization()
        {
            if (enableDebugVisualization && _targetObject != null && endEffectorBase != null)
                Debug.DrawLine(
                    endEffectorBase.position,
                    _targetObject.transform.position,
                    Color.blue
                );
        }

        private void OnDestroy()
        {
            if (_cachedTempTargetMove)
                Destroy(_cachedTempTargetMove);
            if (_cachedTempTargetGrasp)
                Destroy(_cachedTempTargetGrasp);
            if (_cachedTempTargetPre)
                Destroy(_cachedTempTargetPre);
            if (_cachedTempTargetRetreat)
                Destroy(_cachedTempTargetRetreat);
        }
    }
}
