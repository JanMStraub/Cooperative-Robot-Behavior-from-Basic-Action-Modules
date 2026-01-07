using System.Collections;
using System.Linq;
using Configuration;
using Core;
using Robotics.Grasp;
using RobotScripts;
using Simulation;
using UnityEngine;
using Utilities;
using MathNet.Numerics.LinearAlgebra;

namespace Robotics
{
    public class RobotController : MonoBehaviour
    {
        // ... (Keep all existing variables and Setup code exactly as is) ...

        // [PASTE THE TOP PART OF YOUR SCRIPT HERE UNTIL ApplyMotorUpdates]
        // For brevity, I am only showing the changed methods below.

        private SimulationManager _simulationManager;
        private RobotManager _robotManager;
        public string robotId = "Robot1";
        private Transform _ikFrame;
        private JointInfo[] _cachedJointInfos;
        private ArticulationDrive[] _cachedDriveUpdates;
        private bool _cachedIsRobotActive = true;
        private float _lastActiveCheckTime = 0f;
        private const float ACTIVE_CHECK_CACHE_INTERVAL = 0.1f;

        [SerializeField]
        private float _dampingFactorLambda = RobotConstants.DEFAULT_DAMPING_FACTOR;
        private float _minStepSpeedNearTarget = RobotConstants.MIN_STEP_SPEED_NEAR_TARGET;
        private float _maxStepSpeed = RobotConstants.MAX_STEP_SPEED;
        private IKSolver _ikSolver;
        private float _sqrDistanceToTarget;
        private float _distanceToTarget;
        private bool _hasReachedTarget = true;
        private bool _isGraspingTarget = false;
        private Transform _targetTransform;
        private Vector3 _endEffectorLocalPosition;
        private Quaternion _endEffectorLocalRotation;
        private Vector3 _targetLocalPosition;
        private Quaternion _targetLocalRotation;
        private GameObject _targetObject;
        public GameObject debugTarget;
        private GameObject _cachedTempTargetMove;
        private GameObject _cachedTempTargetGrasp;
        private GameObject _cachedTempTargetPre;
        private GameObject _cachedTempTargetRetreat;
        private Coroutine _activeGraspCoroutine;
        public ArticulationBody[] robotJoints;
        public Transform endEffectorBase;
        public Transform IKReferenceFrame;
        public float[] jointDriveTargets;

        [SerializeField]
        private GripperController _gripperController;

        // NEW: Contact sensor for grasp verification
        private GripperContactSensor _contactSensor;

        public bool _closeGripperAfterReach = false;
        private GraspApproach _currentGraspApproach;

        [SerializeField]
        private bool _enableMovingTargetTracking = true;

        [SerializeField]
        private float _targetMovementThreshold = 0.01f;
        private float _sqrTargetMovementThreshold;
        private Vector3 _lastTrackedTargetPosition;
        private bool _isTrackingMovingTarget = false;
        private Vector3 _smoothedTargetPosition;
        private Vector3 _targetPositionVelocity;
        private const float TARGET_SMOOTHING_TIME = 0.2f;

        [SerializeField]
        private GraspConfig _graspConfig;
        private GraspPlanningPipeline _graspPipeline;
        private CartesianPath _currentPath;
        private bool _followingPath = false;
        private float _pathProgress = 0f;
        private VelocityProfile _velocityProfile;
        private float _pathStartTime;
        private float _maxCartesianVelocity = RobotConstants.DEFAULT_MAX_CARTESIAN_VELOCITY;
        private float _cartesianAcceleration = RobotConstants.DEFAULT_CARTESIAN_ACCELERATION;
        private float _waypointSpacing = RobotConstants.DEFAULT_WAYPOINT_SPACING;
        private bool _useCartesianControl = true;

        // NEW: Trajectory controller with PD control for velocity damping
        private TrajectoryController _trajectoryController;
        private bool _useVelocityLevelIK = true; // Feature flag for gradual rollout

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

            // NEW: Initialize contact sensor for grasp verification
            if (_contactSensor == null)
                _contactSensor = GetComponentInChildren<GripperContactSensor>();

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
            _gripperController?.ResetGrippers();
            _sqrTargetMovementThreshold = _targetMovementThreshold * _targetMovementThreshold;

            // NEW: Initialize trajectory controller with PD gains
            _trajectoryController = new TrajectoryController(
                positionGains: new Vector3(10f, 10f, 10f),  // K_p
                velocityGains: new Vector3(2f, 2f, 2f)      // K_d (damping)
            );

            // NEW: Tune ArticulationBody drives for critical damping
            TuneJointDrivesForCriticalDamping();
        }

        /// <summary>
        /// Tune ArticulationBody drives for critical damping to prevent oscillation.
        /// Uses formula: damping = 2 * sqrt(stiffness * inertia)
        /// This ensures the mechanical system is critically damped and won't oscillate.
        /// </summary>
        private void TuneJointDrivesForCriticalDamping()
        {
            if (robotJoints == null || robotJoints.Length == 0)
                return;

            Debug.Log($"{_logPrefix} [{robotId}] Tuning ArticulationBody drives for critical damping...");

            for (int i = 0; i < robotJoints.Length; i++)
            {
                var joint = robotJoints[i];

                // Estimate inertia from mass (simplified model)
                // For rotating link: I ≈ m * r² where r is distance to center of mass
                float mass = joint.mass;
                float inertia = mass * 0.1f; // Rough estimate for robot arm links

                // Target stiffness (reduced from 5000 to 2000 for less aggressive response)
                float targetStiffness = 2000f;

                // Critical damping formula: ζ = 1 when damping = 2 * sqrt(k * I)
                float criticalDamping = 2f * Mathf.Sqrt(targetStiffness * inertia);

                // Get current drive settings
                var drive = joint.xDrive;
                float oldStiffness = drive.stiffness;
                float oldDamping = drive.damping;

                // Apply new critically damped settings
                drive.stiffness = targetStiffness;
                drive.damping = criticalDamping;

                // ⚠️ IMPORTANT: Ensure physics anchor matches visual mesh
                // If visual/collider transforms are misaligned, IK will fight physics
                joint.matchAnchors = true; // Auto-compute from parent

                joint.xDrive = drive;

                Debug.Log($"{_logPrefix} Joint[{i}] tuned: stiffness {oldStiffness:F0}→{targetStiffness:F0}, " +
                         $"damping {oldDamping:F0}→{criticalDamping:F0}, " +
                         $"mass={mass:F2}kg, inertia={inertia:F4}");
            }

            Debug.Log($"{_logPrefix} [{robotId}] ✅ ArticulationBody tuning complete (critically damped)");
        }

        private void InitializeGraspPipeline()
        {
            if (
                _graspConfig != null
                && robotJoints != null
                && _ikFrame != null
                && endEffectorBase != null
            )
                _graspPipeline = new GraspPlanningPipeline(
                    _graspConfig,
                    robotJoints,
                    _ikFrame,
                    endEffectorBase,
                    _dampingFactorLambda
                );
        }

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
                    OnTargetReached?.Invoke();
                    if (_closeGripperAfterReach && _gripperController != null)
                        _gripperController.CloseGrippers();
                }
            }
        }

        // --- UPDATED METHOD 1: Faster Safe Speed ---
        private void ApplyMotorUpdates(double[] jointDeltas)
        {
            RobotConfig robotProfile =
                _robotManager?.GetRobotProfile(robotId) ?? _robotManager?.RobotProfile;

            // REDUCED from 0.15f to 0.05f (approx 2.9 degrees per frame)
            // Smaller steps prevent overshoot and oscillation
            // Combined with velocity damping, this ensures smooth convergence
            float maxStep = 0.05f;

            float adjSpeed = robotProfile != null ? robotProfile.adjustmentSpeed : 1.0f;
            float globalSpeed = _robotManager != null ? _robotManager.globalSpeedMultiplier : 1.0f;
            float adaptiveGain = _maxStepSpeed;

            if (_distanceToTarget < 0.1f)
                adaptiveGain = Mathf.Lerp(
                    _minStepSpeedNearTarget,
                    _maxStepSpeed,
                    _distanceToTarget / 0.1f
                );

            float speedFactor = adjSpeed * globalSpeed * adaptiveGain;
            bool hasUpdates = false;

            for (int i = 0; i < robotJoints.Length; i++)
            {
                if (double.IsNaN(jointDeltas[i]) || double.IsInfinity(jointDeltas[i]))
                    jointDeltas[i] = 0.0;

                double clampedDeltaRad = System.Math.Clamp(jointDeltas[i], -maxStep, maxStep);
                double finalDeltaRad = clampedDeltaRad * speedFactor;
                float deltaAngleDegree = (float)(finalDeltaRad * Mathf.Rad2Deg);

                ArticulationDrive drive = robotJoints[i].xDrive;
                float newTarget = Mathf.Clamp(
                    drive.target + deltaAngleDegree,
                    drive.lowerLimit,
                    drive.upperLimit
                );

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
        }

        // --- UPDATED METHOD 2: Smart Timeout ---
        private IEnumerator WaitForTargetWithTimeout(
            float timeoutSeconds,
            System.Action onTimeout = null
        )
        {
            float lastDist = _distanceToTarget;
            float timeSinceLastProgress = 0f;
            float stallTimeout = 4.0f; // If we don't move for 4 seconds, THEN fail.

            // Reset loop variables
            float absoluteStartTime = Time.time;

            while (!_hasReachedTarget)
            {
                // Progress Check: Are we getting closer?
                // If we moved at least 1cm closer, reset the stall timer.
                if (lastDist - _distanceToTarget > 0.01f)
                {
                    lastDist = _distanceToTarget;
                    timeSinceLastProgress = 0f;
                }
                else
                {
                    timeSinceLastProgress += Time.deltaTime;
                }

                // Hard limit: If the WHOLE operation takes > 30s, abort (safety).
                bool hardTimeout = (Time.time - absoluteStartTime > 30f);

                // Stall limit: If we haven't made progress in 4s, abort.
                bool stalled = (timeSinceLastProgress > stallTimeout);

                if (hardTimeout || stalled)
                {
                    string reason = hardTimeout ? "Hard Time Limit" : "Stalled";
                    Debug.LogWarning(
                        $"{_logPrefix} {robotId} Timeout ({reason}). Dist: {_distanceToTarget:F3}"
                    );
                    onTimeout?.Invoke();
                    yield break;
                }
                yield return null;
            }
        }

        // [KEEP THE REST OF THE SCRIPT EXACTLY AS IS]
        // Include FixedUpdate, PerformInverseKinematicsStep (with the Leash logic), SetTarget, etc.
        // Copy them from the previous correct version I gave you.

        // ... (Paste FixedUpdate, PerformInverseKinematicsStep, SetTarget, etc. here) ...

        private void FixedUpdate()
        {
            if (_simulationManager != null && _simulationManager.ShouldStopRobots)
                return;
            if (_simulationManager != null && !GetCachedIsRobotActive())
                return;
            if (!_hasReachedTarget)
                PerformInverseKinematicsStep();
            DrawDebugVisualization();
        }

        private void UpdateJointInfoCache()
        {
            Quaternion ikFrameInverseRot = Quaternion.Inverse(_ikFrame.rotation);
            for (int i = 0; i < robotJoints.Length; i++)
            {
                var joint = robotJoints[i];
                Vector3 jointLocalPos = _ikFrame.InverseTransformPoint(joint.transform.position);
                Vector3 axisWorld = joint.transform.rotation * joint.anchorRotation * Vector3.right;
                Vector3 axisLocal = ikFrameInverseRot * axisWorld;
                _cachedJointInfos[i] = new JointInfo(jointLocalPos, axisLocal.normalized);
            }
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
                    > 0.0004f
                )
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
            _distanceToTarget = Mathf.Sqrt(_sqrDistanceToTarget);
        }

        public void PerformInverseKinematicsStep()
        {
            if (robotJoints.Length == 0 || endEffectorBase == null || _targetTransform == null)
                return;
            UpdateEndEffectorState();
            float baseThreshold = _isGraspingTarget
                ? 0.005f
                : RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD;
            float posThreshold = _followingPath ? baseThreshold * 1.2f : baseThreshold;
            float sqrPosThreshold = posThreshold * posThreshold;
            float angleError = Quaternion.Angle(_endEffectorLocalRotation, _targetLocalRotation);
            bool isOrientationCorrect =
                angleError < RobotConstants.DEFAULT_ORIENTATION_THRESHOLD_DEGREES;
            if (_sqrDistanceToTarget <= sqrPosThreshold && isOrientationCorrect)
            {
                if (!_hasReachedTarget)
                {
                    Debug.Log(
                        $"{_logPrefix} Target reached! Dist: {Mathf.Sqrt(_sqrDistanceToTarget):F4}"
                    );
                    SetTargetReached(true);
                    OnTargetReached?.Invoke();
                }
                return;
            }
            else if (
                _hasReachedTarget && (_distanceToTarget > posThreshold * 2.0f || angleError > 15f)
            )
            {
                SetTargetReached(false);
            }
            UpdateJointInfoCache();
            Vector3 constrainedTargetPos;
            Quaternion constrainedTargetRot;
            if (_followingPath && _currentPath != null)
            {
                float elapsedTime = Time.time - _pathStartTime;
                float expectedDistance = CalculateDistanceFromTime(elapsedTime);
                _pathProgress = Mathf.Min(expectedDistance, _currentPath.totalDistance);
                if (_pathProgress >= _currentPath.totalDistance)
                {
                    constrainedTargetPos = _targetLocalPosition;
                    constrainedTargetRot = _targetLocalRotation;
                    _followingPath = false;
                    _currentPath = null;
                }
                else
                {
                    CartesianWaypoint currentWaypoint = _currentPath.GetWaypointAtDistance(
                        _pathProgress
                    );
                    constrainedTargetPos = currentWaypoint.position;
                    constrainedTargetRot = currentWaypoint.rotation;
                    float lagDistance = Vector3.Distance(
                        _endEffectorLocalPosition,
                        constrainedTargetPos
                    );
                    if (lagDistance > 0.05f)
                        _pathStartTime += Time.deltaTime;
                }
            }
            else
            {
                Vector3 vectorToTarget = _targetLocalPosition - _endEffectorLocalPosition;
                float blendFactor = Mathf.Clamp01((_distanceToTarget - 0.05f) / 0.10f);
                Vector3 carrotPos =
                    _endEffectorLocalPosition + (vectorToTarget / _distanceToTarget) * 0.05f;
                constrainedTargetPos = Vector3.Lerp(_targetLocalPosition, carrotPos, blendFactor);
                constrainedTargetRot = _targetLocalRotation;
            }
            float baseWeight = GetOrientationWeight(_currentGraspApproach);
            float orientationRamp = _followingPath ? 1.0f : 0f;
            if (!_followingPath && _distanceToTarget < 0.50f)
            {
                float t = Mathf.Clamp01((_distanceToTarget - 0.05f) / 0.45f);
                orientationRamp = 1.0f - (t * t * t);
            }
            float finalWeight = baseWeight * orientationRamp;
            float dynamicLambda = _dampingFactorLambda;
            if (_distanceToTarget < 0.01f && _isGraspingTarget)
                dynamicLambda *= 0.5f;
            else if (_distanceToTarget < 0.1f)
                dynamicLambda = Mathf.Lerp(
                    _dampingFactorLambda,
                    _dampingFactorLambda * 1.5f,
                    1f - (_distanceToTarget / 0.1f)
                );
            IKState currentState = new IKState(
                _endEffectorLocalPosition,
                _endEffectorLocalRotation
            );
            IKState targetState = new IKState(constrainedTargetPos, constrainedTargetRot);

            // NEW: Use velocity-level IK for natural damping (eliminates oscillation)
            Vector<double> jointDeltas;
            if (_useVelocityLevelIK && _trajectoryController != null)
            {
                // Get current end effector velocity
                Vector3 currentVelocity = GetEndEffectorVelocity();

                // Get target velocity from trajectory (if following path)
                Vector3 targetVelocity = Vector3.zero;
                if (_followingPath && _currentPath != null && _velocityProfile != null)
                {
                    var trajectoryState = _trajectoryController.GetTrajectoryState(
                        Time.time - _pathStartTime,
                        _currentPath,
                        _velocityProfile
                    );
                    targetVelocity = trajectoryState.targetVel;
                }

                // Compute joint deltas with velocity feedback (PD control)
                jointDeltas = _ikSolver.ComputeJointDeltasWithVelocity(
                    currentState,
                    targetState,
                    currentVelocity,
                    targetVelocity,
                    _cachedJointInfos,
                    posThreshold,
                    Kp: 1.0f,  // Position gain
                    Kd: 0.5f,  // Velocity gain (damping)
                    orientationWeight: finalWeight,
                    orientationConvergenceThreshold: RobotConstants.DEFAULT_ORIENTATION_THRESHOLD_DEGREES * Mathf.Deg2Rad,
                    overrideDamping: dynamicLambda
                );
            }
            else
            {
                // Fallback to original position-only IK
                jointDeltas = _ikSolver.ComputeJointDeltas(
                    currentState,
                    targetState,
                    _cachedJointInfos,
                    posThreshold,
                    finalWeight,
                    dynamicLambda
                );
            }
            if (jointDeltas == null)
            {
                if (_followingPath)
                    return;
                if (_distanceToTarget < 0.005f && isOrientationCorrect && !_hasReachedTarget)
                {
                    Debug.Log($"{_logPrefix} [{robotId}] Force snap (Precision Reach).");
                    SetTargetReached(true);
                    OnTargetReached?.Invoke();
                    return;
                }
                return;
            }
            ApplyMotorUpdates(jointDeltas.ToArray());
        }

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
                _lastTrackedTargetPosition = originalObject.transform.position;
            _ikSolver?.ResetIterationCount();
            GenerateCartesianPathIfNeeded(options);
        }

        private float CalculateDistanceFromTime(float time)
        {
            if (_velocityProfile == null)
                return 0f;
            float a = _velocityProfile.acceleration;
            float vMax = _velocityProfile.cruiseVelocity;
            float tAccel = vMax / a;
            if (time <= tAccel)
                return 0.5f * a * time * time;
            else if (_velocityProfile.cruisePhaseDistance > 0f)
            {
                float tCruiseEnd = tAccel + (_velocityProfile.cruisePhaseDistance / vMax);
                if (time <= tCruiseEnd)
                    return _velocityProfile.accelerationPhaseDistance + vMax * (time - tAccel);
                else
                {
                    float tDecel = time - tCruiseEnd;
                    return _velocityProfile.accelerationPhaseDistance
                        + _velocityProfile.cruisePhaseDistance
                        + (vMax * tDecel - 0.5f * a * tDecel * tDecel);
                }
            }
            else
            {
                float tTotal = 2f * tAccel;
                if (time <= tAccel)
                    return 0.5f * a * time * time;
                else if (time < tTotal)
                {
                    float tDecel = time - tAccel;
                    return _velocityProfile.accelerationPhaseDistance
                        + (vMax * tDecel - 0.5f * a * tDecel * tDecel);
                }
                else
                    return _currentPath.totalDistance;
            }
        }

        private void GenerateCartesianPathIfNeeded(GraspOptions options)
        {
            if (!_useCartesianControl || _targetTransform == null)
                return;
            Vector3 targetLocalPos = _ikFrame.InverseTransformPoint(_targetTransform.position);
            Quaternion targetLocalRot =
                Quaternion.Inverse(_ikFrame.rotation) * _targetTransform.rotation;
            Vector3 currentLocalPos = _ikFrame.InverseTransformPoint(endEffectorBase.position);
            Quaternion currentLocalRot =
                Quaternion.Inverse(_ikFrame.rotation) * endEffectorBase.rotation;
            float distance = Vector3.Distance(currentLocalPos, targetLocalPos);
            if (distance > RobotConstants.MIN_DISTANCE_FOR_CARTESIAN_PATH)
            {
                _currentPath = CartesianPathGenerator.GenerateLinearPath(
                    currentLocalPos,
                    currentLocalRot,
                    targetLocalPos,
                    targetLocalRot,
                    _waypointSpacing
                );
                _velocityProfile = VelocityProfile.CreateTrapezoidal(
                    _currentPath.totalDistance,
                    _maxCartesianVelocity,
                    _cartesianAcceleration
                );
                _followingPath = true;
                _pathProgress = 0f;
                _pathStartTime = Time.time;
                Debug.Log(
                    $"{_logPrefix} [{robotId}] Generated Cartesian path. Dist: {_currentPath.totalDistance:F3}m"
                );
            }
            else
            {
                _followingPath = false;
                _currentPath = null;
                Debug.Log($"{_logPrefix} [{robotId}] Distance {distance:F3}m too small for path.");
            }
        }

        public void SetTarget(GameObject target, GraspOptions options = default)
        {
            if (target == null)
                return;
            StopActiveGraspCoroutine();
            if (options.Equals(default(GraspOptions)))
                options = GraspOptions.Default;
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
            if (options.openGripperOnSet)
                _gripperController?.OpenGrippers();
            _isGraspingTarget = false;
            SetTargetInternal(target.transform, target, options);
        }

        // (Copy SetTarget overloads for Vector3 position/rotation from previous)
        public void SetTarget(Vector3 position, GraspOptions options = default)
        {
            StopActiveGraspCoroutine();
            if (options.Equals(default(GraspOptions)))
                options = GraspOptions.MoveOnly;
            if (ObjectFinder.Instance != null)
            {
                GameObject real = ObjectFinder.Instance.FindClosestObject(
                    position,
                    RobotConstants.OBJECT_FINDING_RADIUS
                );
                if (
                    real != null
                    && Vector3.SqrMagnitude(position - real.transform.position)
                        < RobotConstants.OBJECT_DISTANCE_THRESHOLD
                            * RobotConstants.OBJECT_DISTANCE_THRESHOLD
                )
                {
                    SetTarget(real, options);
                    return;
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

        private IEnumerator ExecuteTwoWaypointGrasp(
            GraspCandidate candidate,
            GameObject targetObject,
            GraspOptions options
        )
        {
            _currentGraspApproach = candidate.approachType;
            if (_gripperController != null)
                _gripperController.SetGripperPosition(candidate.preGraspGripperWidth);
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
                yield break;
            yield return new WaitForSeconds(0.2f);
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
                yield break;
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
            GameObject pre = GetCachedTempObject(ref _cachedTempTargetPre, "_pre");
            pre.transform.SetPositionAndRotation(
                candidate.preGraspPosition,
                candidate.preGraspRotation
            );
            _followingPath = false;
            _currentPath = null;
            _isGraspingTarget = false;
            SetTargetInternal(
                pre.transform,
                null,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return StartCoroutine(
                WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS)
            );
            if (!_hasReachedTarget)
                yield break;
            _targetTransform = null;
            _followingPath = false;
            _currentPath = null;
            yield return null;
            GameObject main = GetCachedTempObject(
                ref _cachedTempTargetGrasp,
                RobotConstants.GRASP_TARGET_SUFFIX
            );
            main.transform.SetPositionAndRotation(candidate.graspPosition, candidate.graspRotation);
            _isGraspingTarget = true;
            SetTargetInternal(
                main.transform,
                null,
                new GraspOptions { closeGripperOnReach = false }
            );

            // NEW: Monitor for early contact during approach
            float approachStartTime = Time.time;
            bool earlyContact = false;

            while (!_hasReachedTarget && Time.time - approachStartTime < RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS)
            {
                // Check for early contact
                if (_contactSensor != null && targetObject != null && _contactSensor.HasContact(targetObject))
                {
                    Debug.Log($"{_logPrefix} [{robotId}] Early contact detected during approach - stopping");
                    earlyContact = true;
                    SetTargetReached(true); // Force convergence
                    break;
                }
                yield return null;
            }

            if (!_hasReachedTarget && !earlyContact)
            {
                Debug.LogWarning($"{_logPrefix} [{robotId}] Approach failed - no target reach or contact");
                yield break;
            }

            // CLOSE GRIPPER
            if (options.closeGripperOnReach && _gripperController != null)
            {
                yield return new WaitForSeconds(0.1f);

                // Reset force history for clean grasp measurement
                if (_contactSensor != null)
                    _contactSensor.ResetForceHistory();

                _gripperController.SetGripperPosition(candidate.graspGripperWidth);
                yield return new WaitWhile(() => _gripperController.IsMoving);
                yield return new WaitForSeconds(0.3f); // Settle time for force measurement

                // NEW: Verify grasp success using contact sensor
                bool graspSuccess = VerifyGraspSuccess(targetObject);
                if (!graspSuccess)
                {
                    Debug.LogWarning($"{_logPrefix} [{robotId}] ⚠️ Grasp verification FAILED - may not be holding object");
                    // Continue with retreat anyway (don't block execution)
                }
                else
                {
                    Debug.Log($"{_logPrefix} [{robotId}] ✅ Grasp verified successfully");
                }
            }
            if (
                options.graspConfig != null
                && options.graspConfig.enableRetreat
                && candidate.retreatPosition != Vector3.zero
            )
            {
                _targetTransform = null;
                _followingPath = false;
                _currentPath = null;
                yield return null;
                GameObject retreat = GetCachedTempObject(ref _cachedTempTargetRetreat, "_retreat");
                retreat.transform.SetPositionAndRotation(
                    candidate.retreatPosition,
                    candidate.retreatRotation
                );
                _isGraspingTarget = false;
                SetTargetInternal(
                    retreat.transform,
                    null,
                    new GraspOptions { closeGripperOnReach = false }
                );
                yield return StartCoroutine(
                    WaitForTargetWithTimeout(RobotConstants.DEFAULT_GRASP_TIMEOUT_SECONDS)
                );
            }
            _activeGraspCoroutine = null;
            OnTargetReached?.Invoke();
        }

        // (Include GetDistanceToTarget, utilities, OnDrawGizmos, etc. from previous)
        public float GetDistanceToTarget() => _distanceToTarget;

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
                robotJoints[i].jointVelocity = new ArticulationReducedSpace(0f);
                robotJoints[i].jointForce = new ArticulationReducedSpace(0f);
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

        /// <summary>
        /// Verify grasp success using multi-criteria verification:
        /// - Contact detection (both fingers touching object)
        /// - Force measurement (sufficient grasp force)
        /// - Gripper closure position (not fully open or fully closed)
        /// </summary>
        /// <param name="targetObject">Object being grasped</param>
        /// <returns>True if grasp is verified successful</returns>
        private bool VerifyGraspSuccess(GameObject targetObject)
        {
            if (_contactSensor == null || _gripperController == null || targetObject == null)
                return false; // Can't verify without sensors

            // Multi-criteria verification
            bool hasContact = _contactSensor.HasContact(targetObject);
            float graspForce = _contactSensor.EstimateGraspForce();
            float gripperClosure = _gripperController.targetPosition;

            // Heuristic thresholds
            bool forceOk = graspForce > 5f; // Minimum force threshold (5N)
            bool closureOk = gripperClosure < 0.1f && gripperClosure > 0.01f; // Not empty, not crushed

            Debug.Log($"{_logPrefix} [{robotId}] Grasp verification: contact={hasContact}, " +
                     $"force={graspForce:F1}N, closure={gripperClosure:F3}m");

            return hasContact && forceOk && closureOk;
        }

        /// <summary>
        /// Get end effector velocity in IK frame coordinates.
        /// Computes velocity from ArticulationBody joint velocities using forward kinematics.
        /// </summary>
        private Vector3 GetEndEffectorVelocity()
        {
            if (robotJoints == null || robotJoints.Length == 0 || endEffectorBase == null)
                return Vector3.zero;

            // Get end effector velocity in world space
            Vector3 endEffectorWorldVelocity = Vector3.zero;

            // Compute velocity contribution from each joint
            // v_ee = sum(v_joint_i) where v_joint = omega x r
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

        private void DrawDebugVisualization()
        {
            if (enableDebugVisualization && _targetObject != null && endEffectorBase != null)
                Debug.DrawLine(
                    endEffectorBase.position,
                    _targetObject.transform.position,
                    Color.blue
                );
        }

        private void OnDrawGizmos()
        {
            if (
                !enableDebugVisualization
                || _currentPath == null
                || !_followingPath
                || _ikFrame == null
            )
                return;
            Gizmos.color = Color.cyan;
            for (int i = 0; i < _currentPath.waypoints.Count - 1; i++)
                Gizmos.DrawLine(
                    _ikFrame.TransformPoint(_currentPath.waypoints[i].position),
                    _ikFrame.TransformPoint(_currentPath.waypoints[i + 1].position)
                );
            if (_pathProgress < _currentPath.totalDistance)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere(
                    _ikFrame.TransformPoint(
                        _currentPath.GetWaypointAtDistance(_pathProgress).position
                    ),
                    0.02f
                );
            }
        }

        public void SetCartesianControlEnabled(bool enabled) => _useCartesianControl = enabled;

        public void SetCartesianMotionParameters(
            float maxVelocity = 0.2f,
            float acceleration = 0.5f,
            float waypointSpacing = 0.03f
        )
        {
            _maxCartesianVelocity = Mathf.Max(0.01f, maxVelocity);
            _cartesianAcceleration = Mathf.Max(0.01f, acceleration);
            _waypointSpacing = Mathf.Clamp(waypointSpacing, 0.01f, 0.1f);
        }

        public CartesianPath GetCurrentPath() => _currentPath;

        public bool IsFollowingPath() => _followingPath;

        public float GetPathProgress() => _pathProgress;

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
