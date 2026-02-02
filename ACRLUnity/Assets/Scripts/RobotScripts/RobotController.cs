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
    /// MERGED CONTROLLER:
    /// Combines the robust state machine, handoffs, and dynamic waits of Script A
    /// with the smooth trajectory generation, velocity profiles, and precise kinematic math of Script B.
    /// </summary>
    public class RobotController : MonoBehaviour
    {
        private SimulationManager _simulationManager;
        private RobotManager _robotManager;

        [Header("Robot Identity")]
        public string robotId = "Robot1";

        private Transform _ikFrame;
        private JointInfo[] _cachedJointInfos;
        private ArticulationDrive[] _cachedDriveUpdates;
        private bool _cachedIsRobotActive = true;
        private float _lastActiveCheckTime = 0f;
        private const float ACTIVE_CHECK_CACHE_INTERVAL = 0.1f;

        [Header("IK Parameters")]
        [SerializeField]
        private IKConfig _ikConfig;
        private IKSolver _ikSolver;

        private TrajectoryController _trajectoryController;
        private float _sqrDistanceToTarget;
        private float _distanceToTarget;
        private bool _hasReachedTarget = true;
        private bool _isGraspingTarget = false;
        private Transform _targetTransform;

        private bool _isManuallyDriven = false;
        public bool IsManuallyDriven
        {
            get => _isManuallyDriven;
            set => _isManuallyDriven = value;
        }

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

        [Header("Robot Components")]
        public ArticulationBody[] robotJoints;
        public Transform endEffectorBase;
        public Transform IKReferenceFrame;
        public float[] jointDriveTargets;

        [Header("Gripper Integration")]
        [SerializeField]
        private GripperController _gripperController;
        public bool _closeGripperAfterReach = false;

        [SerializeField]
        [Tooltip("Automatically attach the target object to the gripper when grasped")]
        private bool _attachObjectOnGrasp = true;

        [SerializeField]
        [Tooltip("Delay in seconds before closing gripper after reaching target")]
        private float _gripperCloseDelay = 0.2f;

        [Header("Fallback Motion Control")]
        [SerializeField]
        [Tooltip("SimpleRobotController for fallback execution when grasp planning fails")]
        private SimpleRobotController _simpleRobotController;

        [Header("Target Tracking")]
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

        [Header("Advanced Grasp Planning")]
        [SerializeField]
        private GraspConfig _graspConfig;
        private GraspPlanningPipeline _graspPipeline;

        [Header("Trajectory Control")]
        [SerializeField]
        private TrajectoryConfig _trajectoryConfig;

        [Header("Debug Visualization")]
        [SerializeField]
        private bool _enableDebugVisualization = true;

        private const string _logPrefix = "[ROBOT_CONTROLLER]";

        public event System.Action OnTargetReached;
        public event System.Action<bool> OnCoordinationStateChanged;

        private void Start()
        {
            _simulationManager = SimulationManager.Instance;
            _robotManager = RobotManager.Instance;

            if (string.IsNullOrEmpty(robotId))
                robotId = gameObject.name;

            if (_ikConfig == null)
                _ikConfig = Resources.Load<IKConfig>("Configuration/DefaultIKConfig");
            if (_ikConfig == null)
                _ikConfig = ScriptableObject.CreateInstance<IKConfig>();

            _ikSolver = new IKSolver(robotJoints.Length, _ikConfig.dampingFactor);

            _ikFrame =
                IKReferenceFrame != null
                    ? IKReferenceFrame
                    : (
                        endEffectorBase != null && endEffectorBase.root != null
                            ? endEffectorBase.root
                            : transform
                    );

            if (_gripperController == null)
            {
                _gripperController = GetComponentInChildren<GripperController>();
                if (_gripperController == null)
                {
                    Debug.LogWarning($"[ROBOT_CONTROLLER] No GripperController found in children of {robotId}");
                }
            }

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
            {
                Debug.LogWarning($"[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");
                return;
            }

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

            _ikSolver = new IKSolver(jointCount, _ikConfig.dampingFactor);

            if (_trajectoryConfig == null)
                _trajectoryConfig = ScriptableObject.CreateInstance<TrajectoryConfig>();

            _trajectoryController = new TrajectoryController(
                positionGains: _trajectoryConfig.positionGains,
                velocityGains: _trajectoryConfig.velocityGains
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
                    _ikConfig
                );
            }
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
                    if (_closeGripperAfterReach && _gripperController != null)
                    {
                        StartCoroutine(CloseGripperAfterDelay());
                    }
                    else
                    {
                        OnTargetReached?.Invoke();
                    }
                }
            }
        }

        private IEnumerator CloseGripperAfterDelay()
        {
            float delayStartTime = Time.time;

            yield return new WaitUntil(
                () =>
                    Time.time - delayStartTime >= _gripperCloseDelay
                    && GetEndEffectorVelocity().magnitude < 0.005f
            );

            if (_attachObjectOnGrasp && _targetObject != null)
            {
                _gripperController.SetTargetObject(_targetObject);
            }

            _gripperController.CloseGrippers();
            yield return new WaitWhile(() => _gripperController.IsMoving);

            float graspStartTime = Time.time;
            yield return new WaitUntil(
                () => Time.time - graspStartTime > 0.2f && !_gripperController.IsMoving
            );

            OnTargetReached?.Invoke();
        }

        private void FixedUpdate()
        {
            if (_simulationManager != null && _simulationManager.ShouldStopRobots)
                return;
            if (_simulationManager != null && !GetCachedIsRobotActive())
                return;

            if (_isManuallyDriven)
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
                Vector3 axisWorld = joint.transform.rotation * joint.anchorRotation * Vector3.right;
                Vector3 axisLocal = ikFrameInverseRot * axisWorld;
                _cachedJointInfos[i] = new JointInfo(jointLocalPos, axisLocal.normalized);
            }
        }

        private Vector3 GetEndEffectorVelocity()
        {
            if (robotJoints == null || robotJoints.Length == 0 || endEffectorBase == null)
                return Vector3.zero;

            Vector3 endEffectorWorldVelocity = Vector3.zero;

            for (int i = 0; i < robotJoints.Length; i++)
            {
                var joint = robotJoints[i];
                float jointAngularVel = joint.jointVelocity[0];
                Vector3 axisWorld = joint.transform.rotation * joint.anchorRotation * Vector3.right;
                Vector3 jointToEE = endEffectorBase.position - joint.transform.position;
                Vector3 velContribution = Vector3.Cross(axisWorld * jointAngularVel, jointToEE);
                endEffectorWorldVelocity += velContribution;
            }

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

            _endEffectorLocalPosition = _ikFrame.InverseTransformPoint(endEffectorBase.position);
            _targetLocalPosition = _ikFrame.InverseTransformPoint(_targetTransform.position);

            Quaternion invFrameRot = Quaternion.Inverse(_ikFrame.rotation);
            _endEffectorLocalRotation = invFrameRot * endEffectorBase.rotation;
            _targetLocalRotation = invFrameRot * _targetTransform.rotation;

            Vector3 distVec = _targetLocalPosition - _endEffectorLocalPosition;
            _sqrDistanceToTarget = distVec.sqrMagnitude;
            _distanceToTarget = Mathf.Sqrt(_sqrDistanceToTarget);
        }

        public void PerformInverseKinematicsStep()
        {
            if (robotJoints.Length == 0 || endEffectorBase == null || _targetTransform == null)
                return;

            UpdateEndEffectorState();

            float posThreshold = _isGraspingTarget
                ? RobotConstants.MOVEMENT_THRESHOLD
                : (_ikConfig != null ? _ikConfig.convergenceThreshold : 0.02f);
            float rotThreshold = 7.0f;

            float angleError = Quaternion.Angle(_endEffectorLocalRotation, _targetLocalRotation);
            bool isPosReached = _distanceToTarget < posThreshold;
            bool isRotReached = angleError < rotThreshold;

            Vector3 currentVelocity = GetEndEffectorVelocity();
            bool isSettled = currentVelocity.sqrMagnitude < 0.005f;

            bool isStalled = isSettled && (!isPosReached || !isRotReached);
            if (_enableDebugVisualization && isStalled && Time.frameCount % 60 == 0)
            {
                Debug.Log(
                    $"{_logPrefix} [{robotId}] STALLED near target. Boosting Gains. Dist: {_distanceToTarget:F4}, Ang: {angleError:F1}"
                );
            }

            if (isPosReached && isRotReached && isSettled)
            {
                if (!_hasReachedTarget)
                {
                    Debug.Log(
                        $"{_logPrefix} [{robotId}] TARGET REACHED: dist={_distanceToTarget:F4}m, vel={currentVelocity.magnitude:F3}m/s"
                    );
                    SetTargetReached(true);
                }
                return;
            }

            if (
                _hasReachedTarget
                && (_distanceToTarget > posThreshold * 1.5f || angleError > rotThreshold * 1.5f)
            )
            {
                SetTargetReached(false);
            }

            UpdateJointInfoCache();

            // STALL COMPENSATION: Increase gain when stopped but not at target
            // This overcomes static friction
            float kpMult = isStalled ? 2.5f : 1.0f;
            float orientationWeight = 1.0f;

            Vector3 constrainedTargetPos = _targetLocalPosition;
            Quaternion constrainedTargetRot = _targetLocalRotation;

            IKState currentState = new(_endEffectorLocalPosition, _endEffectorLocalRotation);
            IKState targetState = new(constrainedTargetPos, constrainedTargetRot);

            var jointDeltas = _ikSolver.ComputeJointDeltasWithVelocity(
                currentState,
                targetState,
                currentVelocity,
                Vector3.zero,
                _cachedJointInfos,
                posThreshold,
                Kp: 3.5f * kpMult,
                Kd: 0.5f,
                orientationWeight: orientationWeight,
                orientationConvergenceThreshold: rotThreshold * Mathf.Deg2Rad,
                overrideDamping: _ikConfig != null ? _ikConfig.dampingFactor : 0.2f
            );

            if (jointDeltas == null)
                return;

            RobotConfig robotProfile =
                _robotManager?.GetRobotProfile(robotId) ?? _robotManager?.RobotProfile;
            float adjSpeed = robotProfile != null ? robotProfile.adjustmentSpeed : 1.0f;
            float globalSpeed = _robotManager != null ? _robotManager.globalSpeedMultiplier : 1.0f;
            float baseScale = Mathf.Rad2Deg * adjSpeed * globalSpeed;

            float maxDegreesPerFrame = isStalled ? 8.0f : 5.0f;

            bool hasUpdates = false;

            for (int i = 0; i < robotJoints.Length; i++)
            {
                float idealStepDegrees = (float)jointDeltas[i] * baseScale;

                if (isStalled && Mathf.Abs(idealStepDegrees) > 0.001f)
                {
                    idealStepDegrees += Mathf.Sign(idealStepDegrees) * 0.05f;
                }

                float clampedStep = Mathf.Clamp(
                    idealStepDegrees,
                    -maxDegreesPerFrame,
                    maxDegreesPerFrame
                );

                ArticulationDrive drive = robotJoints[i].xDrive;
                float newTarget = Mathf.Clamp(
                    drive.target + clampedStep,
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

            _isTrackingMovingTarget = originalObject != null;
            if (_isTrackingMovingTarget)
            {
                _lastTrackedTargetPosition = originalObject.transform.position;
            }

            _ikSolver?.ResetIterationCount();
        }

        public void SetTarget(GameObject target, GraspOptions options = default)
        {
            if (target == null)
                return;
            StopActiveGraspCoroutine();

            if (options.Equals(default(GraspOptions)))
                options = GraspOptions.Default;

            GripperController holdingGripper = FindGripperHoldingObject(target);
            if (holdingGripper != null && holdingGripper != _gripperController)
            {
                Debug.Log(
                    $"{_logPrefix} {robotId} detected handoff scenario - object '{target.name}' held by another gripper"
                );
                _activeGraspCoroutine = StartCoroutine(
                    ExecuteHandoffGrasp(target, options)
                );
                return;
            }

            if (_graspPipeline != null)
            {
                GraspCandidate candidate;
                if (options.useAdvancedPlanning)
                {
                    if (options.graspConfig == null)
                        options.graspConfig = _graspConfig;
                    candidate = _graspPipeline.PlanGrasp(target, endEffectorBase.position, options);
                    if (candidate != null)
                    {
                        if (candidate.useSimplifiedExecution)
                        {
                            _activeGraspCoroutine = StartCoroutine(
                                ExecuteSimplifiedGrasp(candidate, target, options)
                            );
                        }
                        else
                        {
                            _activeGraspCoroutine = StartCoroutine(
                                ExecuteThreeWaypointGrasp(candidate, target, options)
                            );
                        }
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
                    if (candidate != null)
                    {
                        if (candidate.useSimplifiedExecution)
                        {
                            _activeGraspCoroutine = StartCoroutine(
                                ExecuteSimplifiedGrasp(candidate, target, options)
                            );
                        }
                        else
                        {
                            _activeGraspCoroutine = StartCoroutine(
                                ExecuteTwoWaypointGrasp(candidate, target, options)
                            );
                        }
                        return;
                    }
                }
            }

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

            if (ObjectFinder.Instance != null)
            {
                float findingRadius = _ikConfig != null ? _ikConfig.objectFindingRadius : 0.15f;
                GameObject realObject = ObjectFinder.Instance.FindClosestObject(
                    position,
                    findingRadius
                );
                if (realObject != null)
                {
                    float distThreshold = _ikConfig != null ? _ikConfig.objectDistanceThreshold : 0.1f;
                    if (
                        Vector3.SqrMagnitude(position - realObject.transform.position)
                        < distThreshold * distThreshold
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

        private IEnumerator WaitForTargetWithTimeout(float timeoutSeconds)
        {
            float startTime = Time.time;
            while (!_hasReachedTarget)
            {
                if (Time.time - startTime > timeoutSeconds)
                {
                    Debug.LogWarning($"{_logPrefix} {robotId} timeout after {timeoutSeconds}s");
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
            float graspTimeout = _ikConfig != null ? _ikConfig.graspTimeoutSeconds : 30f;

            _gripperController?.SetGripperPosition(candidate.preGraspGripperWidth);

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
                WaitForTargetWithTimeout(graspTimeout)
            );

            if (!_hasReachedTarget)
                yield break;

            // Robust Wait (Dynamic)
            yield return new WaitUntil(() => GetEndEffectorVelocity().magnitude < 0.01f);

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
                WaitForTargetWithTimeout(graspTimeout)
            );

            if (!_hasReachedTarget)
                yield break;

            if (options.closeGripperOnReach && _gripperController != null)
            {
                yield return new WaitUntil(() => GetEndEffectorVelocity().magnitude < 0.005f);
                _gripperController.SetTargetObject(targetObject);
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
            float graspTimeout = _ikConfig != null ? _ikConfig.graspTimeoutSeconds : 30f;

            _gripperController?.SetGripperPosition(candidate.preGraspGripperWidth);

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
                WaitForTargetWithTimeout(graspTimeout)
            );

            if (!_hasReachedTarget)
                yield break;
            yield return new WaitUntil(() => GetEndEffectorVelocity().magnitude < 0.01f);

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
                WaitForTargetWithTimeout(graspTimeout)
            );

            if (!_hasReachedTarget)
                yield break;

            if (options.closeGripperOnReach && _gripperController != null)
            {
                yield return new WaitUntil(() => GetEndEffectorVelocity().magnitude < 0.005f);
                _gripperController.SetTargetObject(targetObject);
                _gripperController.SetGripperPosition(candidate.graspGripperWidth);
                yield return new WaitWhile(() => _gripperController.IsMoving);

                float graspStartTime = Time.time;
                yield return new WaitUntil(
                    () => Time.time - graspStartTime > 0.3f && !_gripperController.IsMoving
                );
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
                    WaitForTargetWithTimeout(graspTimeout)
                );
            }

            _activeGraspCoroutine = null;
            OnTargetReached?.Invoke();
        }

        private IEnumerator ExecuteHandoffGrasp(
            GameObject targetObject,
            GraspOptions options
        )
        {
            Debug.Log($"{_logPrefix} {robotId} executing handoff for '{targetObject.name}'");

            if (options.openGripperOnSet && _gripperController != null)
            {
                _gripperController.OpenGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);
            }

            Vector3 objectPosition = targetObject.transform.position;
            GameObject handoffTarget = GetCachedTempObject(ref _cachedTempTargetGrasp, "_handoff");
            handoffTarget.transform.position = objectPosition;
            handoffTarget.transform.rotation = targetObject.transform.rotation;

            _isGraspingTarget = true;
            // This will generate a trajectory to the handoff point!
            SetTargetInternal(
                handoffTarget.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );

            float graspTimeout = _ikConfig != null ? _ikConfig.graspTimeoutSeconds : 30f;
            yield return StartCoroutine(
                WaitForTargetWithTimeout(graspTimeout)
            );
            if (!_hasReachedTarget)
            {
                Debug.LogWarning($"{_logPrefix} {robotId} failed to reach handoff position");
                _activeGraspCoroutine = null;
                yield break;
            }

            yield return new WaitUntil(() => GetEndEffectorVelocity().magnitude < 0.005f);

            if (options.closeGripperOnReach && _gripperController != null)
            {
                _gripperController.SetTargetObject(targetObject);
                _gripperController.CloseGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);

                float graspStartTime = Time.time;
                yield return new WaitUntil(
                    () => Time.time - graspStartTime > 0.3f && !_gripperController.IsMoving
                );
            }

            _activeGraspCoroutine = null;
            OnTargetReached?.Invoke();
        }

        /// <summary>
        /// Execute grasp using SimpleRobotController's IK algorithm as fallback.
        /// RobotController drives the SimpleRobotController manually (not autonomous).
        /// Used when advanced grasp planning fails and a simpler approach is needed.
        /// </summary>
        /// <param name="candidate">Grasp candidate from fallback planner</param>
        /// <param name="targetObject">Object to grasp</param>
        /// <param name="options">Grasp options</param>
        private IEnumerator ExecuteSimplifiedGrasp(
            GraspCandidate candidate,
            GameObject targetObject,
            GraspOptions options
        )
        {
            Debug.Log(
                $"{_logPrefix} {robotId} executing SIMPLIFIED grasp using SimpleRobotController backup IK (fallback mode)"
            );

            // Ensure SimpleRobotController is available
            if (_simpleRobotController == null)
            {
                Debug.LogWarning(
                    $"{_logPrefix} {robotId} SimpleRobotController not assigned! Falling back to standard execution."
                );
                // Fall back to two-waypoint grasp if SimpleRobotController is missing
                yield return StartCoroutine(
                    ExecuteTwoWaypointGrasp(candidate, targetObject, options)
                );
                yield break;
            }

            // Open gripper
            if (options.openGripperOnSet && _gripperController != null)
            {
                _gripperController.OpenGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);
            }

            // Set target in SimpleRobotController (SimpleRobotController won't run its own FixedUpdate)
            // We'll manually call its IK step method each frame
            _simpleRobotController.SetTarget(candidate.graspPosition, candidate.graspRotation);

            // Manually drive SimpleRobotController's IK until target is reached
            // This gives us control over when IK steps happen (in this coroutine)
            float timeout = _ikConfig != null ? _ikConfig.graspTimeoutSeconds : 30f;
            float startTime = Time.time;

            while (!_simpleRobotController.HasReachedTarget)
            {
                // Check timeout
                if (Time.time - startTime > timeout)
                {
                    Debug.LogWarning(
                        $"{_logPrefix} {robotId} simplified grasp timed out after {timeout}s"
                    );
                    _activeGraspCoroutine = null;
                    yield break;
                }

                // Manually perform one IK step using SimpleRobotController's algorithm
                _simpleRobotController.PerformInverseKinematicsStep();

                // Wait for next physics frame
                yield return new WaitForFixedUpdate();
            }

            Debug.Log(
                $"{_logPrefix} {robotId} simplified grasp reached target position. Distance: {_simpleRobotController.DistanceToTarget:F4}m"
            );

            // Wait for robot to settle
            yield return new WaitForSeconds(0.2f);

            // Close gripper
            if (options.closeGripperOnReach && _gripperController != null)
            {
                _gripperController.SetTargetObject(targetObject);
                _gripperController.CloseGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);

                float graspStartTime = Time.time;
                yield return new WaitUntil(
                    () => Time.time - graspStartTime > 0.3f && !_gripperController.IsMoving
                );

                Debug.Log(
                    $"{_logPrefix} {robotId} simplified grasp complete. Object held: {_gripperController.IsHoldingObject}"
                );
            }

            _activeGraspCoroutine = null;
            OnTargetReached?.Invoke();
        }

        private static GripperController FindGripperHoldingObject(GameObject obj)
        {
            GripperController[] allGrippers = FindObjectsByType<GripperController>(
                FindObjectsSortMode.None
            );
            foreach (var gripper in allGrippers)
            {
                if (gripper.IsHoldingObject && gripper.GraspedObject == obj)
                    return gripper;
            }
            return null;
        }

        // --- Utilities ---

        public float GetDistanceToTarget() => _targetTransform == null ? 0f : _distanceToTarget;

        public Vector3? GetCurrentTarget() => _targetTransform?.position;

        public Quaternion? GetCurrentTargetRotation() => _targetTransform?.rotation;

        public bool HasTarget => _targetTransform != null;

        public bool TargetReached => _hasReachedTarget;

        public GameObject GetTargetObject() => _targetObject;

        public void SetMovingTargetTracking(bool enable) => _enableMovingTargetTracking = enable;

        public void SetTargetMovementThreshold(float threshold)
        {
            _targetMovementThreshold = Mathf.Max(0.001f, threshold);
            _sqrTargetMovementThreshold = _targetMovementThreshold * _targetMovementThreshold;
        }

        /// <summary>
        /// Clear the current target and stop IK tracking.
        /// Call this when the robot should no longer maintain its current position
        /// (e.g., after opening gripper to release an object).
        /// </summary>
        public void ClearTarget()
        {
            _targetTransform = null;
            _targetObject = null;
            _hasReachedTarget = true;
            _isGraspingTarget = false;
            _closeGripperAfterReach = false;
            _isTrackingMovingTarget = false;

            // Reset trajectory
            _trajectoryController?.Reset();

            Debug.Log($"{_logPrefix} [{robotId}] Target cleared");
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

        private void DrawDebugVisualization()
        {
            if (_enableDebugVisualization && _targetObject != null && endEffectorBase != null)
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
