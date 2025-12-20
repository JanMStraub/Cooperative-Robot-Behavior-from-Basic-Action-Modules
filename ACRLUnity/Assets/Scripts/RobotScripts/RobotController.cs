using Configuration;
using Core;
using Simulation;
using UnityEngine;
using Utilities;

namespace Robotics
{
    /// <summary>
    /// Controls a robotic arm using inverse kinematics (IK) with a damped least squares
    /// pseudo-inverse approach. Handles target assignment, IK step computation,
    /// and joint drive updates.
    /// </summary>
    public class RobotController : MonoBehaviour
    {
        private SimulationManager _simulationManager;
        private RobotManager _robotManager;

        [Header("Robot Identity")]
        public string robotId = "Robot1";

        [Header("IK Parameters")]
        [SerializeField]
        private float _dampingFactorLambda = RobotConstants.DEFAULT_DAMPING_FACTOR;

        [SerializeField]
        private float _minStepSpeedNearTarget = RobotConstants.MIN_STEP_SPEED_NEAR_TARGET;

        [SerializeField]
        private float _maxStepSpeed = RobotConstants.MAX_STEP_SPEED;

        // IK Solver (extracted for testability)
        private IKSolver _ikSolver;

        private float _distanceToTarget;
        private bool _hasReachedTarget = true;
        private Transform _targetTransform;

        // Current state (updated before IK calculation)
        private Vector3 _endEffectorWorldPosition;
        private Quaternion _endEffectorWorldRotation;
        private Vector3 _targetVector;

        private Vector3 _endEffectorLocalPosition,
            _targetLocalPosition; // positions in IK frame
        private Quaternion _endEffectorLocalRotation,
            _targetLocalRotation; // rotations in IK frame

        // Original target object (for tracking what we're actually grasping)
        private GameObject _targetObject;

        // Cached temporary GameObjects (memory leak fix)
        private GameObject _cachedGraspTarget;
        private GameObject _cachedTempTarget;

        [Header("Robot Components")]
        public ArticulationBody[] robotJoints;
        public Transform endEffectorBase;
        public Transform IKReferenceFrame;
        public float[] jointDriveTargets = { 0, 0, 0, 0, 0, 0 };

        [Header("Gripper Integration")]
        [SerializeField]
        private GripperController _gripperController;

        // Grasp behavior configuration
        public bool _closeGripperAfterReach = false;
        private GraspApproach _currentGraspApproach;

        // Helper variables
        private const string _logPrefix = "[ROBOT_CONTROLLER]";

        // Events

        public event System.Action OnTargetReached;

        /// <summary>
        /// Updates the flag indicating whether the target has been reached.
        /// Triggers gripper close if configured.
        /// </summary>
        /// <param name="setting">The value to set the targetReached flag to</param>
        public void SetTargetReached(bool setting)
        {
            if (_hasReachedTarget != setting)
            {
                _hasReachedTarget = setting;

                // Notify simulation manager of target state change
                if (_simulationManager != null)
                {
                    _simulationManager.NotifyTargetReached(robotId, setting);
                }

                // Fire event and handle gripper when target is reached
                if (setting)
                {
                    OnTargetReached?.Invoke();

                    // [NEW] Close gripper if configured to do so
                    if (_closeGripperAfterReach && _gripperController != null)
                    {
                        _gripperController.CloseGrippers();
                        Debug.Log($"{_logPrefix} Closing gripper after reaching target");
                    }
                }
            }
        }

        /// <summary>
        /// Get or create a cached GameObject for temporary targets.
        /// Prevents memory leaks by reusing the same GameObjects instead of creating new ones.
        /// </summary>
        private GameObject GetOrCreateTempObject(string suffix)
        {
            GameObject temp =
                suffix == RobotConstants.GRASP_TARGET_SUFFIX
                    ? _cachedGraspTarget
                    : _cachedTempTarget;

            if (temp == null)
            {
                temp = new GameObject($"{robotId}{suffix}");
                if (suffix == RobotConstants.GRASP_TARGET_SUFFIX)
                    _cachedGraspTarget = temp;
                else
                    _cachedTempTarget = temp;
            }

            return temp;
        }

        /// <summary>
        /// Internal method to set target with pre-configured transform.
        /// Handles common logic for all SetTarget overloads.
        /// </summary>
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
        }

        /// <summary>
        /// Updates the target object with grasp options.
        /// By default, uses grasp planning to compute optimal grasp pose.
        /// </summary>
        /// <param name="target">The target GameObject to grasp</param>
        /// <param name="options">Grasp planning and gripper control options (default: grasp planning enabled)</param>
        public void SetTarget(GameObject target, GraspOptions options = default)
        {
            if (target == null)
            {
                Debug.LogWarning($"{_logPrefix} SetTarget called with null target");
                return;
            }

            // Use default options if not specified
            if (options.Equals(default(GraspOptions)))
            {
                options = GraspOptions.Default;
            }

            // [1] Store the original target object reference
            _targetObject = target;

            // [2] Apply grasp planning if enabled
            Transform targetTransform;

            if (options.useGraspPlanning && endEffectorBase != null)
            {
                // Determine optimal approach direction
                GraspApproach approach =
                    options.approach
                    ?? GraspPlanner.DetermineOptimalApproach(
                        objectPosition: target.transform.position,
                        gripperPosition: endEffectorBase.position,
                        objectSize: GraspPlanner.GetObjectSize(target)
                    );

                // Calculate grasp pose
                var (graspPos, graspRot) = GraspPlanner.CalculateGraspPose(
                    targetObject: target,
                    gripperPosition: endEffectorBase.position,
                    approachDirection: approach
                );

                _currentGraspApproach = approach;

                Debug.Log(
                    $"{_logPrefix} Using grasp planning: approach={approach}, pos={graspPos}, rot={graspRot.eulerAngles}"
                );
                Debug.Log(
                    $"{_logPrefix} Original object: {target.name}, targeting grasp pose instead of object center"
                );

                // Get cached grasp target GameObject
                GameObject graspTarget = GetOrCreateTempObject(RobotConstants.GRASP_TARGET_SUFFIX);
                graspTarget.transform.SetPositionAndRotation(graspPos, graspRot);
                targetTransform = graspTarget.transform;
            }
            else
            {
                // No grasp planning - target the object directly
                targetTransform = target.transform;
            }

            // [3] Open gripper if requested (prepare for grasp)
            if (options.openGripperOnSet && _gripperController != null)
            {
                _gripperController.OpenGrippers();
                Debug.Log($"{_logPrefix} Opening gripper for target {target.name}");
            }

            // [4] Set target using internal method
            SetTargetInternal(targetTransform, target, options);
        }

        /// <summary>
        /// Updates the target to a specific position with grasp options.
        /// Uses GraspOptions.MoveOnly by default (no grasp planning for coordinates).
        /// Attempts to find a real scene object at that position first.
        /// </summary>
        /// <param name="position">Target position in world coordinates</param>
        /// <param name="options">Grasp options (default: MoveOnly for coordinates)</param>
        public void SetTarget(Vector3 position, GraspOptions options = default)
        {
            // Use MoveOnly if not specified (coordinates don't have geometry for grasp planning)
            if (options.Equals(default(GraspOptions)))
            {
                options = GraspOptions.MoveOnly;
            }

            // Try to find real object at this position using ObjectFinder
            if (ObjectFinder.Instance != null)
            {
                GameObject realObject = ObjectFinder.Instance.FindClosestObject(
                    position,
                    radius: RobotConstants.OBJECT_FINDING_RADIUS
                );
                if (realObject != null)
                {
                    float distance = Vector3.Distance(position, realObject.transform.position);
                    if (distance < RobotConstants.OBJECT_DISTANCE_THRESHOLD)
                    {
                        Debug.Log(
                            $"{_logPrefix} Found real object '{realObject.name}' at target position"
                        );
                        SetTarget(realObject, options); // Use GameObject method with grasp planning
                        return;
                    }
                }
            }

            // Fallback: Get cached temporary target for pure coordinate movement
            GameObject tempTarget = GetOrCreateTempObject(RobotConstants.TEMP_TARGET_SUFFIX);
            tempTarget.transform.position = position;

            // For coordinate targets, skip grasp planning (no object to analyze)
            var coordOptions = options;
            coordOptions.useGraspPlanning = false;

            // Handle gripper if needed
            if (coordOptions.openGripperOnSet && _gripperController != null)
                _gripperController.OpenGrippers();

            SetTargetInternal(
                tempTarget.transform,
                null, // No real object for coordinate targets
                coordOptions
            );
        }

        /// <summary>
        /// Updates the target to a specific position and rotation with grasp options.
        /// Uses GraspOptions.MoveOnly by default (explicit pose provided, skip grasp planning).
        /// </summary>
        /// <param name="position">Target position in world coordinates</param>
        /// <param name="rotation">Target rotation in world coordinates</param>
        /// <param name="options">Grasp options (default: MoveOnly for explicit poses)</param>
        public void SetTarget(Vector3 position, Quaternion rotation, GraspOptions options = default)
        {
            // Use MoveOnly if not specified (explicit pose provided, skip grasp planning)
            if (options.Equals(default(GraspOptions)))
            {
                options = GraspOptions.MoveOnly;
            }

            GameObject tempTarget = GetOrCreateTempObject(RobotConstants.TEMP_TARGET_SUFFIX);
            tempTarget.transform.position = position;
            tempTarget.transform.rotation = rotation;

            // Skip grasp planning for explicit pose targets
            var poseOptions = options;
            poseOptions.useGraspPlanning = false;

            // Handle gripper if needed
            if (poseOptions.openGripperOnSet && _gripperController != null)
                _gripperController.OpenGrippers();

            SetTargetInternal(
                tempTarget.transform,
                null, // No real object for pose targets
                poseOptions
            );
        }

        /// <summary>
        /// Returns the current distance between end effector and target.
        /// Returns 0 if no target is set.
        /// </summary>
        public float GetDistanceToTarget()
        {
            if (_targetTransform == null)
                return 0f;

            UpdateEndEffectorState();
            return _distanceToTarget;
        }

        /// <summary>
        /// Returns the current target position, or null if no target is set
        /// </summary>
        public Vector3? GetCurrentTarget()
        {
            if (_targetTransform == null)
                return null;
            return _targetTransform.position;
        }

        /// <summary>
        /// Gets the current target rotation, or null if no target is set
        /// </summary>
        public Quaternion? GetCurrentTargetRotation()
        {
            if (_targetTransform == null)
                return null;
            return _targetTransform.rotation;
        }

        /// <summary>
        /// Check if robot has an active target
        /// </summary>
        public bool HasTarget => _targetTransform != null;

        /// <summary>
        /// Gets the actual target object being grasped (not the temporary grasp pose target).
        /// Returns null if no target is set or if targeting a pure coordinate.
        /// </summary>
        public GameObject GetTargetObject() => _targetObject;

        /// <summary>
        /// Phase 4: Get current end effector position
        /// </summary>
        public Vector3 GetCurrentEndEffectorPosition()
        {
            if (endEffectorBase == null)
                return Vector3.zero;
            return endEffectorBase.position;
        }

        /// <summary>
        /// Initializes joint drive properties (stiffness, damping, limits, etc.)
        /// and allocates matrices for IK calculations.
        /// </summary>
        private void InitializeRobot()
        {
            // Ensure robot joints are assigned
            if (robotJoints == null || robotJoints.Length == 0)
            {
                Debug.LogError(
                    $"{_logPrefix} Robot joints are not assigned. Please assign ArticulationBodies."
                );
                return;
            }

            int jointCount = robotJoints.Length;

            // Configure joint drives using RobotManager profiles
            for (int i = 0; i < jointCount; i++)
            {
                var drive = robotJoints[i].xDrive;

                // Try to get robot instance and use its profile
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
            }

            // Initialize IK solver
            _ikSolver = new IKSolver(jointCount, _dampingFactorLambda);
        }

        /// <summary>
        /// Updates cached world and local positions/rotations of the end effector and target.
        /// </summary>
        private void UpdateEndEffectorState()
        {
            // Safety check - should not be called without a target, but guard anyway
            if (_targetTransform == null)
            {
                _distanceToTarget = 0f;
                return;
            }

            // World-state
            _endEffectorWorldPosition = endEffectorBase.position;
            _endEffectorWorldRotation = endEffectorBase.rotation;

            // Transfrom both EE and Target into the IK frame
            var frame = IKReferenceFrame != null ? IKReferenceFrame : endEffectorBase.root; // Fallback if not set

            _endEffectorLocalPosition = frame.InverseTransformPoint(_endEffectorWorldPosition);
            _targetLocalPosition = frame.InverseTransformPoint(_targetTransform.position);

            _endEffectorLocalRotation =
                Quaternion.Inverse(frame.rotation) * _endEffectorWorldRotation;
            _targetLocalRotation = Quaternion.Inverse(frame.rotation) * _targetTransform.rotation;

            _targetVector = _targetLocalPosition - _endEffectorLocalPosition;
            _distanceToTarget = _targetVector.magnitude;
        }

        /// <summary>
        /// Resets all drive targets and joint states to zero.
        /// </summary>
        public void ResetJointTargets()
        {
            foreach (var joint in robotJoints)
            {
                var drive = joint.xDrive;
                drive.target = 0;
                joint.xDrive = drive;

                joint.jointPosition = new ArticulationReducedSpace(0f);
                joint.jointForce = new ArticulationReducedSpace(0f);
                joint.jointVelocity = new ArticulationReducedSpace(0f);
            }
        }

        /// <summary>
        /// Build array of joint information for IK solver (positions and axes in IK frame)
        /// </summary>
        private JointInfo[] BuildJointInfoArray()
        {
            var frame = IKReferenceFrame != null ? IKReferenceFrame : endEffectorBase.root;
            JointInfo[] joints = new JointInfo[robotJoints.Length];

            for (int i = 0; i < robotJoints.Length; i++)
            {
                var joint = robotJoints[i];
                var jointTransform = joint.transform;

                // World joint position/axis
                Vector3 jointWorldPosition = jointTransform.position;
                Vector3 axisWorld = jointTransform.rotation * joint.anchorRotation * Vector3.right;

                // Transform to IK frame
                Vector3 jointLocalPosition = frame.InverseTransformPoint(jointWorldPosition);
                Vector3 axisLocal = frame.InverseTransformDirection(axisWorld).normalized;

                joints[i] = new JointInfo(jointLocalPosition, axisLocal);
            }

            return joints;
        }

        /// <summary>
        /// Performes one inverse kinematics step to compute and apply new joint
        /// angles that move the robot towards the target.
        /// </summary>
        public void PerformInverseKinematicsStep()
        {
            if (robotJoints == null || robotJoints.Length == 0)
            {
                Debug.LogWarning($"{_logPrefix}  No robot joints found or IK not initialized.");
                return;
            }
            if (endEffectorBase == null || _targetTransform == null)
            {
                Debug.LogError($"{_logPrefix}  EndEffector or Target is not assigned.");
                return;
            }

            UpdateEndEffectorState(); // Get latest positions/rotations

            int jointCount = robotJoints.Length;

            // Get robot's specific profile or fall back to default
            RobotConfig robotProfile = null;
            float globalSpeedMultiplier = 1.0f; // Default fallback

            if (_robotManager != null)
            {
                robotProfile = _robotManager.GetRobotProfile(robotId);
                if (robotProfile == null)
                    robotProfile = _robotManager.RobotProfile;
                globalSpeedMultiplier = _robotManager.globalSpeedMultiplier;
            }

            // If still no profile, create minimal defaults for testing
            if (robotProfile == null)
            {
                // Create a temporary profile with default values from RobotConstants
                robotProfile = ScriptableObject.CreateInstance<RobotConfig>();
                robotProfile.convergenceThreshold = RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD;
                robotProfile.maxJointStepRad = RobotConstants.DEFAULT_MAX_JOINT_STEP_RAD;
                robotProfile.adjustmentSpeed = 1.0f;
            }

            // Early convergence check - skip expensive IK computation if already at target
            if (_distanceToTarget < robotProfile.convergenceThreshold)
            {
                if (!_hasReachedTarget) // Only log/notify once
                {
                    SetTargetReached(true);
                    Debug.Log($"{_logPrefix}  {robotId} IK converged to target (early exit)");
                    OnTargetReached?.Invoke();
                }
                return; // Skip expensive IK computation
            }

            // Build joint info array for IK solver
            JointInfo[] joints = BuildJointInfoArray();

            // Create IK states for current and target
            IKState currentState = new IKState(
                _endEffectorLocalPosition,
                _endEffectorLocalRotation
            );
            IKState targetState = new IKState(_targetLocalPosition, _targetLocalRotation);

            // Get approach-specific orientation weight
            float orientationWeight = GetOrientationWeight(_currentGraspApproach);

            // Compute joint deltas using IK solver with approach-aware weighting
            var jointDeltas = _ikSolver.ComputeJointDeltas(
                currentState,
                targetState,
                joints,
                robotProfile.convergenceThreshold,
                orientationWeight
            );

            // Check if converged
            if (jointDeltas == null)
            {
                SetTargetReached(true); // This will notify SimulationManager
                Debug.Log($"{_logPrefix}  {robotId} IK converged to target");
                OnTargetReached?.Invoke();

                return; // Already converged
            }

            // Clamp joint increments
            for (int i = 0; i < jointCount; ++i)
            {
                jointDeltas[i] = System.Math.Clamp(
                    jointDeltas[i],
                    -robotProfile.maxJointStepRad,
                    robotProfile.maxJointStepRad
                );
            }

            // Adaptive step scaling
            float normalizedDistance = Mathf.Clamp01(
                _targetVector.magnitude / endEffectorBase.lossyScale.x
            );
            float adaptiveGain = Mathf.Lerp(
                _minStepSpeedNearTarget,
                _maxStepSpeed,
                normalizedDistance
            ); // Slower as it nears target

            // Apply approach-specific motion scaling for better grasp execution
            float approachMultiplier = GetApproachSpeedMultiplier(
                _currentGraspApproach,
                normalizedDistance
            );

            float stepScale =
                robotProfile.adjustmentSpeed
                * globalSpeedMultiplier
                * adaptiveGain
                * approachMultiplier;

            // Batch joint updates for better performance
            ArticulationDrive[] driveUpdates = new ArticulationDrive[jointCount];
            bool hasUpdates = false;

            // Collect all drive updates first
            for (int i = 0; i < jointCount; i++)
            {
                ArticulationDrive drive = robotJoints[i].xDrive;

                // The change in angle from IK (deltaTheta) is in radians
                // Convert to degrees and apply step scale
                float deltaAngleDegree = (float)jointDeltas[i] * Mathf.Rad2Deg * stepScale;

                // Apply the scaled change and clamp to joint limits
                float newTarget = Mathf.Clamp(
                    drive.target + deltaAngleDegree,
                    drive.lowerLimit,
                    drive.upperLimit
                );

                if (!Mathf.Approximately(newTarget, drive.target))
                {
                    drive.target = newTarget;
                    driveUpdates[i] = drive;
                    hasUpdates = true;
                }
                else
                {
                    driveUpdates[i] = drive; // Keep existing drive
                }
            }

            // Apply all updates in one pass (reduces Unity overhead)
            if (hasUpdates)
            {
                for (int i = 0; i < jointCount; i++)
                {
                    if (!Mathf.Approximately(driveUpdates[i].target, robotJoints[i].xDrive.target))
                    {
                        robotJoints[i].xDrive = driveUpdates[i];
                    }
                }
            }
        }

        private void Start()
        {
            _simulationManager = SimulationManager.Instance;
            _robotManager = RobotManager.Instance;

            // Set robot ID if empty
            if (string.IsNullOrEmpty(robotId))
            {
                robotId = gameObject.name;
            }

            // [NEW] Find gripper controller if not assigned
            if (_gripperController == null)
            {
                _gripperController = GetComponentInChildren<GripperController>();
                if (_gripperController == null)
                {
                    Debug.LogWarning(
                        $"{_logPrefix} No GripperController found in children of {robotId}"
                    );
                }
                else
                {
                    Debug.Log($"{_logPrefix} Auto-discovered GripperController for {robotId}");
                }
            }

            // Register with RobotManager if available (skip if already registered)
            if (_robotManager != null && !_robotManager.IsRobotRegistered(robotId))
            {
                _robotManager.RegisterRobot(robotId, gameObject);
            }

            InitializeRobot();
        }

        private void FixedUpdate()
        {
            // Early exit if the robot should stop
            if (_simulationManager != null && _simulationManager.ShouldStopRobots)
                return;

            // Check if this robot is allowed to move (respects coordination mode)
            if (_simulationManager != null && !_simulationManager.IsRobotActive(robotId))
                return;

            if (!_hasReachedTarget)
                PerformInverseKinematicsStep();
        }

        /// <summary>
        /// Get approach-specific orientation weight for IK solver.
        /// Different grasp approaches require different orientation precision.
        /// </summary>
        /// <param name="approach">The grasp approach being used</param>
        /// <returns>Orientation weight (0.5 to 2.0, default 1.0)</returns>
        private float GetOrientationWeight(GraspApproach approach)
        {
            switch (approach)
            {
                case GraspApproach.Top:
                    // Top approach: High orientation precision needed (gripper must point down)
                    // Weight > 1.0 emphasizes orientation error in IK solution
                    return 1.5f;

                case GraspApproach.Front:
                    // Front approach: Medium orientation precision (direct approach)
                    return 1.0f;

                case GraspApproach.Side:
                    // Side approach: Lower orientation precision (more flexible)
                    // Weight < 1.0 de-emphasizes orientation, prioritizes position
                    return 0.7f;

                default:
                    // Default: Balanced weight
                    return 1.0f;
            }
        }

        /// <summary>
        /// Get approach-specific speed multiplier for motion planning.
        /// Different grasp approaches require different motion characteristics for optimal execution.
        /// </summary>
        /// <param name="approach">The grasp approach being used</param>
        /// <param name="normalizedDistance">Distance to target (0=at target, 1=far away)</param>
        /// <returns>Speed multiplier (0.5 to 1.0)</returns>
        private float GetApproachSpeedMultiplier(GraspApproach approach, float normalizedDistance)
        {
            // When far from target (normalizedDistance > 0.5), all approaches move at full speed
            if (normalizedDistance > 0.5f)
                return 1.0f;

            // When close to target, apply approach-specific scaling
            switch (approach)
            {
                case GraspApproach.Top:
                    // Top approach can move faster - gravity assists and more stable
                    return Mathf.Lerp(0.9f, 1.0f, normalizedDistance * 2f);

                case GraspApproach.Side:
                    // Side approach needs more precision - slow down earlier
                    return Mathf.Lerp(0.6f, 1.0f, normalizedDistance * 2f);

                case GraspApproach.Front:
                    // Front approach is medium precision
                    return Mathf.Lerp(0.75f, 1.0f, normalizedDistance * 2f);

                default:
                    // Default to conservative (slower) approach
                    return Mathf.Lerp(0.7f, 1.0f, normalizedDistance * 2f);
            }
        }

        /// <summary>
        /// Cleanup cached GameObjects to prevent memory leaks
        /// </summary>
        private void OnDestroy()
        {
            if (_cachedGraspTarget != null)
                Destroy(_cachedGraspTarget);
            if (_cachedTempTarget != null)
                Destroy(_cachedTempTarget);
        }
    }
}
