using Core;
using Logging;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Simulation;
using UnityEngine;

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
        private MainLogger _logger;

        [Header("Robot Identity")]
        public string robotId = "AR4_Robot";

        [Header("IK Parameters")]
        [SerializeField]
        private float _dampingFactorLambda = RobotConstants.DEFAULT_DAMPING_FACTOR;

        [SerializeField]
        private float _minStepSpeedNearTarget = RobotConstants.MIN_STEP_SPEED_NEAR_TARGET;

        [SerializeField]
        private float _maxStepSpeed = RobotConstants.MAX_STEP_SPEED;

        private float _distanceToTarget;
        private bool _hasReachedTarget = true;
        private Transform _targetTransform;

        private const int _JacobianRows = RobotConstants.JACOBIAN_DIMENSIONS;

        // Pre-allocated for performance to reduce GC allocs
        private Matrix<double> _jacobianMatrix;
        private Vector<double> _errorVector;
        private Vector<double> _jointDelta; // Joint angle changes

        // Current state (updated before IK calculation)
        private Vector3 _endEffectorWorldPosition;
        private Quaternion _endEffectorWorldRotation;
        private Vector3 _targetVector;

        private Vector3 _endEffectorLocalPosition,
            _targetLocalPosition; // positions in IK frame
        private Quaternion _endEffectorLocalRotation,
            _targetLocalRotation; // rotations in IK frame

        [Header("Robot Components")]
        public ArticulationBody[] robotJoints;
        public Transform endEffectorBase;
        public Transform IKReferenceFrame;
        public float[] jointDriveTargets = { 0, 0, 0, 0, 0, 0 };

        // Helper variables
        private const string _logPrefix = "[ROBOT_CONTROLLER]";

        // Events

        public event System.Action OnTargetReached;

        /// <summary>
        /// Updates the flag indicating whether the target has been reached.
        /// </summary>
        /// <param name="setting"> The value to set the targetReached flag to.
        /// </param>
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
            }
        }

        /// <summary>
        /// Updates the target object.
        /// </summary>
        /// <param name="target"> The new target for the robot arm.
        /// </param>
        public void SetTarget(GameObject target)
        {
            _targetTransform = target.transform;
            SetTargetReached(false); // This will notify SimulationManager

            // Log target assignment
            if (_logger != null)
            {
                string actionId = _logger.StartAction(
                    actionName: "set_target",
                    type: ActionType.Movement,
                    robotIds: new[] { robotId },
                    startPos: endEffectorBase.position,
                    targetPos: target.transform.position,
                    objectIds: new[] { target.name },
                    description: $"Setting target to {target.name}"
                );
                _logger.CompleteAction(actionId, success: true, qualityScore: 1f);
            }
        }

        /// <summary>
        /// Updates the target to a specific position by creating a temporary target transform.
        /// </summary>
        /// <param name="position"> The target position in world coordinates.
        /// </param>
        public void SetTarget(Vector3 position)
        {
            // Create or reuse a temporary target object
            GameObject tempTarget = GameObject.Find($"{robotId}_TempTarget");
            if (tempTarget == null)
            {
                tempTarget = new GameObject($"{robotId}_TempTarget");
            }

            tempTarget.transform.position = position;
            SetTarget(tempTarget);
        }

        /// <summary>
        /// Updates the target to a specific position and rotation by creating a temporary target transform.
        /// </summary>
        /// <param name="position"> The target position in world coordinates.
        /// </param>
        /// <param name="rotation"> The target rotation in world coordinates.
        /// </param>
        public void SetTarget(Vector3 position, Quaternion rotation)
        {
            // Create or reuse a temporary target object
            GameObject tempTarget = GameObject.Find($"{robotId}_TempTarget");
            if (tempTarget == null)
            {
                tempTarget = new GameObject($"{robotId}_TempTarget");
            }

            tempTarget.transform.position = position;
            tempTarget.transform.rotation = rotation;
            SetTarget(tempTarget);
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

        public float GetDriveTarget(int i) => robotJoints[i].xDrive.target;

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
        /// Initializes joint drive properties (stiffness, damping, limits, etc.)
        /// and allocates matrices for IK calculations.
        /// </summary>
        private void InitializeRobot()
        {
            int jointCount = robotJoints.Length;

            // Ensure robot joints are assigned
            if (robotJoints == null || jointCount == 0)
            {
                Debug.LogError(
                    $"{_logPrefix} Robot joints are not assigned. Please assign ArticulationBodies."
                );
                return;
            }

            // Configure joint drives using RobotManager profiles
            for (int i = 0; i < jointCount; i++)
            {
                var drive = robotJoints[i].xDrive;

                // Try to get robot instance and use its profile
                if (
                    _robotManager.RobotInstances.TryGetValue(robotId, out var robotInstance)
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

            _jacobianMatrix = DenseMatrix.Build.Dense(_JacobianRows, jointCount);
            // Pseudo-inverse will be calculated based on Jacobian's dimensions
            _errorVector = Vector<double>.Build.Dense(_JacobianRows);
            _jointDelta = Vector<double>.Build.Dense(jointCount);
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
        /// Computes the 6xN Jacobian matrix for the robot at its current
        /// configuration.
        /// </summary>
        /// <param name="jointCount"> Number of robot joints.</param>
        /// </param>
        private void CalculateJacobian(int jointCount)
        {
            if (_jacobianMatrix.ColumnCount != jointCount)
                _jacobianMatrix = DenseMatrix.Build.Dense(_JacobianRows, jointCount);

            var frame = IKReferenceFrame != null ? IKReferenceFrame : endEffectorBase.root;

            for (int i = 0; i < jointCount; i++)
            {
                var joint = robotJoints[i];
                var jointTransform = joint.transform;

                // World joint position/axis of xDrive
                Vector3 jointWorldPosition = jointTransform.position;
                Vector3 axisWorld = jointTransform.rotation * joint.anchorRotation * Vector3.right;

                // Bring them into IK frame
                Vector3 jointLocalPosition = frame.InverseTransformPoint(jointWorldPosition);
                Vector3 axisLocal = frame.InverseTransformDirection(axisWorld).normalized;

                // Vector from joint to EE in IK frame
                Vector3 vectorJointToEndEffector = _endEffectorLocalPosition - jointLocalPosition;

                // Jacobian column in IK frame
                Vector3 linearComponent = Vector3.Cross(axisLocal, vectorJointToEndEffector);
                Vector3 angularComponent = axisLocal;

                _jacobianMatrix[0, i] = linearComponent.x;
                _jacobianMatrix[1, i] = linearComponent.y;
                _jacobianMatrix[2, i] = linearComponent.z;
                _jacobianMatrix[3, i] = angularComponent.x;
                _jacobianMatrix[4, i] = angularComponent.y;
                _jacobianMatrix[5, i] = angularComponent.z;
            }
        }

        /// <summary>
        /// Computes the damped least squares pseudo-inverse of the Jacobian
        /// and updates joint deltas.
        /// </summary>
        private void ComputePseudoInverseJacobian()
        {
            // Jacobian: 6xN
            var jacobianTranspose = _jacobianMatrix.Transpose(); // Nx6
            var jacobianJacobianTransform = _jacobianMatrix * jacobianTranspose; // 6x6
            var identity = DenseMatrix.Build.DenseIdentity(jacobianJacobianTransform.RowCount);

            var regularized =
                jacobianJacobianTransform + _dampingFactorLambda * _dampingFactorLambda * identity;

            // Solve once; avoid explicit inverse
            var y = regularized.Solve(_errorVector); // 6x1
            _jointDelta = jacobianTranspose * y; // Nx1
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

            // Compute orientation error in local frame
            Quaternion quaternionError =
                _targetLocalRotation * Quaternion.Inverse(_endEffectorLocalRotation);
            quaternionError.ToAngleAxis(out float angleDegree, out Vector3 axisLocal);
            if (float.IsNaN(axisLocal.x) || float.IsNaN(axisLocal.y) || float.IsNaN(axisLocal.z))
            {
                axisLocal = Vector3.zero;
                angleDegree = 0f;
            }
            Vector3 orientationError = axisLocal.normalized * (angleDegree * Mathf.Deg2Rad);

            // Build error vector (6D task space: position + orientation)
            _errorVector[0] = _targetVector.x;
            _errorVector[1] = _targetVector.y;
            _errorVector[2] = _targetVector.z;
            _errorVector[3] = orientationError.x;
            _errorVector[4] = orientationError.y;
            _errorVector[5] = orientationError.z;

            // Get robot's specific profile or fall back to default
            var robotProfile = _robotManager.GetRobotProfile(robotId);
            if (robotProfile == null)
                robotProfile = _robotManager.RobotProfile;
            if (_errorVector.L2Norm() < robotProfile.convergenceThreshold)
            {
                SetTargetReached(true); // This will notify SimulationManager
                Debug.Log($"{_logPrefix}  {robotId} IK converged to target");
                OnTargetReached?.Invoke();

                // Log successful convergence
                if (_logger != null)
                {
                    string actionId = _logger.StartAction(
                        actionName: "reach_target",
                        type: ActionType.Movement,
                        robotIds: new[] { robotId },
                        startPos: endEffectorBase.position,
                        targetPos: _targetTransform.position,
                        objectIds: new[] { _targetTransform.name },
                        description: $"Reached target {_targetTransform.name}"
                    );
                    float distance = GetDistanceToTarget();
                    float quality = Mathf.Max(
                        0f,
                        1f - distance / RobotConstants.DEFAULT_CONVERGENCE_THRESHOLD
                    );
                    _logger.CompleteAction(actionId, success: true, qualityScore: quality);
                }

                return; // Already close enough
            }

            // Calculate the 6xN Jacobian matrix
            CalculateJacobian(jointCount);

            // Compute the pseudo-inverse of the Jacobian
            ComputePseudoInverseJacobian();

            // Clamp joint increments
            for (int i = 0; i < jointCount; ++i)
            {
                _jointDelta[i] = System.Math.Clamp(
                    _jointDelta[i],
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

            float stepScale =
                robotProfile.adjustmentSpeed * _robotManager.globalSpeedMultiplier * adaptiveGain;

            // Apply joint updates
            for (int i = 0; i < jointCount; i++)
            {
                var joint = robotJoints[i];
                ArticulationDrive drive = joint.xDrive;

                // The change in angle from IK (deltaTheta) is in radians
                // Convert to degrees and apply step scale
                float deltaAngleDegree = (float)_jointDelta[i] * Mathf.Rad2Deg * stepScale;

                // Apply the scaled change
                float newTarget = drive.target + deltaAngleDegree;

                // Clamp to joint limits
                newTarget = Mathf.Clamp(newTarget, drive.lowerLimit, drive.upperLimit);

                if (!Mathf.Approximately(newTarget, drive.target))
                {
                    drive.target = newTarget;
                    joint.xDrive = drive;
                }
            }
        }

        private void Start()
        {
            _simulationManager = SimulationManager.Instance;
            _robotManager = RobotManager.Instance;
            _logger = MainLogger.Instance;

            // Set robot ID if empty
            if (string.IsNullOrEmpty(robotId))
            {
                robotId = gameObject.name;
            }

            // Register with RobotManager if available
            if (_robotManager != null)
            {
                _robotManager.RegisterRobot(robotId, gameObject);
            }

            // Log initialization
            if (_logger != null)
            {
                string actionId = _logger.StartAction(
                    actionName: "initialize_robot",
                    type: ActionType.Task,
                    robotIds: new[] { robotId },
                    startPos: endEffectorBase.position,
                    objectIds: new[] { gameObject.name },
                    description: $"Initialized robot {robotId}"
                );
                _logger.CompleteAction(actionId, success: true, qualityScore: 1f);
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
    }
}
