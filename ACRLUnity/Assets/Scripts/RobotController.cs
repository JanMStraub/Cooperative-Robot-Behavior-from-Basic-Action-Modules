using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using UnityEngine;
using System.Collections;

public class RobotController : MonoBehaviour
{
    private SimulationManager _simulationManagerInstance;
    private RobotManager _robotManagerInstance;
    private float _convergenceThreshold = 1e-4f;
    private float _dampingFactorLambda = 0.1f;
    private float _distanceToTarget;
    private float _maxStepSpeed = 0.5f;
    private float _minStepSpeedNearTarget = 0.1f;
    private bool _targetReached = true;
    private const int _JacobianRows = 6; // 3 for position, 3 for orientation

    // Pre-allocated for performance to reduce GC allocs
    private Matrix<double> _jacobian;
    private Matrix<double> _pseudoInverseJacobian;
    private Vector<double> _errorVector;
    private Vector<double> _deltaTheta; // Joint angle changes

    // Current state (updated before IK calculation)
    private Vector3 _currentEndEffectorPosition;
    private Quaternion _currentEndEffectorRotation;
    private Vector3 _vectorToTarget;

    [Header("Joints and Gripper objects")]
    public ArticulationBody[] robotJoints;
    public Transform robotGripperBase;
    public Transform _target;
    public float[] ArticulationBodyTargets = { 0, 0, 0, 0, 0, 0 };

    /// <summary>
    /// Updates the flag indicating whether the target has been reached.
    /// </summary>
    /// <param name="setting"> The value to set the targetReached flag to.
    /// </param>
    public void SetTargetReached(bool setting)
    {
        _targetReached = setting;
    }

    /// <summary>
    /// Sets up the robot's joints with the necessary configuration values like
    /// stiffness, damping, force limits, etc.
    /// </summary>
    private void SetUpRobot()
    {
        int numJoints = robotJoints.Length;

        // Ensure robot joints are assigned
        if (robotJoints == null || numJoints == 0)
        {
            Debug.LogError(
                "No ArticulationBody components assigned to the robot. Check your hierarchy."
            );
            return;
        }

        // Configure joint drives
        for (int i = 0; i < numJoints; i++)
        {
            var drive = robotJoints[i].xDrive;
            drive.stiffness = _robotManagerInstance.GetStiffnessValue(i);
            drive.damping = _robotManagerInstance.GetDampingValue(i);
            drive.forceLimit = _robotManagerInstance.GetForceLimits(i);
            drive.upperLimit = _robotManagerInstance.GetDriveUpperLimits(i);
            drive.lowerLimit = _robotManagerInstance.GetDriveLowerLimits(i);
            robotJoints[i].xDrive = drive;
        }

        _jacobian = DenseMatrix.Build.Dense(_JacobianRows, numJoints);
        // Pseudo-inverse will be calculated based on Jacobian's dimensions
        _errorVector = Vector<double>.Build.Dense(_JacobianRows);
        _deltaTheta = Vector<double>.Build.Dense(numJoints);

        Debug.Log($"IK Controller Initialized with {numJoints} joints.");
    }

    /// <summary>
    /// Updates the target object.
    /// </summary>
    /// <param name="target"> The new target for the robot arm.
    /// </param>
    public void SetTarget(GameObject target)
    {
        _target = target.transform;

        _targetReached = false;
    }

    public float GetDistanceToTarget()
    {
        CalculateDistanceToTarget();

        return _distanceToTarget;
    }

    public float GetDriveTarget(int i)
    {
        return robotJoints[i].xDrive.target;
    }

    public Vector3 GetCurrentTarget() => _target.position;

    public float GetMaxStepSpeed() => _maxStepSpeed;

    /// <summary>
    /// Calculates the distance from the closest end-effector sensor
    /// to the target position.
    /// </summary>
    private void CalculateDistanceToTarget()
    {
        Vector3 endEffectorPosition = robotGripperBase.transform.position;

        _distanceToTarget = Vector3.Distance(_target.position, endEffectorPosition);
        _currentEndEffectorPosition = endEffectorPosition;
        _currentEndEffectorRotation = robotGripperBase.rotation;
        _vectorToTarget = _target.position - endEffectorPosition;
    }

    public void SetDriveTargetsToZero()
    {
        for (int i = 0; i < robotJoints.Length; i++)
        {
            ArticulationBody joint = robotJoints[i];
            ArticulationDrive drive = joint.xDrive;

            drive.target = 0;
            joint.xDrive = drive;

            joint.jointPosition = new ArticulationReducedSpace(0f);
            joint.jointForce = new ArticulationReducedSpace(0f);
            joint.jointVelocity = new ArticulationReducedSpace(0f);
        }
    }

    /// <summary>
    /// Computes the Jacobian matrix for the robot's kinematics.
    /// </summary>
    /// <param name="numJoints"> Number of robot joints.</param>
    /// </param>
    private void CalculateJacobian(int numJoints)
    {
        if (_jacobian.ColumnCount != numJoints)
        {
            _jacobian = DenseMatrix.Build.Dense(_JacobianRows, numJoints); // Resize if necessary
        }

        for (int i = 0; i < numJoints; i++)
        {
            var joint = robotJoints[i];
            Transform jointTransform = joint.transform;
            Vector3 jointPosition = jointTransform.position;

            Vector3 worldRotationAxis = jointTransform
                .TransformDirection(joint.anchorRotation * Vector3.right)
                .normalized; // Normalize for robustness

            // Linear velocity component: v = omega x r
            // r is the vector from the joint to the end-effector.
            Vector3 r = _currentEndEffectorPosition - jointPosition;
            Vector3 linearVelocityContribution = Vector3.Cross(worldRotationAxis, r);

            // Angular velocity component
            Vector3 angularVelocityContribution = worldRotationAxis;

            // Assign to Jacobian matrix (column 'i' corresponds to joint 'i')
            // Top 3 rows: linear velocity
            _jacobian[0, i] = linearVelocityContribution.x;
            _jacobian[1, i] = linearVelocityContribution.y;
            _jacobian[2, i] = linearVelocityContribution.z;

            // Bottom 3 rows: angular velocity
            _jacobian[3, i] = angularVelocityContribution.x;
            _jacobian[4, i] = angularVelocityContribution.y;
            _jacobian[5, i] = angularVelocityContribution.z;
        }
    }

    /// <summary>
    /// Calculates the pseudo inverse (with damping factor) of the Jacobian
    /// matrix.
    /// </summary>
    private void CalculatePseudoInverseJacobian()
    {
        Matrix<double> JT = _jacobian.Transpose();
        Matrix<double> JJT = _jacobian * JT;
        Matrix<double> identity = DenseMatrix.Build.DenseIdentity(JJT.RowCount);
        try
        {
            Matrix<double> termToInvert =
                JJT + (_dampingFactorLambda * _dampingFactorLambda * identity);
            _pseudoInverseJacobian = JT * termToInvert.Inverse();
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning(
                $"DLS Matrix inversion failed: {ex.Message}. Falling back to standard pseudo-inverse."
            );
            _pseudoInverseJacobian = _jacobian.PseudoInverse(); // Fallback
        }
    }

    /// <summary>
    /// Performes one inverse kinematics step to compute and apply new joint
    /// angles that move the robot towards the target.
    /// </summary>
    public void PerformInverseKinematicsStep()
    {
        if (robotJoints == null || robotJoints.Length == 0)
        {
            Debug.LogWarning("No robot joints found or IK not initialized.");
            return;
        }
        if (robotGripperBase == null || _target == null)
        {
            Debug.LogError("EndEffector or Target is not assigned.");
            return;
        }

        CalculateDistanceToTarget(); // Get latest positions/rotations

        int numJoints = robotJoints.Length;

        Quaternion rotationDifference =
            _target.rotation * Quaternion.Inverse(_currentEndEffectorRotation);
        rotationDifference.ToAngleAxis(out float angleDegrees, out Vector3 rotationAxis);
        // Convert angle to radians and scale by axis to get error vector
        Vector3 orientationError = rotationAxis * angleDegrees * Mathf.Deg2Rad;

        // Using pre-allocated _errorVector
        _errorVector[0] = _vectorToTarget.x;
        _errorVector[1] = _vectorToTarget.y;
        _errorVector[2] = _vectorToTarget.z;
        _errorVector[3] = orientationError.x;
        _errorVector[4] = orientationError.y;
        _errorVector[5] = orientationError.z;

        if (_errorVector.L2Norm() < _convergenceThreshold)
        {
            _targetReached = true;

            if (_targetReached)
                Debug.Log("IK converged to target.");

            return; // Already close enough
        }

        // Calculate the 6xN Jacobian matrix
        CalculateJacobian(numJoints);

        // Compute the pseudo-inverse of the Jacobian
        CalculatePseudoInverseJacobian();

        _pseudoInverseJacobian.Multiply(_errorVector, _deltaTheta);

        for (int i = 0; i < numJoints; ++i)
        {
            _deltaTheta[i] = System.Math.Clamp(
                _deltaTheta[i],
                -_robotManagerInstance.maxRawJointStepRad,
                _robotManagerInstance.maxRawJointStepRad
            );
        }

        // Adaptive speed based on distance to target and
        // normalize by a scale factor
        float normalizedDistance = Mathf.Clamp01(
            _vectorToTarget.magnitude / robotGripperBase.lossyScale.x
        );
        float adaptiveGain = Mathf.Lerp(_minStepSpeedNearTarget, _maxStepSpeed, normalizedDistance); // Slower as it nears target

        float overallSpeedMultiplier = _robotManagerInstance.robotAdjustmentSpeed * adaptiveGain;

        for (int i = 0; i < numJoints; i++)
        {
            var joint = robotJoints[i];
            ArticulationDrive drive = joint.xDrive;

            // The change in angle from IK (deltaTheta) is typically in radians,
            // but xDrive takes degrees.
            float jointAngleChangeDeg = (float)_deltaTheta[i] * Mathf.Rad2Deg;

            // Apply the scaled change
            float newTargetAngle =
                drive.target + (float)jointAngleChangeDeg * overallSpeedMultiplier;

            // Clamp to joint limits
            newTargetAngle = Mathf.Clamp(newTargetAngle, drive.lowerLimit, drive.upperLimit);

            if (!Mathf.Approximately(newTargetAngle, drive.target))
            {
                drive.target = newTargetAngle;
                joint.xDrive = drive;
            }
        }
    }

    private void Start()
    {
        _simulationManagerInstance = SimulationManager.Instance;
        _robotManagerInstance = RobotManager.Instance;

        SetUpRobot();
    }

    private void FixedUpdate()
    {
        // Early exit if the robot should stop
        if (_simulationManagerInstance.stopRobot)
            return;

        if (!_targetReached)
        {
            // If the target is not reached perform inverse kinematics.
            PerformInverseKinematicsStep();
        }
    }
}
