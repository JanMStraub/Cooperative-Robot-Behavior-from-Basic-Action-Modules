using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using UnityEngine;

public class RobotController : MonoBehaviour
{
    private SimulationManager _simulationManagerInstance;
    private RobotManager _robotManagerInstance;
    private Vector3 _distanceToTarget;
    private Vector3 _endEffectorPosition;
    private Vector3 _defaultPosition;
    private Vector3 _target;
    private float _tolerance = 0.001f;
    private Quaternion _targetRotation;
    private Quaternion _gripperRotation;
    private bool _targetReached = true;
    private bool _waitForCommand = true;

    public ArticulationBody[] robotJoints;
    public Transform[] robotGripper;

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
        // Ensure robot joints are assigned
        if (robotJoints == null || robotJoints.Length == 0)
        {
            Debug.LogError(
                "No ArticulationBody components assigned to the left robot. Check your hierarchy."
            );
            return;
        }

        // Configure joint drives
        for (int i = 0; i < robotJoints.Length; i++)
        {
            var drive = robotJoints[i].xDrive;
            drive.stiffness = _robotManagerInstance.GetStiffnessValue(i);
            drive.damping = _robotManagerInstance.GetDampingValue(i);
            drive.forceLimit = _robotManagerInstance.GetForceLimits(i);
            drive.upperLimit = _robotManagerInstance.GetDriveUpperLimits(i);
            drive.lowerLimit = _robotManagerInstance.GetDriveLowerLimits(i);
            robotJoints[i].xDrive = drive;
        }
    }

    /// <summary>
    /// Moves the robot back to its default position.
    /// </summary>
    public void ReturnToDefaultPosition()
    {
        Debug.Log("Returning to default position: " + _defaultPosition);

        MoveTo(_defaultPosition);
    }

    /// <summary>
    /// Moves the robot to a specified target position.
    /// </summary>
    /// <param name="target"> The target position to move the robot to.</param>
    public void MoveTo(Vector3 target)
    {
        _targetReached = false;
        _waitForCommand = false;

        _target = target;
    }

    /// <summary>
    /// Calculates the distance from the closest end-effector sensor
    /// to the target position.
    /// </summary>
    private void CalculateDistanceToTarget()
    {
        float minDistance = float.MaxValue;

        for (int i = 0; i < robotGripper.Length; i++)
        {
            Vector3 gripperPosition = robotGripper[i].transform.position;
            Vector3 distanceToGripper = _target - gripperPosition;

            float distanceMagnitude = distanceToGripper.magnitude;
            if (distanceMagnitude < minDistance)
            {
                minDistance = distanceMagnitude;
                _endEffectorPosition = gripperPosition;
                _distanceToTarget = distanceToGripper;
            }
        }
    }

    /// <summary>
    /// Computes the Jacobian matrix for the robot's kinematics.
    /// </summary>
    /// <param name="numJoints"> Number of robot joints.</param>
    /// <param name="endEffectorPosition"> Current position of the end-effector.
    /// </param>
    /// <returns>
    /// A 6xN Jacobian matrix representing the robot's movement.
    /// </returns>
    private Matrix<double> CalculateJacobian(int numJoints, Vector3 endEffectorPosition)
    {
        Matrix<double> jacobian = DenseMatrix.Build.Dense(6, numJoints);

        for (int i = 0; i < numJoints; i++)
        {
            var joint = robotJoints[i];
            var jointTransform = joint.transform;
            Vector3 jointPosition = jointTransform.position;

            // Determine the rotation axis in world space
            Quaternion anchorRotation = joint.anchorRotation;
            Vector3 worldRotationAxis = jointTransform.TransformDirection(
                anchorRotation * Vector3.right
            );

            // Compute linear velocity part (same as before)
            Vector3 linearVelocity = Vector3.Cross(
                worldRotationAxis,
                endEffectorPosition - jointPosition
            );

            // Compute angular velocity part (just the rotation axis)
            Vector3 angularVelocity = worldRotationAxis;

            // Assign values to the Jacobian matrix
            jacobian[0, i] = linearVelocity.x;
            jacobian[1, i] = linearVelocity.y;
            jacobian[2, i] = linearVelocity.z;

            jacobian[3, i] = angularVelocity.x;
            jacobian[4, i] = angularVelocity.y;
            jacobian[5, i] = angularVelocity.z;
        }

        return jacobian;
    }

    /// <summary>
    /// Calculates the pseudo inverse of the Jacobian matrix.
    /// </summary>
    /// <param name="matrix"> The Jacobian matrix.</param>
    /// <returns>
    /// The pseudo-inverse of the input matrix.
    /// </returns>
    private Matrix<double> PseudoInverse(Matrix<double> matrix)
    {
        return matrix.PseudoInverse();
    }

    /// <summary>
    /// Uses inverse kinematics to compute and apply new joint angles
    /// that move the robot towards the target.
    /// </summary>
    public void InverseKinematics()
    {
        int numJoints = robotJoints.Length;
        if (numJoints == 0)
        {
            Debug.LogWarning("No robot joints found for inverse kinematics.");
            return;
        }

        // Calculate the 6xN Jacobian matrix
        Matrix<double> jacobian = CalculateJacobian(numJoints, _endEffectorPosition);

        // Compute the pseudo-inverse of the Jacobian
        Matrix<double> pseudoInverseJacobian = PseudoInverse(jacobian);

        // Position error vector (X, Y, Z)
        Vector<double> positionError = Vector<double>.Build.DenseOfArray(
            new double[] { _distanceToTarget.x, _distanceToTarget.y, _distanceToTarget.z }
        );

        // Compute orientation error (rotation difference)
        Quaternion currentRotation = _gripperRotation;
        Quaternion rotationDifference = _targetRotation * Quaternion.Inverse(currentRotation);

        // Convert quaternion difference to axis-angle representation
        Vector3 rotationAxis;
        float rotationAngle;
        rotationDifference.ToAngleAxis(out rotationAngle, out rotationAxis);
        Vector3 orientationError = rotationAxis * Mathf.Deg2Rad * rotationAngle; // Convert to radians

        // Combine position and orientation errors into a single 6D error vector
        Vector<double> errorVector = Vector<double>.Build.DenseOfArray(
            new double[]
            {
                positionError[0],
                positionError[1],
                positionError[2],
                orientationError.x,
                orientationError.y,
                orientationError.z,
            }
        );

        float distanceFactor = Mathf.Clamp01(_distanceToTarget.magnitude); // 0 to 1 based on distance
        float adaptiveSpeed = Mathf.Lerp(0.3f, 1.0f, distanceFactor); // Slower as it nears target
        Vector<double> deltaTheta = pseudoInverseJacobian * errorVector * adaptiveSpeed;

        // Apply the joint adjustments
        float robotSpeed = RobotManager.Instance.GetRobotSpeed();
        for (int i = 0; i < numJoints; i++)
        {
            var joint = robotJoints[i];
            var drive = joint.xDrive;

            float newTarget = Mathf.Clamp(
                drive.target + (float)deltaTheta[i] * robotSpeed,
                drive.lowerLimit,
                drive.upperLimit
            );

            if (!Mathf.Approximately(newTarget, drive.target))
            {
                drive.target = newTarget;
                joint.xDrive = drive;
            }
        }
    }

    /// <summary>
    /// Checks if the robot has reached the target within the allowed tolerance.
    /// If the tolerance is met, it stops the robot and updates the simulation
    /// state.
    /// </summary>
    private void CheckTolerance()
    {
        if (_distanceToTarget.magnitude < _tolerance)
        {
            _waitForCommand = true;
            _targetReached = true;

            Debug.Log("Tolerance reached");
        }
    }

    public void StartRobot(GameObject target)
    {
        _targetRotation = target.transform.rotation;
        MoveTo(target.transform.position);
    }

    private void Start()
    {
        _distanceToTarget = Vector3.zero;
        _endEffectorPosition = Vector3.zero;
        _target = robotGripper[0].transform.position;
        _defaultPosition = robotGripper[0].transform.position;

        _gripperRotation = robotGripper[0].transform.rotation;

        _simulationManagerInstance = SimulationManager.Instance;
        _robotManagerInstance = RobotManager.Instance;

        SetUpRobot();
    }

    private void FixedUpdate()
    {
        // Early exit if the robot should stop
        if (_simulationManagerInstance.stopRobot)
            return;

        CheckTolerance();

        // If there is an active target
        if (_target != Vector3.zero)
        {
            if (_targetReached)
            {
                // Update the distance to the target before any movement logic
                CalculateDistanceToTarget();
                if (_waitForCommand)
                {
                    // If the target is not reached or a robot command is active, perform inverse kinematics.
                    InverseKinematics();
                }
            }
        }
    }
}
