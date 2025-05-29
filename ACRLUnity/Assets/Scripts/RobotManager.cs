using UnityEngine;

public class RobotManager : MonoBehaviour
{
    private readonly float[] _jointLengths = { 0.45f, 0.75f, 0.0f, 0.55f, 0.1f, 0.11f }; // Currently unused, but may be useful in the future

    // Configuration values for an AR4 robotic arm
    private float[] _stiffnessValues = { 15000, 12000, 10000, 8000, 5000, 2000 };
    private float[] _dampingValues = { 100, 150, 150, 100, 80, 50 };
    private float[] _forceLimits = { 1000, 1500, 1000, 800, 500, 300 };
    private float[] _driveUpperLimits = { 170, 90, 65, 135, 100, 180 };
    private float[] _driveLowerLimits = { -170, -90, -70, -135, -100, -180 };
    private float _robotSpeed = 0.3f;

    public GameObject leftTarget,
        rightTarget;
    public GameObject leftRobot,
        rightRobot;
    public static RobotManager Instance { get; private set; }

    [Range(0.001f, 1.0f)]
    public float resolution = 0.1f; // Resolution for movement

    [Range(0.001f, 0.1f)]
    public float raiseHeight = 0.05f; // Height to raise the robot arm

    public float GetStiffnessValue(int i) => _stiffnessValues[i];

    public float GetDampingValue(int i) => _dampingValues[i];

    public float GetForceLimits(int i) => _forceLimits[i];

    public float GetDriveUpperLimits(int i) => _driveUpperLimits[i];

    public float GetDriveLowerLimits(int i) => _driveLowerLimits[i];

    public float GetRobotSpeed() => _robotSpeed;

    /// <summary>
    /// Sets the speed of the robot.
    /// </summary>
    /// <param name="speed"> The speed to set for the robot.</param>
    public void SetRobotSpeed(float speed)
    {
        _robotSpeed = speed;
    }

    // Singleton pattern initialization
    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    private void Start()
    {
        leftRobot.GetComponent<RobotController>().StartRobot(leftTarget);
        rightRobot.GetComponent<RobotController>().StartRobot(rightTarget);
    }
}
