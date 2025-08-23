using UnityEngine;

public class RobotManager : MonoBehaviour
{
    // Configuration values for an AR4 robotic arm
    private float[] _stiffnessValues = { 800, 700, 600, 300, 200, 100 };
    private float[] _dampingValues = { 250, 200, 150, 100, 80, 50 };
    private float[] _forceLimits = { 1000, 1500, 1000, 800, 500, 300 };
    private float[] _driveUpperLimits = { 170, 90, 65, 135, 100, 180 };
    private float[] _driveLowerLimits = { -170, -90, -70, -135, -100, -180 };

    [SerializeField, Range(0.1f, 5f)]
    public float robotAdjustmentSpeed = 1.0f;

    [SerializeField, Range(0.01f, 1f)]
    public float convergenceThreshold = 0.1f;

    [Tooltip(
        "Maximum raw angular change (radians) a joint is allowed from one IK step, before adaptive speed scaling."
    )]
    public float maxRawJointStepRad = 0.1f; // Approx 5.7 degrees. Tune this value!

    public GameObject leftTarget,
        rightTarget;
    public GameObject leftRobot,
        rightRobot;
    public static RobotManager Instance { get; private set; }

    public float GetStiffnessValue(int i) => _stiffnessValues[i];

    public float GetDampingValue(int i) => _dampingValues[i];

    public float GetForceLimits(int i) => _forceLimits[i];

    public float GetDriveUpperLimits(int i) => _driveUpperLimits[i];

    public float GetDriveLowerLimits(int i) => _driveLowerLimits[i];

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
        leftRobot.GetComponent<RobotController>().SetTarget(leftTarget);
        rightRobot.GetComponent<RobotController>().SetTarget(rightTarget);
    }

    private void FixedUpdate()
    {
        if (leftTarget.transform.hasChanged)
        {
            leftRobot.GetComponent<RobotController>().SetTarget(leftTarget);
            leftTarget.transform.hasChanged = false;
        }

        if (rightTarget.transform.hasChanged)
        {
            rightRobot.GetComponent<RobotController>().SetTarget(rightTarget);
            rightTarget.transform.hasChanged = false;
        }
    }
}
