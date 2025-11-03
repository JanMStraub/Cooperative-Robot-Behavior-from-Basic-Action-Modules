using UnityEngine;

namespace Configuration
{
    [System.Serializable]
    public class JointConfiguration
{
    [Header("Joint Parameters")]
    public float stiffness = 800f;
    public float damping = 250f;
    public float forceLimit = 1000f;
    public float upperLimit = 170f;
    public float lowerLimit = -170f;

    [Header("Performance Settings")]
    public float maxVelocity = 180f; // degrees per second
    public float acceleration = 360f; // degrees per second squared

    public JointConfiguration() { }

    public JointConfiguration(float stiff, float damp, float force, float upper, float lower)
    {
        stiffness = stiff;
        damping = damp;
        forceLimit = force;
        upperLimit = upper;
        lowerLimit = lower;
    }
}

[CreateAssetMenu(fileName = "RobotProfile", menuName = "Robotics/RobotProfile")]
public class RobotConfig : ScriptableObject
{
    [Header("Robot Identity")]
    public string profileName = "AR4_Default";
    public string description = "Standard AR4 robotic arm configuration";

    [Header("Joint Configurations")]
    public JointConfiguration[] joints = new JointConfiguration[6];

    [Header("IK Settings")]
    [Range(0.1f, 5f)]
    public float adjustmentSpeed = 1.0f;

    [Range(0.001f, 0.5f)]
    public float convergenceThreshold = 0.1f;

    [Range(0.01f, 0.5f)]
    public float maxJointStepRad = 0.1f;

    [Header("Performance Limits")]
    public float maxReachDistance = 0.8f;
    public float minReachDistance = 0.1f;
    public int maxIKIterations = 100;
    public float ikTimeout = 5f;


    public void InitializeDefaultAR4Profile()
    {
        joints = new JointConfiguration[6]
        {
            new JointConfiguration(800, 250, 1000, 170, -170), // Base
            new JointConfiguration(700, 200, 1500, 90, -90), // Shoulder
            new JointConfiguration(600, 150, 1000, 65, -70), // Elbow
            new JointConfiguration(300, 100, 800, 135, -135), // Wrist 1
            new JointConfiguration(200, 80, 500, 100, -100), // Wrist 2
            new JointConfiguration(100, 50, 300, 180, -180), // Wrist 3
        };
    }
}
}
