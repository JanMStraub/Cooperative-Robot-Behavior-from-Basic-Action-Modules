using UnityEngine;

namespace Configuration
{
    [System.Serializable]
    public class JointConfiguration
    {
        [Header("Joint Parameters")]
        public float stiffness = 8000f;
        public float damping = 2000f;
        public float forceLimit = 5000f;
        public float upperLimit = 170f;
        public float lowerLimit = -170f;

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

        [Range(0.01f, 0.5f)]
        public float maxJointStepRad = 0.1f;

        public void InitializeDefaultAR4Profile()
        {
            joints = new JointConfiguration[6]
            {
                new JointConfiguration(8000, 2000, 5000, 170, -170), // Base
                new JointConfiguration(6000, 1500, 6000, 150, 22), // Shoulder
                new JointConfiguration(5000, 1200, 5000, 52, -89), // Elbow
                new JointConfiguration(3000, 1000, 4000, 180, -180), // Wrist 1
                new JointConfiguration(2000, 800, 3000, 105, -105), // Wrist 2
                new JointConfiguration(1000, 600, 2000, 180, -180), // Wrist 3
            };
        }
    }
}

/*
                new JointConfiguration(10000, 2000, 5000, 170, -170), // Base
                new JointConfiguration(8000, 1500, 6000, 90, -90), // Shoulder
                new JointConfiguration(7000, 1200, 5000, 65, -70), // Elbow
                new JointConfiguration(5000, 1000, 4000, 135, -135), // Wrist 1
                new JointConfiguration(4000, 800, 3000, 100, -100), // Wrist 2
                new JointConfiguration(3000, 600, 2000, 180, -180), // Wrist 3
*/