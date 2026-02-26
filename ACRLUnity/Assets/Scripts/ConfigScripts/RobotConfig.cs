using UnityEngine;

namespace Configuration
{
    [System.Serializable]
    public class JointConfiguration
    {
        [Header("Joint Parameters")]
        public float stiffness = 5000f;
        public float damping = 1500f;
        public float forceLimit = 2000f;
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
        [Range(0.1f, 1f)]
        public float adjustmentSpeed = 0.5f;

        public void InitializeDefaultAR4Profile()
        {
            joints = new JointConfiguration[6]
            {
                new JointConfiguration(5000f, 1500f, 2000f, 170f, -170f), // Base
                new JointConfiguration(4000f, 1200f, 2000f, 90f, -42.5f), // Shoulder
                new JointConfiguration(3000f, 800f, 1000f, 125f, -90f), // Elbow
                new JointConfiguration(2500f, 500f, 500f, 105f, -105f), // Wrist 1
                new JointConfiguration(2000f, 400f, 500f, 100f, -100f), // Wrist 2
                new JointConfiguration(1500f, 300f, 500f, 180f, -180f), // Wrist 3
            };
        }

#if UNITY_EDITOR
        /// <summary>
        /// Validate configuration values to ensure consistency.
        /// </summary>
        private void OnValidate()
        {
            // Validate adjustment speed
            adjustmentSpeed = Mathf.Clamp(adjustmentSpeed, 0.1f, 1f);

            // Validate each joint configuration
            if (joints != null)
            {
                for (int i = 0; i < joints.Length; i++)
                {
                    JointConfiguration joint = joints[i];
                    if (joint != null)
                    {
                        // Ensure positive values
                        joint.stiffness = Mathf.Max(0f, joint.stiffness);
                        joint.damping = Mathf.Max(0f, joint.damping);
                        joint.forceLimit = Mathf.Max(0f, joint.forceLimit);

                        // Ensure lower < upper limit
                        if (joint.lowerLimit >= joint.upperLimit)
                        {
                            joint.upperLimit = joint.lowerLimit + 10f;
                        }
                    }
                }
            }
        }
#endif
    }
}
