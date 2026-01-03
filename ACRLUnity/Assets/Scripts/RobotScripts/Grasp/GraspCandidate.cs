using UnityEngine;

namespace Robotics.Grasp
{
    /// <summary>
    /// Extended grasp candidate with scoring metadata, retreat pose, and validation flags.
    /// Used in MoveIt2-inspired grasp planning pipeline for multi-criteria evaluation.
    /// </summary>
    public struct GraspCandidate
    {
        // Pose information
        public Vector3 preGraspPosition;
        public Quaternion preGraspRotation;
        public Vector3 graspPosition;
        public Quaternion graspRotation;
        public Vector3 retreatPosition;      // Post-grasp retreat pose
        public Quaternion retreatRotation;

        // Gripper configuration
        public float preGraspGripperWidth;   // 0-1, 1=open
        public float graspGripperWidth;      // 0-1, typically 0=closed

        // Metadata for filtering/scoring
        public GraspApproach approachType;
        public float approachDistance;
        public float graspDepth;
        public Vector3 contactPointEstimate;
        public Vector3 approachDirection;

        // Scoring
        public float totalScore;
        public float ikScore;
        public float approachScore;
        public float depthScore;
        public float stabilityScore;
        public float antipodalScore;      // Antipodal grasp quality (force closure)

        // Validation state
        public bool ikValidated;
        public bool collisionValidated;
        public bool isValid => ikValidated && collisionValidated;

        /// <summary>
        /// Create a basic grasp candidate with minimal information.
        /// Other fields initialized to defaults.
        /// </summary>
        /// <param name="preGrasp">Pre-grasp position</param>
        /// <param name="preGraspRot">Pre-grasp rotation</param>
        /// <param name="grasp">Grasp position</param>
        /// <param name="graspRot">Grasp rotation</param>
        /// <param name="approach">Approach type</param>
        /// <returns>Initialized grasp candidate</returns>
        public static GraspCandidate Create(
            Vector3 preGrasp,
            Quaternion preGraspRot,
            Vector3 grasp,
            Quaternion graspRot,
            GraspApproach approach
        )
        {
            return new GraspCandidate
            {
                preGraspPosition = preGrasp,
                preGraspRotation = preGraspRot,
                graspPosition = grasp,
                graspRotation = graspRot,
                retreatPosition = grasp + Vector3.up * 0.1f, // Default retreat upward
                retreatRotation = graspRot,
                preGraspGripperWidth = 1.0f,
                graspGripperWidth = 0.0f,
                approachType = approach,
                approachDistance = Vector3.Distance(preGrasp, grasp),
                graspDepth = 0.5f,
                contactPointEstimate = grasp,
                approachDirection = (preGrasp - grasp).normalized,
                totalScore = 0f,
                ikScore = 0f,
                approachScore = 0f,
                depthScore = 0f,
                stabilityScore = 0f,
                antipodalScore = 0f,
                ikValidated = false,
                collisionValidated = false
            };
        }
    }

    /// <summary>
    /// Gripper geometry specification for grasp validation.
    /// Used to check if object fits between gripper fingers.
    /// </summary>
    [System.Serializable]
    public struct GripperGeometry
    {
        [Tooltip("Maximum opening width of gripper fingers (meters)")]
        public float maxWidth;

        [Tooltip("Width of each gripper finger pad (meters)")]
        public float fingerPadWidth;

        [Tooltip("Depth of gripper finger pads (meters)")]
        public float fingerPadDepth;

        [Tooltip("Length of gripper finger (meters)")]
        public float fingerLength;

        [Tooltip("Width of gripper finger (meters)")]
        public float fingerWidth;

        [Tooltip("Offset from wrist center to gripper center (meters)")]
        public Vector3 gripperCenterOffset;

        /// <summary>
        /// Check if an object of given size can be grasped by this gripper.
        /// </summary>
        /// <param name="objectSize">Size of the target object</param>
        /// <returns>True if object can fit between gripper fingers</returns>
        public bool CanGrasp(Vector3 objectSize)
        {
            float minDimension = Mathf.Min(objectSize.x, objectSize.y, objectSize.z);
            float maxDimension = Mathf.Max(objectSize.x, objectSize.y, objectSize.z);

            // Object must be small enough to fit in gripper opening
            // and large enough to make contact with finger pads
            return maxDimension < maxWidth && minDimension > fingerPadWidth * 0.1f;
        }

        /// <summary>
        /// Get default AR4 gripper geometry.
        /// </summary>
        /// <returns>Default gripper geometry</returns>
        public static GripperGeometry Default()
        {
            return new GripperGeometry
            {
                maxWidth = 0.08f,  // 8cm max opening
                fingerPadWidth = 0.015f,  // 1.5cm finger pad
                fingerPadDepth = 0.02f,  // 2cm depth
                fingerLength = 0.04f,  // 4cm finger length
                fingerWidth = 0.01f,  // 1cm finger width
                gripperCenterOffset = new Vector3(0f, 0f, 0.05f)  // 5cm forward offset
            };
        }
    }
}
