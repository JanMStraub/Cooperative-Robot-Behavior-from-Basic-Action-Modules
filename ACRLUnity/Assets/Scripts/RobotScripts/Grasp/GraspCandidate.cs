using UnityEngine;

namespace Robotics.Grasp
{
    /// <summary>
    /// Extended grasp candidate with scoring metadata, retreat pose, and validation flags.
    /// Used in MoveIt2-inspired grasp planning pipeline for multi-criteria evaluation.
    /// </summary>
    public class GraspCandidate
    {
        public Vector3 preGraspPosition;
        public Quaternion preGraspRotation;
        public Vector3 graspPosition;
        public Quaternion graspRotation;
        public Vector3 retreatPosition;
        public Quaternion retreatRotation;

        public float[] preGraspJointPositions;
        public float[] graspJointPositions;

        public float preGraspGripperWidth;
        public float graspGripperWidth;

        public GraspApproach approachType;
        public float approachDistance;
        public float graspDepth;
        public Vector3 contactPointEstimate;
        public Vector3 approachDirection;

        public float totalScore;
        public float ikScore;
        public float approachScore;
        public float depthScore;
        public float stabilityScore;
        public float antipodalScore;

        public bool ikValidated;
        public bool collisionValidated;
        public bool isValid => ikValidated && collisionValidated;

        public bool useSimplifiedExecution;

        /// <summary>
        /// Create a basic grasp candidate with minimal information.
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
            Vector3 approachDir = (preGrasp - grasp).normalized;
            float approachDist = Vector3.Distance(preGrasp, grasp);

            return new GraspCandidate
            {
                preGraspPosition = preGrasp,
                preGraspRotation = preGraspRot,
                graspPosition = grasp,
                graspRotation = graspRot,
                retreatPosition = grasp + (approachDir * 0.1f),
                retreatRotation = graspRot,
                preGraspGripperWidth = 1.0f,
                graspGripperWidth = 0.0f,
                approachType = approach,
                approachDistance = approachDist,
                graspDepth = 0.5f,
                contactPointEstimate = grasp,
                approachDirection = approachDir,
                totalScore = 0f,
                ikScore = 0f,
                approachScore = 0f,
                depthScore = 0f,
                stabilityScore = 0f,
                antipodalScore = 0f,
                ikValidated = false,
                collisionValidated = false,
                useSimplifiedExecution = false,
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
                maxWidth = 0.08f, // 8cm max opening
                fingerPadWidth = 0.015f, // 1.5cm finger pad
                fingerPadDepth = 0.02f, // 2cm depth
                fingerLength = 0.04f, // 4cm finger length
                fingerWidth = 0.01f, // 1cm finger width
                gripperCenterOffset = new Vector3(0f, 0f, 0.05f), // 5cm forward offset
            };
        }
    }
}
