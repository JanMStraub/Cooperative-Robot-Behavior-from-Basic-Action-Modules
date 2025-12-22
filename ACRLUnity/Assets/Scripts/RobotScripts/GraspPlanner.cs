using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Grasp plan containing both pre-grasp (approach) and final grasp poses.
    /// Two-stage approach prevents collisions and improves grasp success.
    /// </summary>
    public struct GraspPlan
    {
        public Vector3 preGraspPosition;    // Safe approach waypoint
        public Quaternion preGraspRotation;
        public Vector3 graspPosition;       // Final grasp pose
        public Quaternion graspRotation;
        public float preGraspGripperWidth;  // Gripper opening for approach (0-1, 1=open)
        public float graspGripperWidth;     // Gripper opening for final grasp (0-1)
    }

    /// <summary>
    /// Computes optimal grasp poses (position + orientation) for objects based on their geometry and position.
    /// Determines approach direction and gripper orientation for successful grasping.
    /// Uses two-waypoint planning: pre-grasp waypoint → final grasp pose.
    /// </summary>
    public static class GraspPlanner
    {
        private const float SIDE_APPROACH_OFFSET = 0.01f; // Distance to side of object
        private const float PRE_GRASP_DISTANCE = 0.08f;   // Distance for pre-grasp waypoint (8cm before grasp)

        /// <summary>
        /// Calculate complete grasp plan with pre-grasp waypoint and final grasp pose.
        /// Two-stage approach prevents collisions and ensures reliable grasping.
        /// </summary>
        /// <param name="targetObject">The object to grasp</param>
        /// <param name="gripperPosition">Current position of gripper (for approach planning)</param>
        /// <param name="approachDirection">Preferred approach direction (default: top-down)</param>
        /// <returns>Complete grasp plan with pre-grasp and grasp poses</returns>
        public static GraspPlan CalculateGraspPlan(
            GameObject targetObject,
            Vector3 gripperPosition,
            GraspApproach approachDirection = GraspApproach.Top
        )
        {
            Vector3 objectPosition = targetObject.transform.position;
            Vector3 objectSize = GetObjectSize(targetObject);

            Vector3 graspPosition;
            Quaternion graspRotation;
            Vector3 approachVector; // Direction to offset for pre-grasp

            switch (approachDirection)
            {
                case GraspApproach.Top:
                    // Approach from above
                    graspPosition = objectPosition + Vector3.up * (objectSize.y * 0.5f);
                    graspRotation = Quaternion.Euler(90f, 0f, 0f);
                    approachVector = Vector3.up; // Offset upward for pre-grasp
                    break;

                case GraspApproach.Side:
                    // Approach from side (along X axis - left/right)
                    float deltaX = gripperPosition.x - objectPosition.x;
                    float sideSign = deltaX > 0 ? 1f : -1f;
                    graspPosition =
                        objectPosition + Vector3.right * sideSign * (objectSize.x * 0.5f + SIDE_APPROACH_OFFSET);
                    graspRotation = Quaternion.Euler(0f, deltaX > 0 ? -90f : 90f, 0f);
                    approachVector = Vector3.right * sideSign; // Offset along X for pre-grasp
                    break;

                case GraspApproach.Front:
                    // Approach from front/back (along Z axis - forward/backward)
                    float deltaZ = gripperPosition.z - objectPosition.z;
                    float frontSign = deltaZ > 0 ? 1f : -1f;
                    graspPosition =
                        objectPosition + Vector3.forward * frontSign * (objectSize.z * 0.5f + SIDE_APPROACH_OFFSET);
                    graspRotation = Quaternion.Euler(0f, deltaZ > 0 ? 180f : 0f, 0f);
                    approachVector = Vector3.forward * frontSign; // Offset along Z for pre-grasp
                    break;

                default:
                    graspPosition = objectPosition;
                    graspRotation = Quaternion.identity;
                    approachVector = Vector3.up;
                    break;
            }

            // Calculate pre-grasp waypoint (offset away from object)
            Vector3 preGraspPosition = graspPosition + approachVector * PRE_GRASP_DISTANCE;

            return new GraspPlan
            {
                preGraspPosition = preGraspPosition,
                preGraspRotation = graspRotation, // Same rotation for both stages
                graspPosition = graspPosition,
                graspRotation = graspRotation,
                preGraspGripperWidth = 1.0f, // Fully open for approach
                graspGripperWidth = 0.0f     // Fully closed for grasp
            };
        }

        /// <summary>
        /// Calculate grasp pose (position + rotation) for a target object.
        /// Legacy method - returns only final grasp pose without pre-grasp waypoint.
        /// Consider using CalculateGraspPlan() for better collision avoidance.
        /// </summary>
        /// <param name="targetObject">The object to grasp</param>
        /// <param name="gripperPosition">Current position of gripper (for approach planning)</param>
        /// <param name="approachDirection">Preferred approach direction (default: top-down)</param>
        /// <returns>Grasp pose with position and rotation</returns>
        public static (Vector3 position, Quaternion rotation) CalculateGraspPose(
            GameObject targetObject,
            Vector3 gripperPosition,
            GraspApproach approachDirection = GraspApproach.Top
        )
        {
            Vector3 objectPosition = targetObject.transform.position;
            Vector3 objectSize = GetObjectSize(targetObject);

            Vector3 graspPosition;
            Quaternion graspRotation;

            switch (approachDirection)
            {
                case GraspApproach.Top:
                    // Approach from above
                    graspPosition = objectPosition + Vector3.up * (objectSize.y * 0.5f);
                    // Gripper points downward (rotate 90 degrees around X axis)
                    graspRotation = Quaternion.Euler(90f, 0f, 0f);
                    break;

                case GraspApproach.Side:
                    // Approach from side (along X axis - left/right)
                    float deltaX = gripperPosition.x - objectPosition.x;
                    // Position gripper to the side of object (along X axis)
                    // Use half object width + offset to approach from edge
                    float sideSign = deltaX > 0 ? 1f : -1f;
                    graspPosition =
                        objectPosition + Vector3.right * sideSign * (objectSize.x * 0.5f + SIDE_APPROACH_OFFSET);
                    // Gripper points toward object center (rotate to face inward)
                    float rotationY = deltaX > 0 ? -90f : 90f;
                    graspRotation = Quaternion.Euler(0f, rotationY, 0f);
                    break;

                case GraspApproach.Front:
                    // Approach from front/back (along Z axis - forward/backward)
                    float deltaZ = gripperPosition.z - objectPosition.z;
                    // Position gripper in front/back of object (along Z axis)
                    // Use half object depth + offset to approach from edge
                    float frontSign = deltaZ > 0 ? 1f : -1f;
                    graspPosition =
                        objectPosition + Vector3.forward * frontSign * (objectSize.z * 0.5f + SIDE_APPROACH_OFFSET);
                    // Gripper points toward object center (rotate to face inward)
                    float rotationYFront = deltaZ > 0 ? 180f : 0f;
                    graspRotation = Quaternion.Euler(0f, rotationYFront, 0f);
                    break;

                default:
                    graspPosition = objectPosition;
                    graspRotation = Quaternion.identity;
                    break;
            }

            return (graspPosition, graspRotation);
        }

        /// <summary>
        /// Get the size of an object based on its collider bounds.
        /// </summary>
        /// <param name="obj">The object to measure</param>
        /// <returns>Size vector (x, y, z)</returns>
        public static Vector3 GetObjectSize(GameObject obj)
        {
            Collider collider = obj.GetComponent<Collider>();
            if (collider != null)
            {
                return collider.bounds.size;
            }

            // Fallback to default cube size if no collider
            return Vector3.one * 0.05f;
        }

        /// <summary>
        /// Determine optimal grasp approach based on object and gripper positions.
        /// </summary>
        /// <param name="objectPosition">Position of target object</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <param name="objectSize">Size of the object</param>
        /// <returns>Recommended grasp approach</returns>
        public static GraspApproach DetermineOptimalApproach(
            Vector3 objectPosition,
            Vector3 gripperPosition,
            Vector3 objectSize
        )
        {
            // Calculate relative position
            Vector3 delta = gripperPosition - objectPosition;

            float distanceX = Mathf.Abs(delta.x);
            float distanceZ = Mathf.Abs(delta.z);

            // If gripper is significantly above object, prefer Top approach
            // This is the most reliable grasp for small objects
            if (delta.y > objectSize.y * 0.5f)
            {
                return GraspApproach.Top;
            }

            // If gripper is at same height or below, use horizontal approaches
            // Choose the axis where gripper is farthest from object (most clearance)
            if (distanceX > distanceZ)
            {
                // Gripper is more displaced in X - approach from side (along X axis)
                return GraspApproach.Side;
            }
            else
            {
                // Gripper is more displaced in Z - approach from front (along Z axis)
                return GraspApproach.Front;
            }
        }
    }

    /// <summary>
    /// Enum defining grasp approach directions.
    /// </summary>
    public enum GraspApproach
    {
        Top, // Approach from above (gripper pointing down)
        Front, // Approach from front (gripper pointing forward)
        Side, // Approach from side (gripper pointing horizontally)
    }
}
