using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Computes optimal grasp poses (position + orientation) for objects based on their geometry and position.
    /// Determines approach direction and gripper orientation for successful grasping.
    /// </summary>
    public static class GraspPlanner
    {
        private const float TOP_APPROACH_OFFSET = 0.01f; // Distance above object to approach from
        private const float SIDE_APPROACH_OFFSET = 0.02f; // Distance to side of object

        /// <summary>
        /// Calculate grasp pose (position + rotation) for a target object.
        /// Approaches from top with gripper pointing downward by default.
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
                    graspPosition = objectPosition + Vector3.up * (objectSize.y * 0.5f + TOP_APPROACH_OFFSET);
                    // Gripper points downward (rotate -90 degrees around X axis)
                    graspRotation = Quaternion.Euler(90f, 0f, 0f);
                    break;

                case GraspApproach.Front:
                    // Approach from front (along Z axis)
                    graspPosition = objectPosition + Vector3.forward * (objectSize.z * 0.5f + SIDE_APPROACH_OFFSET);
                    // Gripper points backward (rotate 180 degrees around Y axis)
                    graspRotation = Quaternion.Euler(0f, 180f, 0f);
                    break;

                case GraspApproach.Side:
                    // Approach from side (determine which side based on gripper position)
                    float deltaX = gripperPosition.x - objectPosition.x;
                    float sideOffset = deltaX > 0 ? SIDE_APPROACH_OFFSET : -SIDE_APPROACH_OFFSET;
                    graspPosition = objectPosition + Vector3.right * (objectSize.x * 0.5f + sideOffset);
                    // Gripper points toward object center
                    float rotationY = deltaX > 0 ? 90f : -90f;
                    graspRotation = Quaternion.Euler(0f, rotationY, 0f);
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
        private static Vector3 GetObjectSize(GameObject obj)
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

            // If gripper is significantly above object, approach from top
            if (delta.y > objectSize.y * 0.5f)
            {
                return GraspApproach.Top;
            }

            // Otherwise determine based on horizontal distance
            float horizontalDist = new Vector2(delta.x, delta.z).magnitude;

            // If close horizontally but at similar height, approach from side
            if (horizontalDist < objectSize.x * 2f && Mathf.Abs(delta.y) < objectSize.y)
            {
                return GraspApproach.Side;
            }

            // Default to top approach (most reliable)
            return GraspApproach.Top;
        }
    }

    /// <summary>
    /// Enum defining grasp approach directions.
    /// </summary>
    public enum GraspApproach
    {
        Top,    // Approach from above (gripper pointing down)
        Front,  // Approach from front (gripper pointing forward)
        Side    // Approach from side (gripper pointing horizontally)
    }
}
