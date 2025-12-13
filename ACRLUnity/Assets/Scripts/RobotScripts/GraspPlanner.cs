using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Computes optimal grasp poses (position + orientation) for objects based on their geometry and position.
    /// Determines approach direction and gripper orientation for successful grasping.
    /// </summary>
    public static class GraspPlanner
    {
        private const float SIDE_APPROACH_OFFSET = 0.01f; // Distance to side of object

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
                    graspPosition =
                        objectPosition + Vector3.up * (objectSize.y * 0.5f);
                    // Gripper points downward (rotate 90 degrees around X axis)
                    graspRotation = Quaternion.Euler(90f, 0f, 0f);
                    break;

                case GraspApproach.Side:
                    // Approach from side (determine which side based on gripper position)

                    float deltaZ = gripperPosition.z - objectPosition.z;
                    float frontOffset = deltaZ > 0 ? SIDE_APPROACH_OFFSET : -SIDE_APPROACH_OFFSET;
                    graspPosition =
                        objectPosition + Vector3.forward * (objectSize.z * 0.3f + frontOffset);
                    // Gripper points toward object center
                    float rotationYFront = deltaZ > 0 ? 180f : 0f;
                    graspRotation = Quaternion.Euler(0f, rotationYFront, 0f);
                    break;

                case GraspApproach.Front:
                    // Approach from front/back (along Z axis, direction based on gripper position)
                    float deltaX = gripperPosition.x - objectPosition.x;
                    float sideOffset = deltaX > 0 ? SIDE_APPROACH_OFFSET : -SIDE_APPROACH_OFFSET;
                    graspPosition =
                        objectPosition + Vector3.right * (objectSize.x * 0.3f + sideOffset);
                    // Gripper points toward object center
                    float rotationY = deltaX > 0 ? -90f : 90f;
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
            float distanceY = Mathf.Abs(delta.y);
            float distanceZ = Mathf.Abs(delta.z);

            // Threshold for "close enough" to consider that axis aligned
            float alignmentThreshold = objectSize.x * 1.5f;

            // If vertically close, check horizontal approaches first
            if (distanceY < alignmentThreshold)
            {
                // If Z-aligned (object is in front/behind), prefer front approach
                if (distanceZ < alignmentThreshold)
                {
                    return GraspApproach.Front;
                }

                // If X-aligned (object is to the side), prefer side approach
                if (distanceX < alignmentThreshold)
                {
                    return GraspApproach.Side;
                }
            }

            // Otherwise, choose based on smallest distance (easiest approach)
            float minDistance = Mathf.Min(distanceX, distanceY, distanceZ);

            if (distanceY == minDistance)
            {
                return GraspApproach.Top;
            }

            if (distanceZ == minDistance)
            {
                return GraspApproach.Front;
            }

            return GraspApproach.Top;
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
