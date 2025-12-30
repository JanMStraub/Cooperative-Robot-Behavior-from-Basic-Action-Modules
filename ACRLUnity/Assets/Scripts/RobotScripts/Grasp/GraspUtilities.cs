using UnityEngine;

namespace Robotics.Grasp
{
    /// <summary>
    /// Utility methods for grasp planning (object size calculation, approach determination).
    /// Extracted from deprecated GraspPlanner for use in new pipeline architecture.
    /// </summary>
    public static class GraspUtilities
    {
        private const float SIDE_APPROACH_OFFSET = 0.01f;

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

        /// <summary>
        /// Calculate basic grasp position and rotation for a given approach.
        /// Simple calculation for fallback or single-candidate generation.
        /// </summary>
        /// <param name="objectPosition">Object center position</param>
        /// <param name="objectSize">Object dimensions</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <param name="approach">Approach direction</param>
        /// <returns>Grasp position and rotation</returns>
        public static (Vector3 position, Quaternion rotation) CalculateBasicGraspPose(
            Vector3 objectPosition,
            Vector3 objectSize,
            Vector3 gripperPosition,
            GraspApproach approach
        )
        {
            Vector3 graspPosition;
            Quaternion graspRotation;

            switch (approach)
            {
                case GraspApproach.Top:
                    graspPosition = objectPosition + Vector3.up * (objectSize.y * 0.5f);
                    graspRotation = Quaternion.Euler(90f, 0f, 0f);
                    break;

                case GraspApproach.Side:
                    float deltaX = gripperPosition.x - objectPosition.x;
                    float sideSign = deltaX > 0 ? 1f : -1f;
                    graspPosition = objectPosition + Vector3.right * sideSign * (objectSize.x * 0.5f + SIDE_APPROACH_OFFSET);
                    graspRotation = Quaternion.Euler(0f, deltaX > 0 ? -90f : 90f, 0f);
                    break;

                case GraspApproach.Front:
                    float deltaZ = gripperPosition.z - objectPosition.z;
                    float frontSign = deltaZ > 0 ? 1f : -1f;
                    graspPosition = objectPosition + Vector3.forward * frontSign * (objectSize.z * 0.5f + SIDE_APPROACH_OFFSET);
                    graspRotation = Quaternion.Euler(0f, deltaZ > 0 ? 180f : 0f, 0f);
                    break;

                default:
                    graspPosition = objectPosition;
                    graspRotation = Quaternion.identity;
                    break;
            }

            return (graspPosition, graspRotation);
        }
    }
}
