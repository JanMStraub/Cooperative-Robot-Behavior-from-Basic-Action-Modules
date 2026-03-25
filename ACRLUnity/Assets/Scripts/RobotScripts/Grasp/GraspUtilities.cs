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

        private const string _logPrefix = "[GRASP_UTILITIES]";

        /// <summary>
        /// Get the size of an object based on its collider bounds.
        /// CRITICAL FIX: Uses local bounds to avoid AABB inflation on rotated objects.
        ///
        /// Priority:
        /// 1. BoxCollider local size (most accurate for boxy objects)
        /// 2. Renderer bounds (tighter than AABB for meshes)
        /// 3. Collider AABB (fallback, inaccurate for rotated objects)
        /// </summary>
        /// <param name="obj">The object to measure</param>
        /// <returns>Size vector (x, y, z) in local object space</returns>
        public static Vector3 GetObjectSize(GameObject obj)
        {
            BoxCollider box = obj.GetComponent<BoxCollider>();
            if (box != null)
            {
                Vector3 size = Vector3.Scale(box.size, obj.transform.lossyScale);
                Debug.Log(
                    $"{_logPrefix} Object '{obj.name}' size from BoxCollider: {size}, "
                        + $"localSize: {box.size}, lossyScale: {obj.transform.lossyScale}"
                );
                return size;
            }

            Renderer renderer = obj.GetComponent<Renderer>();
            if (renderer != null)
            {
                Vector3 size = renderer.bounds.size;
                Debug.Log($"{_logPrefix} Object '{obj.name}' size from Renderer: {size}");
                return size;
            }

            Collider collider = obj.GetComponent<Collider>();
            if (collider != null)
            {
                Vector3 size = collider.bounds.size;
                Debug.LogWarning(
                    $"{_logPrefix} Object '{obj.name}' using AABB size (may be inaccurate if rotated): {size}"
                );
                return size;
            }

            Debug.LogWarning(
                $"{_logPrefix} Object '{obj.name}' has no collider or renderer, using default size"
            );
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
            Vector3 delta = gripperPosition - objectPosition;

            float distanceX = Mathf.Abs(delta.x);
            float distanceZ = Mathf.Abs(delta.z);

            Debug.Log($"{_logPrefix} Object: {objectPosition}, Gripper: {gripperPosition}");
            Debug.Log($"{_logPrefix} Delta: {delta}, ObjectSize: {objectSize}");
            Debug.Log(
                $"{_logPrefix} distanceX: {distanceX:F3}, distanceZ: {distanceZ:F3}, delta.y: {delta.y:F3}, threshold: {objectSize.y * 0.5f:F3}"
            );

            if (delta.y > objectSize.y * 0.5f)
            {
                Debug.Log($"{_logPrefix} Selected: TOP (gripper above object)");
                return GraspApproach.Top;
            }

            if (distanceX > distanceZ)
            {
                Debug.Log($"{_logPrefix} Selected: SIDE (distanceX > distanceZ)");
                return GraspApproach.Side;
            }
            else
            {
                Debug.Log($"{_logPrefix} Selected: FRONT (distanceZ >= distanceX)");
                return GraspApproach.Front;
            }
        }

        /// <summary>
        /// Calculate basic grasp position and rotation for a given approach.
        /// Simple calculation for fallback or single-candidate generation.
        ///
        /// Position offsets and the base rotation are both composed through
        /// <paramref name="objectRotation"/> so that a rotated object receives a
        /// correctly-oriented grasp pose. Pass <c>default</c> (or omit) for
        /// axis-aligned objects — it degrades to <c>Quaternion.identity</c>.
        /// </summary>
        /// <param name="objectPosition">Object center position</param>
        /// <param name="objectSize">Object dimensions in local space</param>
        /// <param name="gripperPosition">Current gripper position</param>
        /// <param name="approach">Approach direction</param>
        /// <param name="objectRotation">World-space rotation of the target object (default: identity)</param>
        /// <returns>Grasp position and rotation in world space</returns>
        public static (Vector3 position, Quaternion rotation) CalculateBasicGraspPose(
            Vector3 objectPosition,
            Vector3 objectSize,
            Vector3 gripperPosition,
            GraspApproach approach,
            Quaternion objectRotation = default
        )
        {
            if (objectRotation == default)
                objectRotation = Quaternion.identity;

            Vector3 graspPosition;
            Quaternion baseRotation;

            switch (approach)
            {
                case GraspApproach.Top:
                    graspPosition = objectPosition + objectRotation * (Vector3.up * (objectSize.y * 0.5f));
                    baseRotation = Quaternion.Euler(90f, 0f, 0f);
                    break;

                case GraspApproach.Side:
                    float deltaX = gripperPosition.x - objectPosition.x;
                    float sideSign = deltaX > 0 ? 1f : -1f;
                    graspPosition =
                        objectPosition
                        + objectRotation * (Vector3.right * sideSign * (objectSize.x * 0.5f + SIDE_APPROACH_OFFSET));
                    baseRotation = Quaternion.Euler(0f, deltaX > 0 ? -90f : 90f, 0f);
                    break;

                case GraspApproach.Front:
                    float deltaZ = gripperPosition.z - objectPosition.z;
                    float frontSign = deltaZ > 0 ? 1f : -1f;
                    graspPosition =
                        objectPosition
                        + objectRotation * (Vector3.forward * frontSign * (objectSize.z * 0.5f + SIDE_APPROACH_OFFSET));
                    baseRotation = Quaternion.Euler(0f, deltaZ > 0 ? 180f : 0f, 0f);
                    break;

                default:
                    graspPosition = objectPosition;
                    baseRotation = Quaternion.identity;
                    break;
            }

            return (graspPosition, objectRotation * baseRotation);
        }
    }
}
