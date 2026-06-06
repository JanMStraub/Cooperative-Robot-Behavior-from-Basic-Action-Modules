using UnityEngine;

namespace Robotics.Grasp
{
    /// <summary>Utility methods for grasp planning (object size calculation, approach determination).</summary>
    public static class GraspUtilities
    {
        private const string _logPrefix = "[GRASP_UTILITIES]";

        // Uses local bounds to avoid AABB inflation on rotated objects.
        // Priority: BoxCollider local size > Renderer bounds > Collider AABB (fallback, inaccurate when rotated).
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
    }
}
