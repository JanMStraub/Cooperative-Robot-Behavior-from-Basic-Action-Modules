using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Forwards trigger events from finger colliders to the GripperContactSensor.
    /// Attach this to each finger GameObject that has a trigger collider.
    /// Note: Uses trigger colliders to work with ArticulationBody physics.
    /// </summary>
    public class GripperCollisionForwarder : MonoBehaviour
    {
        [Tooltip("The GripperContactSensor to forward collisions to")]
        public GripperContactSensor sensor;

        [Tooltip("Which finger this forwarder represents")]
        public GripperContactSensor.FingerType fingerType;

        void OnTriggerEnter(Collider collider)
        {
            // Ignore collisions with gripper's own colliders (parent and siblings)
            if (IsGripperSelfCollision(collider))
                return;

            if (sensor != null)
            {
                sensor.OnFingerTriggerEnter(collider, fingerType);
            }
        }

        void OnTriggerStay(Collider collider)
        {
            // Ignore collisions with gripper's own colliders
            if (IsGripperSelfCollision(collider))
                return;

            if (sensor != null)
            {
                sensor.OnFingerTriggerStay(collider, fingerType);
            }
        }

        void OnTriggerExit(Collider collider)
        {
            // Ignore collisions with gripper's own colliders
            if (IsGripperSelfCollision(collider))
                return;

            if (sensor != null)
            {
                sensor.OnFingerTriggerExit(collider, fingerType);
            }
        }

        /// <summary>
        /// Check if the collider belongs to the gripper itself (parent or sibling).
        /// Returns true for self-collisions that should be ignored.
        /// </summary>
        private bool IsGripperSelfCollision(Collider collider)
        {
            if (collider == null)
                return true;

            // Check if collider is on the sensor's GameObject (gripper root)
            if (sensor != null && collider.gameObject == sensor.gameObject)
                return true;

            // Check if collider is on a sibling finger (same parent)
            Transform colliderTransform = collider.transform;
            Transform thisTransform = transform;

            // If both have the same parent, they're siblings (both fingers)
            if (colliderTransform.parent != null && thisTransform.parent != null &&
                colliderTransform.parent == thisTransform.parent)
            {
                return true;
            }

            return false;
        }
    }
}
