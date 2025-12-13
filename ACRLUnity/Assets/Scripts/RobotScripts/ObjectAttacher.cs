using UnityEngine;

public class ObjectAttacher : MonoBehaviour
{
    [Header("Settings")]
    [Tooltip("The Tag of objects you want to stick. Leave empty to attach everything.")]
    public string targetTag = "Attachable";

    [Tooltip("If true, the attached object will stop reacting to physics (gravity/collisions).")]
    public bool disablePhysicsOnAttach = true;

    // Helper variables
    private const string _logPrefix = "[OBJECT_ATTACHER]";

    // This function runs when a solid physics collision occurs
    private void OnCollisionEnter(Collision collision)
    {
        GameObject otherObj = collision.gameObject;
        Debug.Log(_logPrefix + " collision object: " + otherObj.name);
        // 1. Check if we hit the right kind of object
        // If targetTag is empty, we accept everything. Otherwise, we check the tag.
        if (string.IsNullOrEmpty(targetTag) || otherObj.CompareTag(targetTag))
        {
            AttachObject(otherObj);
        }
    }

    private void AttachObject(GameObject obj)
    {
        // 2. Set the parent
        // This makes the object move/rotate exactly with this object
        obj.transform.SetParent(this.transform);

        // 3. Handle Physics (Optional but Recommended)
        // If we don't do this, the attached object might weigh down the parent
        // or cause physics glitches (jittering).
        if (disablePhysicsOnAttach)
        {
            Rigidbody rb = obj.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.isKinematic = true; // Disables physics simulation
                rb.linearVelocity = Vector3.zero; // Stops existing momentum
            }
        }

        Debug.Log(_logPrefix + obj.name + " has been attached!");
    }
}
