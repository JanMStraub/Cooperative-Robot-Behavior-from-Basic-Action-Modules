using System.Collections.Generic;
using System.Linq;
using Configuration;
using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Contact sensor for gripper fingers with force estimation.
    /// Provides reliable grasp detection through multi-criteria verification:
    /// - Contact detection (both fingers touching object)
    /// - Force estimation (moving average to handle Unity physics noise)
    /// - Closure position (gripper not fully closed or fully open)
    ///
    /// Usage:
    /// - Attach to gripper GameObject
    /// - Assign left and right finger ArticulationBodies
    /// - Call HasContact() and EstimateGraspForce() to verify grasp success
    /// </summary>
    [RequireComponent(typeof(Collider))]
    public class GripperContactSensor : MonoBehaviour
    {
        /// <summary>
        /// Enum to identify which finger a collision belongs to
        /// </summary>
        public enum FingerType
        {
            Left,
            Right,
        }

        [Header("Finger Configuration")]
        [Tooltip("Left finger ArticulationBody for force sensing")]
        public ArticulationBody leftFinger;

        [Tooltip("Right finger ArticulationBody for force sensing")]
        public ArticulationBody rightFinger;

        [Header("Configuration")]
        [SerializeField]
        [Tooltip("Gripper configuration (force thresholds, contact duration, etc.)")]
        private GripperConfig _gripperConfig;

        [Header("Contact Tracking")]
        [Tooltip("Enable debug logging for contact events")]
        public bool debugLogging = false;

        // Contact tracking per finger
        private HashSet<Collider> _leftContacts = new HashSet<Collider>();
        private HashSet<Collider> _rightContacts = new HashSet<Collider>();

        // ⚠️ CRITICAL: Moving average for force to handle Unity physics noise
        // Unity physics forces can spike to infinity on impact
        // Must use moving average over multiple frames for stable readings
        private Queue<float> _forceHistory = new Queue<float>();
        private float _currentForceSum = 0f; // Running sum for O(1) average calculation

        // Contact duration tracking (helps distinguish stable grasp from collision)
        private Dictionary<GameObject, float> _contactStartTime =
            new Dictionary<GameObject, float>();

        void Start()
        {
            // Load default config if not assigned
            if (_gripperConfig == null)
            {
                _gripperConfig = Resources.Load<GripperConfig>(
                    "Configuration/DefaultGripperConfig"
                );
            }
            if (_gripperConfig == null)
            {
                _gripperConfig = ScriptableObject.CreateInstance<GripperConfig>();
            }

            // Validate configuration
            if (leftFinger == null || rightFinger == null)
            {
                Debug.LogWarning(
                    "[GripperContactSensor] Left or right finger not assigned! Contact detection disabled."
                );
            }

            // Ensure colliders are set to trigger if needed
            var colliders = GetComponentsInChildren<Collider>();
            if (colliders.Length == 0)
            {
                Debug.LogWarning(
                    "[GripperContactSensor] No colliders found! Contact detection may not work."
                );
            }

            if (debugLogging)
            {
                Debug.Log(
                    "[GripperContactSensor] Initialized with "
                        + $"{colliders.Length} colliders, force window size={_gripperConfig.forceWindowSize}"
                );
            }
        }

        void FixedUpdate()
        {
            // Update force history every physics frame
            if (leftFinger != null && rightFinger != null)
            {
                float currentForce = CalculateInstantaneousForce();
                UpdateForceHistory(currentForce);
            }
        }

        /// <summary>
        /// Check if both fingers are in contact with the target object.
        /// Uses contact duration to filter transient collisions.
        /// </summary>
        /// <param name="targetObject">Object to check contact with</param>
        /// <returns>True if both fingers have stable contact with object</returns>
        public bool HasContact(GameObject targetObject)
        {
            if (targetObject == null)
                return false;

            // Check if both fingers are touching the target
            bool leftTouching = _leftContacts.Any(c => c != null && c.gameObject == targetObject);
            bool rightTouching = _rightContacts.Any(c => c != null && c.gameObject == targetObject);

            // Require stable contact (not just momentary collision)
            if (leftTouching && rightTouching)
            {
                // Check contact duration
                if (_contactStartTime.TryGetValue(targetObject, out float startTime))
                {
                    float duration = Time.time - startTime;
                    float minDuration =
                        _gripperConfig != null ? _gripperConfig.minContactDuration : 0.1f;
                    return duration >= minDuration;
                }
            }

            return false;
        }

        /// <summary>
        /// Estimate grasp force using moving average to handle Unity physics noise.
        /// ⚠️ CRITICAL: Unity physics forces are noisy and can spike to infinity on impact.
        /// This method averages force over multiple frames for stable readings.
        /// </summary>
        /// <returns>Average grasp force over last N frames (Newtons)</returns>
        public float EstimateGraspForce()
        {
            if (leftFinger == null || rightFinger == null)
                return 0f;

            // Return average force over window (O(1) calculation, 0 garbage)
            if (_forceHistory.Count == 0)
                return 0f;

            return _currentForceSum / _forceHistory.Count;
        }

        /// <summary>
        /// Check if grasp is stable based on multiple criteria:
        /// - Both fingers in contact
        /// - Force above minimum threshold
        /// - Contact duration sufficient
        /// </summary>
        /// <param name="targetObject">Object being grasped</param>
        /// <param name="minForce">Minimum required force (default 5N)</param>
        /// <returns>True if grasp is stable</returns>
        public bool IsGraspStable(GameObject targetObject, float minForce = 5f)
        {
            bool hasContact = HasContact(targetObject);
            float graspForce = EstimateGraspForce();

            return hasContact && graspForce >= minForce;
        }

        /// <summary>
        /// Calculate instantaneous force from both fingers.
        /// This raw value is noisy - use EstimateGraspForce() for stable readings.
        /// </summary>
        private float CalculateInstantaneousForce()
        {
            if (leftFinger == null || rightFinger == null)
                return 0f;

            // Get joint forces from ArticulationBody
            // jointForce[0] is the force along the joint's primary axis
            float leftForce =
                leftFinger.jointForce.dofCount > 0 ? Mathf.Abs(leftFinger.jointForce[0]) : 0f;
            float rightForce =
                rightFinger.jointForce.dofCount > 0 ? Mathf.Abs(rightFinger.jointForce[0]) : 0f;

            // Average force from both fingers
            float totalForce = (leftForce + rightForce) / 2f;

            // Clamp to reasonable range (prevent infinity spikes)
            return Mathf.Clamp(totalForce, 0f, 1000f);
        }

        /// <summary>
        /// Update force history with moving average window
        /// </summary>
        private void UpdateForceHistory(float instantaneousForce)
        {
            int windowSize = _gripperConfig != null ? _gripperConfig.forceWindowSize : 5;

            _forceHistory.Enqueue(instantaneousForce);
            _currentForceSum += instantaneousForce; // Add new

            if (_forceHistory.Count > windowSize)
            {
                float removed = _forceHistory.Dequeue();
                _currentForceSum -= removed; // Remove old
            }
        }

        /// <summary>
        /// Reset force history (call when starting new grasp attempt)
        /// </summary>
        public void ResetForceHistory()
        {
            _forceHistory.Clear();
            _currentForceSum = 0f;
        }

        /// <summary>
        /// Get all objects currently in contact with gripper
        /// </summary>
        public List<GameObject> GetContactedObjects()
        {
            var objects = new HashSet<GameObject>();

            foreach (var collider in _leftContacts)
            {
                if (collider != null)
                    objects.Add(collider.gameObject);
            }

            foreach (var collider in _rightContacts)
            {
                if (collider != null)
                    objects.Add(collider.gameObject);
            }

            return objects.ToList();
        }

        // Public methods for trigger forwarding (called by GripperCollisionForwarder)
        public void OnFingerTriggerEnter(Collider collider, FingerType finger)
        {
            if (debugLogging)
            {
                Debug.Log(
                    $"[GripperContactSensor] OnFingerTriggerEnter received: {finger} finger, object: {collider.gameObject.name}"
                );
            }
            TrackContact(collider, finger, true);
        }

        public void OnFingerTriggerStay(Collider collider, FingerType finger)
        {
            TrackContact(collider, finger, true);
        }

        public void OnFingerTriggerExit(Collider collider, FingerType finger)
        {
            if (debugLogging)
            {
                Debug.Log(
                    $"[GripperContactSensor] OnFingerTriggerExit received: {finger} finger, object: {collider.gameObject.name}"
                );
            }
            TrackContact(collider, finger, false);
        }

        // Unity trigger callbacks (legacy support - for when sensor itself has trigger)
        void OnTriggerEnter(Collider collider)
        {
            // Ignore self-collisions with own fingers
            if (IsFingerCollider(collider))
                return;

            TrackContact(collider, true);
        }

        void OnTriggerStay(Collider collider)
        {
            // Ignore self-collisions with own fingers
            if (IsFingerCollider(collider))
                return;

            TrackContact(collider, true);
        }

        void OnTriggerExit(Collider collider)
        {
            // Ignore self-collisions with own fingers
            if (IsFingerCollider(collider))
                return;

            TrackContact(collider, false);
        }

        /// <summary>
        /// Check if a collider belongs to one of this gripper's fingers.
        /// Used to filter self-collisions in legacy trigger callbacks.
        /// </summary>
        private bool IsFingerCollider(Collider collider)
        {
            if (collider == null)
                return false;

            // Check if collider is on the left or right finger GameObject
            if (leftFinger != null && collider.gameObject == leftFinger.gameObject)
                return true;

            if (rightFinger != null && collider.gameObject == rightFinger.gameObject)
                return true;

            return false;
        }

        /// <summary>
        /// Track contact for a specific finger (called by forwarder)
        /// </summary>
        private void TrackContact(Collider collider, FingerType finger, bool isContact)
        {
            if (collider == null)
                return;

            bool isLeftFinger = (finger == FingerType.Left);
            bool isRightFinger = (finger == FingerType.Right);

            GameObject obj = collider.gameObject;

            if (isContact)
            {
                // Add contact
                if (isLeftFinger)
                {
                    _leftContacts.Add(collider);
                }
                else if (isRightFinger)
                {
                    _rightContacts.Add(collider);
                }

                // Track contact start time (set when BOTH fingers touching)
                // Check if both fingers are now touching after adding this contact
                bool bothTouching =
                    _leftContacts.Any(c => c?.gameObject == obj)
                    && _rightContacts.Any(c => c?.gameObject == obj);

                if (bothTouching && !_contactStartTime.ContainsKey(obj))
                {
                    // Both fingers touching - start timing
                    _contactStartTime[obj] = Time.time;

                    if (debugLogging)
                    {
                        Debug.Log(
                            $"[GripperContactSensor] Contact START (both fingers): {obj.name}"
                        );
                    }
                }
            }
            else
            {
                // Remove contact
                if (isLeftFinger)
                {
                    _leftContacts.Remove(collider);
                }
                else if (isRightFinger)
                {
                    _rightContacts.Remove(collider);
                }

                // Check if object is still in contact with either finger after removal
                bool stillInContact =
                    _leftContacts.Any(c => c?.gameObject == obj)
                    || _rightContacts.Any(c => c?.gameObject == obj);

                // Remove timing if no longer in contact with either finger
                // OR if no longer touching with BOTH fingers (reset timer)
                if (!stillInContact)
                {
                    if (_contactStartTime.ContainsKey(obj))
                    {
                        _contactStartTime.Remove(obj);

                        if (debugLogging)
                        {
                            Debug.Log($"[GripperContactSensor] Contact END: {obj.name}");
                        }
                    }
                }
                else
                {
                    // Still touching with one finger but not both - reset timer
                    bool bothTouching =
                        _leftContacts.Any(c => c?.gameObject == obj)
                        && _rightContacts.Any(c => c?.gameObject == obj);

                    if (!bothTouching && _contactStartTime.ContainsKey(obj))
                    {
                        _contactStartTime.Remove(obj);

                        if (debugLogging)
                        {
                            Debug.Log(
                                $"[GripperContactSensor] Contact timer RESET (only one finger): {obj.name}"
                            );
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Track contact per finger based on collider hierarchy (legacy method)
        /// </summary>
        private void TrackContact(Collider collider, bool isContact)
        {
            if (collider == null)
                return;

            // Determine which finger this collider belongs to
            bool isLeftFinger = IsChildOf(collider.transform, leftFinger?.transform);
            bool isRightFinger = IsChildOf(collider.transform, rightFinger?.transform);

            if (isContact)
            {
                // Add contact
                if (isLeftFinger)
                {
                    _leftContacts.Add(collider);
                }
                else if (isRightFinger)
                {
                    _rightContacts.Add(collider);
                }

                // Track contact start time
                GameObject obj = collider.gameObject;
                if (!_contactStartTime.ContainsKey(obj))
                {
                    _contactStartTime[obj] = Time.time;

                    if (debugLogging)
                    {
                        Debug.Log(
                            $"[GripperContactSensor] Contact START: {obj.name} "
                                + $"(finger: {(isLeftFinger ? "LEFT" : "RIGHT")})"
                        );
                    }
                }
            }
            else
            {
                // Remove contact
                if (isLeftFinger)
                {
                    _leftContacts.Remove(collider);
                }
                else if (isRightFinger)
                {
                    _rightContacts.Remove(collider);
                }

                // Remove from timing if no longer in contact with either finger
                GameObject obj = collider.gameObject;
                bool stillInContact =
                    _leftContacts.Any(c => c?.gameObject == obj)
                    || _rightContacts.Any(c => c?.gameObject == obj);

                if (!stillInContact && _contactStartTime.ContainsKey(obj))
                {
                    _contactStartTime.Remove(obj);

                    if (debugLogging)
                    {
                        Debug.Log($"[GripperContactSensor] Contact END: {obj.name}");
                    }
                }
            }
        }

        /// <summary>
        /// Check if a transform is a child of a parent transform
        /// </summary>
        private bool IsChildOf(Transform child, Transform parent)
        {
            if (child == null || parent == null)
                return false;

            Transform current = child;
            while (current != null)
            {
                if (current == parent)
                    return true;
                current = current.parent;
            }
            return false;
        }

        /// <summary>
        /// Cleanup on destroy
        /// </summary>
        void OnDestroy()
        {
            _leftContacts.Clear();
            _rightContacts.Clear();
            _contactStartTime.Clear();
            _forceHistory.Clear();
        }

        /// <summary>
        /// Visualize contact points in scene view
        /// </summary>
        void OnDrawGizmos()
        {
            if (!debugLogging)
                return;

            // Draw contact indicators
            foreach (var collider in _leftContacts)
            {
                if (collider != null)
                {
                    Gizmos.color = Color.green;
                    Gizmos.DrawWireSphere(collider.transform.position, 0.01f);
                }
            }

            foreach (var collider in _rightContacts)
            {
                if (collider != null)
                {
                    Gizmos.color = Color.blue;
                    Gizmos.DrawWireSphere(collider.transform.position, 0.01f);
                }
            }
        }
    }
}
