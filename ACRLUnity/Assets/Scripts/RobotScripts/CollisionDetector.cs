using System;
using System.Collections.Generic;
using System.Linq;
using Core;
using UnityEngine;

namespace Robotics
{
    [System.Serializable]
    public class CollisionData
    {
        public string timestamp;
        public string robotId;
        public string targetId;
        public Vector3 collisionPoint;
        public float approachSpeed;
        public string collisionType;
        public bool wasIntended;
    }

    [System.Serializable]
    public class CollisionConfig
    {
        [Header("Collision Behavior")]
        public bool enableCollisionDetection = true;
        public bool enableTargetReached = true;
        public float collisionCooldown = CollisionConstants.DEFAULT_COLLISION_COOLDOWN;

        [Header("Filtering")]
        public LayerMask robotLayerMask = -1;
        public string[] ignoredTags = { "Untagged" };
    }

    public class CollisionDetector : MonoBehaviour
    {
        [Header("Configuration")]
        [SerializeField]
        private CollisionConfig config = new CollisionConfig();

        [Header("Target Information")]
        [SerializeField]
        private string targetId;

        [SerializeField]
        private bool isGoalTarget = true;

        [SerializeField]
        private float targetRewardValue = CollisionConstants.DEFAULT_TARGET_REWARD;

        // Core components
        private RobotManager _robotManager;

        // Collision tracking
        private Dictionary<string, float> _lastCollisionTime = new Dictionary<string, float>();
        private int _totalCollisions;

        // Properties
        public CollisionConfig Config => config;
        public int TotalCollisions => _totalCollisions;

        // Helper variables
        private const string _logPrefix = "[COLLISION_DETECTOR]";

        /// <summary>
        /// Unity Awake callback - auto-detects target ID if not set.
        /// </summary>
        private void Awake()
        {
            if (string.IsNullOrEmpty(targetId))
            {
                targetId = gameObject.name;
            }
        }

        /// <summary>
        /// Unity Start callback - initializes component references and validates configuration.
        /// </summary>
        private void Start()
        {
            try
            {
                _robotManager = RobotManager.Instance;

                ValidateConfiguration();

                Debug.Log(
                    $"{_logPrefix} Initialized for target: {targetId}, GoalTarget: {isGoalTarget}, RewardValue: {targetRewardValue}"
                );
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Failed to initialize GetCollision for {targetId}: {ex.Message}"
                );
            }
        }

        /// <summary>
        /// Validates the collision configuration and corrects invalid values.
        /// </summary>
        private void ValidateConfiguration()
        {
            if (config.collisionCooldown < 0)
            {
                Debug.LogWarning(
                    $"{_logPrefix} {targetId}: Collision cooldown cannot be negative. Setting to 0."
                );
                config.collisionCooldown = 0;
            }

            if (targetRewardValue <= 0)
            {
                Debug.LogWarning(
                    $"{_logPrefix} {targetId}: Target reward value should be positive."
                );
            }
        }

        /// <summary>
        /// Unity OnTriggerEnter callback - processes collision when collider enters trigger.
        /// </summary>
        /// <param name="other">The collider that entered the trigger</param>
        private void OnTriggerEnter(Collider other)
        {
            if (!config.enableCollisionDetection)
                return;

            try
            {
                ProcessCollision(other, "trigger_enter");
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error processing collision for {targetId}: {ex.Message}"
                );
            }
        }

        /// <summary>
        /// Unity OnTriggerStay callback - processes collision while collider remains in trigger.
        /// </summary>
        /// <param name="other">The collider that is inside the trigger</param>
        private void OnTriggerStay(Collider other)
        {
            if (!config.enableCollisionDetection)
                return;

            try
            {
                ProcessCollision(other, "trigger_stay");
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error processing trigger stay for {targetId}: {ex.Message}"
                );
            }
        }

        /// <summary>
        /// Processes a collision event by filtering, cooldown checking, and handling robot collision.
        /// </summary>
        /// <param name="other">The collider that triggered the collision</param>
        /// <param name="collisionType">The type of collision (trigger_enter, trigger_stay)</param>
        private void ProcessCollision(Collider other, string collisionType)
        {
            if ((config.robotLayerMask.value & (1 << other.gameObject.layer)) == 0)
                return;

            if (config.ignoredTags.Contains(other.tag))
                return;

            var robotController = other.GetComponent<RobotController>();
            if (robotController == null)
                return;

            string robotId = robotController.gameObject.name;

            if (IsInCooldown(robotId))
                return;

            _lastCollisionTime[robotId] = Time.time;

            HandleRobotCollision(other, robotController, robotId, collisionType);
        }

        /// <summary>
        /// Checks if a robot is within the collision cooldown period.
        /// </summary>
        /// <param name="robotId">The robot identifier to check</param>
        /// <returns>True if in cooldown period, false otherwise</returns>
        private bool IsInCooldown(string robotId)
        {
            return _lastCollisionTime.ContainsKey(robotId)
                && Time.time - _lastCollisionTime[robotId] < config.collisionCooldown;
        }

        /// <summary>
        /// Handles a robot collision by calculating collision details and logging the event.
        /// </summary>
        /// <param name="other">The collider that triggered the collision</param>
        /// <param name="robotController">The robot controller component</param>
        /// <param name="robotId">The robot identifier</param>
        /// <param name="collisionType">The type of collision event</param>
        private void HandleRobotCollision(
            Collider other,
            RobotController robotController,
            string robotId,
            string collisionType
        )
        {
            try
            {
                Collider triggerCollider = GetComponent<Collider>();
                Vector3 closestPoint =
                    triggerCollider != null
                        ? triggerCollider.ClosestPoint(other.transform.position)
                        : transform.position;
                float approachSpeed = CalculateApproachSpeed(other);

                if (config.enableTargetReached && robotController != null)
                {
                    robotController.SetTargetReached(true);
                }

                var collisionData = new CollisionData
                {
                    timestamp = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                    robotId = robotId,
                    targetId = targetId,
                    collisionPoint = closestPoint,
                    approachSpeed = approachSpeed,
                    collisionType = collisionType,
                    wasIntended = isGoalTarget,
                };

                UpdateCollisionMetrics(robotId, collisionData);

                Debug.Log(
                    $"{_logPrefix} Collision detected: {robotId} -> {targetId} ({collisionType}, Speed: {approachSpeed:F2})"
                );
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error handling robot collision for {robotId}: {ex.Message}"
                );
            }
        }

        /// <summary>
        /// Calculates the approach speed of a colliding object.
        /// </summary>
        /// <param name="other">The collider to calculate speed for</param>
        /// <returns>The magnitude of the linear velocity, or 0 if no rigidbody</returns>
        private float CalculateApproachSpeed(Collider other)
        {
            try
            {
                var rigidbody = other.attachedRigidbody;
                if (rigidbody != null)
                {
                    return rigidbody.linearVelocity.magnitude;
                }
                return 0f;
            }
            catch
            {
                return 0f;
            }
        }

        /// <summary>
        /// Updates collision metrics by incrementing the total collision count.
        /// </summary>
        /// <param name="robotId">The robot identifier</param>
        /// <param name="collisionData">The collision data</param>
        private void UpdateCollisionMetrics(string robotId, CollisionData collisionData)
        {
            _totalCollisions++;
        }

        /// <summary>
        /// Resets all collision metrics and tracking data.
        /// </summary>
        public void ResetMetrics()
        {
            _totalCollisions = 0;
            _lastCollisionTime.Clear();

            Debug.Log($"{_logPrefix}  Metrics reset for target: {targetId}");
        }

        /// <summary>
        /// Sets the target reward value for this collision target.
        /// </summary>
        /// <param name="rewardValue">The new reward value</param>
        public void SetTargetReward(float rewardValue)
        {
            targetRewardValue = rewardValue;
            Debug.Log($"{_logPrefix}  Target reward changed: {targetId}, NewReward: {rewardValue}");
        }

        /// <summary>
        /// Unity OnDrawGizmos callback - visualizes the collision target in the scene view.
        /// </summary>
        private void OnDrawGizmos()
        {
#if UNITY_EDITOR
            if (_totalCollisions > 0)
            {
                var style = new GUIStyle();
                style.normal.textColor = Color.red;
                style.fontSize = 12;
                UnityEditor.Handles.Label(
                    transform.position + Vector3.up * 0.2f,
                    $"Collisions: {_totalCollisions}",
                    style
                );
            }
#endif
        }

        /// <summary>
        /// Unity OnDestroy callback - logs final collision statistics.
        /// </summary>
        private void OnDestroy()
        {
            try
            {
                if (_totalCollisions > 0)
                {
                    Debug.Log(
                        $"{_logPrefix}  Detector destroyed for target {targetId}, final stats: {_totalCollisions} collisions"
                    );
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error during GetCollision cleanup: {ex.Message}");
            }
        }
    }
}
