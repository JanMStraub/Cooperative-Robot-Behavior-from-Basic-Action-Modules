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

        // Helper variables
        private const string _logPrefix = "[COLLISION_DETECTOR]";

        private void Awake()
        {
            if (string.IsNullOrEmpty(targetId))
            {
                targetId = gameObject.name;
            }
        }

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

        private bool IsInCooldown(string robotId)
        {
            return _lastCollisionTime.ContainsKey(robotId)
                && Time.time - _lastCollisionTime[robotId] < config.collisionCooldown;
        }

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

        private void UpdateCollisionMetrics(string robotId, CollisionData collisionData)
        {
            _totalCollisions++;
        }

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
