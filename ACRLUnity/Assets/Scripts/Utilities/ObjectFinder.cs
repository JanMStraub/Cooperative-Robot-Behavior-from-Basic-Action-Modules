using System.Collections.Generic;
using Robotics;
using UnityEngine;

namespace Utilities
{
    public class ObjectFinder : MonoBehaviour
    {
        public static ObjectFinder Instance { get; private set; }

        [Header("Search Settings")]
        [SerializeField]
        private float _defaultSearchRadius = 2.0f;

        [SerializeField]
        [Tooltip("Layers to include in search. Exclude floor/walls to avoid noise")]
        private LayerMask _searchLayerMask = ~0;

        [SerializeField]
        private float _minObjectSize = 0.01f;

        [SerializeField]
        private bool _skipRobotParts = true;

        [SerializeField]
        [Tooltip("Skip objects tagged as 'Ground' or 'Floor' to avoid detecting terrain")]
        private bool _skipGroundObjects = true;

        [Header("Performance Settings")]
        [SerializeField]
        [Tooltip(
            "Enable caching for robot part detection. Disable if you spawn/destroy many objects frequently"
        )]
        private bool _enableCaching = true;

        [SerializeField]
        [Tooltip("Interval in seconds to clean up dead cache entries (0 = never)")]
        private float _cacheCleanupInterval = 60f;

        private const string _logPrefix = "[OBJECT_FINDER]";

        // Pre-allocated buffers to eliminate per-call GC allocations.
        private const int _colliderBufferSize = 512;
        private readonly Collider[] _colliderBuffer = new Collider[_colliderBufferSize];
        private readonly List<GameObject> _deadKeyBuffer = new List<GameObject>();

        private Dictionary<GameObject, bool> _robotPartCache = new Dictionary<GameObject, bool>();
        private float _lastCacheCleanupTime = 0f;

        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                _lastCacheCleanupTime = Time.time;
            }
            else
            {
                Destroy(gameObject);
            }
        }

        private void Update()
        {
            if (_enableCaching && _cacheCleanupInterval > 0)
            {
                if (Time.time - _lastCacheCleanupTime >= _cacheCleanupInterval)
                {
                    CleanupDeadCacheEntries();
                    _lastCacheCleanupTime = Time.time;
                }
            }
        }

        /// <summary>
        /// Removes destroyed objects from the cache while preserving valid entries.
        /// Automatically called periodically based on _cacheCleanupInterval.
        /// Uses a pre-allocated buffer to avoid GC allocations during cleanup.
        /// </summary>
        private void CleanupDeadCacheEntries()
        {
            _deadKeyBuffer.Clear();

            foreach (var key in _robotPartCache.Keys)
            {
                if (key == null)
                {
                    _deadKeyBuffer.Add(key);
                }
            }

            foreach (var deadKey in _deadKeyBuffer)
            {
                _robotPartCache.Remove(deadKey);
            }

            if (_deadKeyBuffer.Count > 0)
            {
                Debug.Log(
                    $"{_logPrefix} Cleaned up {_deadKeyBuffer.Count} dead cache entries. Cache size: {_robotPartCache.Count}"
                );
            }
        }

        /// <summary>
        /// Finds the closest GameObject to a position within the specified radius.
        /// Iterates directly over the NonAlloc physics buffer to avoid allocating
        /// an intermediate list.
        /// </summary>
        /// <param name="position">Reference position</param>
        /// <param name="radius">Maximum search radius in meters</param>
        /// <returns>Closest GameObject, or null if none found</returns>
        public GameObject FindClosestObject(Vector3 position, float radius)
        {
            float searchRadius = radius > 0 ? radius : _defaultSearchRadius;
            int hitCount = Physics.OverlapSphereNonAlloc(
                position,
                searchRadius,
                _colliderBuffer,
                _searchLayerMask
            );

            GameObject closest = null;
            float minSqrDistance = float.MaxValue;

            for (int i = 0; i < hitCount; i++)
            {
                Collider col = _colliderBuffer[i];
                GameObject obj = col.gameObject;

                if (_skipGroundObjects && IsGroundObject(obj))
                    continue;
                if (_skipRobotParts && IsRobotPart(obj))
                    continue;
                if (col.bounds.size.magnitude < _minObjectSize)
                    continue;

                float sqrDist = (position - obj.transform.position).sqrMagnitude;
                if (sqrDist < minSqrDistance)
                {
                    minSqrDistance = sqrDist;
                    closest = obj;
                }
            }

            return closest;
        }

        /// <summary>
        /// Checks if a GameObject is part of a robot assembly.
        /// Uses GetComponentInParent to handle child objects and caching for performance.
        /// </summary>
        /// <param name="obj">GameObject to check</param>
        /// <returns>True if the object is part of a robot</returns>
        private bool IsRobotPart(GameObject obj)
        {
            if (obj == null)
                return false;

            if (_enableCaching && _robotPartCache.TryGetValue(obj, out bool cachedResult))
            {
                return cachedResult;
            }

            bool isRobotPart = obj.GetComponentInParent<ArticulationBody>() != null;

            if (!isRobotPart)
            {
                isRobotPart =
                    obj.GetComponentInParent<RobotController>() != null
                    || obj.GetComponentInParent<GripperController>() != null
                    || obj.GetComponentInParent<SimpleRobotController>() != null;
            }

            if (_enableCaching)
            {
                _robotPartCache[obj] = isRobotPart;
            }

            return isRobotPart;
        }

        private bool IsGroundObject(GameObject obj)
        {
            return CompareSafeTag(obj, "Ground");
        }

        private static bool CompareSafeTag(GameObject obj, string tag)
        {
            try
            {
                return obj.CompareTag(tag);
            }
            catch (UnityException)
            {
                return false;
            }
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                Instance = null;
            }
        }
    }
}
