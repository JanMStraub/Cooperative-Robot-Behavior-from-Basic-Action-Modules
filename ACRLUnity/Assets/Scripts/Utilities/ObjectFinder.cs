using System.Collections.Generic;
using Core;
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

        [SerializeField]
        [Tooltip(
            "Maximum mass for graspable objects (kg). Objects heavier than this are too heavy to grasp"
        )]
        private float _maxGraspableMass = 5.0f;

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
        /// Clears the robot part detection cache completely.
        /// Call this when scene changes or when many robots are spawned/destroyed at once.
        /// </summary>
        public void ClearCache()
        {
            _robotPartCache.Clear();
        }

        /// <summary>
        /// Removes destroyed objects from the cache while preserving valid entries.
        /// Automatically called periodically based on _cacheCleanupInterval.
        /// </summary>
        private void CleanupDeadCacheEntries()
        {
            List<GameObject> deadKeys = new List<GameObject>();

            foreach (var key in _robotPartCache.Keys)
            {
                if (key == null)
                {
                    deadKeys.Add(key);
                }
            }

            foreach (var deadKey in deadKeys)
            {
                _robotPartCache.Remove(deadKey);
            }

            if (deadKeys.Count > 0)
            {
                Debug.Log(
                    $"{_logPrefix} Cleaned up {deadKeys.Count} dead cache entries. Cache size: {_robotPartCache.Count}"
                );
            }
        }

        /// <summary>
        /// Finds all GameObjects near a position within the specified radius.
        /// Excludes robot parts and objects smaller than the minimum size threshold.
        /// </summary>
        /// <param name="position">Center position for the search sphere</param>
        /// <param name="radius">Search radius in meters (uses default if negative)</param>
        /// <returns>List of GameObjects within the search radius</returns>
        public List<GameObject> FindObjectsNearPosition(Vector3 position, float radius = -1f)
        {
            float searchRadius = radius > 0 ? radius : _defaultSearchRadius;
            HashSet<GameObject> foundObjects = new HashSet<GameObject>();
            Collider[] colliders = Physics.OverlapSphere(position, searchRadius, _searchLayerMask);

            foreach (Collider col in colliders)
            {
                GameObject obj = col.gameObject;

                if (foundObjects.Contains(obj))
                    continue;
                if (_skipGroundObjects && IsGroundObject(obj))
                    continue;
                if (_skipRobotParts && IsRobotPart(obj))
                    continue;
                if (col.bounds.size.magnitude < _minObjectSize)
                    continue;

                foundObjects.Add(obj);
            }

            return new List<GameObject>(foundObjects);
        }

        /// <summary>
        /// Finds all graspable objects near a position, sorted by distance.
        /// Only returns non-kinematic Rigidbody objects that meet graspability criteria.
        /// Validates the root object's properties to avoid false positives from child colliders.
        /// </summary>
        /// <param name="position">Center position for the search sphere</param>
        /// <param name="radius">Search radius in meters (uses default if negative)</param>
        /// <returns>List of graspable GameObjects sorted by distance</returns>
        public List<GameObject> FindGraspableObjects(Vector3 position, float radius = -1f)
        {
            float searchRadius = radius > 0 ? radius : _defaultSearchRadius;
            HashSet<GameObject> distinctObjects = new HashSet<GameObject>();
            Collider[] colliders = Physics.OverlapSphere(position, searchRadius, _searchLayerMask);

            foreach (Collider col in colliders)
            {
                GameObject obj = col.gameObject;

                if (distinctObjects.Contains(obj))
                    continue;
                if (_skipGroundObjects && IsGroundObject(obj))
                    continue;
                if (_skipRobotParts && IsRobotPart(obj))
                    continue;

                if (!obj.TryGetComponent(out Rigidbody rb))
                    continue;
                if (rb.isKinematic)
                    continue;

                Bounds objectBounds = GetObjectBounds(obj);
                float objectSize = objectBounds.size.magnitude;

                if (objectSize < _minObjectSize)
                    continue;
                if (objectSize > SceneConstants.GRASPABLE_OBJECT_SIZE_THRESHOLD)
                    continue;
                if (rb.mass > _maxGraspableMass)
                    continue;

                distinctObjects.Add(obj);
            }

            List<GameObject> resultList = new List<GameObject>(distinctObjects);
            resultList.Sort(
                (a, b) =>
                {
                    float distA = (position - a.transform.position).sqrMagnitude;
                    float distB = (position - b.transform.position).sqrMagnitude;
                    return distA.CompareTo(distB);
                }
            );

            return resultList;
        }

        /// <summary>
        /// Finds the closest GameObject to a position within the specified radius.
        /// Uses squared distance for performance optimization.
        /// </summary>
        /// <param name="position">Reference position</param>
        /// <param name="radius">Maximum search radius in meters</param>
        /// <returns>Closest GameObject, or null if none found</returns>
        public GameObject FindClosestObject(Vector3 position, float radius)
        {
            List<GameObject> objects = FindObjectsNearPosition(position, radius);

            if (objects.Count == 0)
                return null;

            GameObject closest = null;
            float minSqrDistance = float.MaxValue;

            foreach (GameObject obj in objects)
            {
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
        /// Counts the number of GameObjects near a position within the specified radius.
        /// </summary>
        /// <param name="position">Center position for the search sphere</param>
        /// <param name="radius">Search radius in meters</param>
        /// <returns>Number of GameObjects found</returns>
        public int CountObjectsNearPosition(Vector3 position, float radius)
        {
            return FindObjectsNearPosition(position, radius).Count;
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
                if (obj == null)
                {
                    _robotPartCache.Remove(obj);
                }
                else
                {
                    return cachedResult;
                }
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

        /// <summary>
        /// Checks if a GameObject is part of the ground/floor/environment.
        /// </summary>
        /// <param name="obj">GameObject to check</param>
        /// <returns>True if the object is ground/floor</returns>
        private bool IsGroundObject(GameObject obj)
        {
            return obj.CompareTag("Ground")
                || obj.CompareTag("Floor")
                || obj.CompareTag("Terrain")
                || obj.CompareTag("Environment");
        }

        /// <summary>
        /// Calculates the combined bounds of all colliders on a GameObject and its children.
        /// </summary>
        /// <param name="obj">GameObject to measure</param>
        /// <returns>Combined bounds of all colliders</returns>
        private Bounds GetObjectBounds(GameObject obj)
        {
            Collider[] allColliders = obj.GetComponentsInChildren<Collider>();

            if (allColliders.Length == 0)
            {
                return new Bounds(obj.transform.position, Vector3.zero);
            }

            Bounds combinedBounds = allColliders[0].bounds;

            for (int i = 1; i < allColliders.Length; i++)
            {
                combinedBounds.Encapsulate(allColliders[i].bounds);
            }

            return combinedBounds;
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
