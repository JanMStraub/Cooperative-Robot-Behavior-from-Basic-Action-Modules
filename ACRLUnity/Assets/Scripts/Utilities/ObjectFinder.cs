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

        // Pre-allocated buffers to eliminate per-call GC allocations.
        private const int _colliderBufferSize = 512;
        private readonly Collider[] _colliderBuffer = new Collider[_colliderBufferSize];
        private readonly List<Collider> _colliderListBuffer = new List<Collider>();
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
        /// Finds all GameObjects near a position within the specified radius.
        /// Excludes robot parts and objects smaller than the minimum size threshold.
        /// Uses NonAlloc physics query and a pre-allocated collider buffer to avoid GC pressure.
        /// </summary>
        /// <param name="position">Center position for the search sphere</param>
        /// <param name="radius">Search radius in meters (uses default if negative)</param>
        /// <returns>List of GameObjects within the search radius</returns>
        public List<GameObject> FindObjectsNearPosition(Vector3 position, float radius = -1f)
        {
            float searchRadius = radius > 0 ? radius : _defaultSearchRadius;
            HashSet<GameObject> foundObjects = new HashSet<GameObject>();
            int hitCount = Physics.OverlapSphereNonAlloc(
                position,
                searchRadius,
                _colliderBuffer,
                _searchLayerMask
            );

            for (int i = 0; i < hitCount; i++)
            {
                Collider col = _colliderBuffer[i];
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
        /// Uses attachedRigidbody to correctly handle compound colliders where the Rigidbody
        /// lives on a parent and colliders are on children. Deduplication is keyed on the
        /// Rigidbody's GameObject to avoid duplicate entries for multi-collider objects.
        /// Uses NonAlloc physics query to avoid GC allocations.
        /// </summary>
        /// <param name="position">Center position for the search sphere</param>
        /// <param name="radius">Search radius in meters (uses default if negative)</param>
        /// <returns>List of graspable GameObjects sorted by distance</returns>
        public List<GameObject> FindGraspableObjects(Vector3 position, float radius = -1f)
        {
            float searchRadius = radius > 0 ? radius : _defaultSearchRadius;
            HashSet<GameObject> distinctObjects = new HashSet<GameObject>();
            int hitCount = Physics.OverlapSphereNonAlloc(
                position,
                searchRadius,
                _colliderBuffer,
                _searchLayerMask
            );

            for (int i = 0; i < hitCount; i++)
            {
                Collider col = _colliderBuffer[i];

                // Use attachedRigidbody so compound colliders on child objects correctly
                // resolve to the Rigidbody on the parent root.
                Rigidbody rb = col.attachedRigidbody;
                if (rb == null)
                    continue;
                if (rb.isKinematic)
                    continue;

                // Key deduplication on the Rigidbody's GameObject, not the collider's
                // GameObject, so compound colliders sharing one Rigidbody only appear once.
                GameObject obj = rb.gameObject;

                if (distinctObjects.Contains(obj))
                    continue;
                if (_skipGroundObjects && IsGroundObject(obj))
                    continue;
                if (_skipRobotParts && IsRobotPart(obj))
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

            // Pre-compute squared distances once and sort by them, so sqrMagnitude is
            // not recalculated O(N log N) times inside the comparator.
            List<GameObject> resultList = new List<GameObject>(distinctObjects);
            int count = resultList.Count;
            float[] sqrDistances = new float[count];
            for (int i = 0; i < count; i++)
            {
                sqrDistances[i] = (position - resultList[i].transform.position).sqrMagnitude;
            }

            // Sort a separate index array, then reorder resultList in one pass.
            // This avoids any O(N) lookup inside the comparator.
            int[] indices = new int[count];
            for (int i = 0; i < count; i++)
                indices[i] = i;

            System.Array.Sort(indices, (a, b) => sqrDistances[a].CompareTo(sqrDistances[b]));

            List<GameObject> sortedList = new List<GameObject>(count);
            for (int i = 0; i < count; i++)
                sortedList.Add(resultList[indices[i]]);

            return sortedList;
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
        /// Counts the number of GameObjects near a position within the specified radius.
        /// Iterates directly over the NonAlloc physics buffer to avoid allocating
        /// an intermediate collection.
        /// </summary>
        /// <param name="position">Center position for the search sphere</param>
        /// <param name="radius">Search radius in meters</param>
        /// <returns>Number of GameObjects found</returns>
        public int CountObjectsNearPosition(Vector3 position, float radius)
        {
            float searchRadius = radius > 0 ? radius : _defaultSearchRadius;
            int hitCount = Physics.OverlapSphereNonAlloc(
                position,
                searchRadius,
                _colliderBuffer,
                _searchLayerMask
            );

            // Count distinct GameObjects, applying the same filters as FindObjectsNearPosition.
            HashSet<GameObject> counted = new HashSet<GameObject>();

            for (int i = 0; i < hitCount; i++)
            {
                Collider col = _colliderBuffer[i];
                GameObject obj = col.gameObject;

                if (counted.Contains(obj))
                    continue;
                if (_skipGroundObjects && IsGroundObject(obj))
                    continue;
                if (_skipRobotParts && IsRobotPart(obj))
                    continue;
                if (col.bounds.size.magnitude < _minObjectSize)
                    continue;

                counted.Add(obj);
            }

            return counted.Count;
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
        /// Uses a pre-allocated List buffer with the list-overload of GetComponentsInChildren
        /// to avoid allocating a new Collider[] on every call.
        /// </summary>
        /// <param name="obj">GameObject to measure</param>
        /// <returns>Combined bounds of all colliders</returns>
        private Bounds GetObjectBounds(GameObject obj)
        {
            // Clear before use: the list overload appends, it does not replace.
            _colliderListBuffer.Clear();
            obj.GetComponentsInChildren<Collider>(_colliderListBuffer);

            if (_colliderListBuffer.Count == 0)
            {
                return new Bounds(obj.transform.position, Vector3.zero);
            }

            Bounds combinedBounds = _colliderListBuffer[0].bounds;

            for (int i = 1; i < _colliderListBuffer.Count; i++)
            {
                combinedBounds.Encapsulate(_colliderListBuffer[i].bounds);
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
