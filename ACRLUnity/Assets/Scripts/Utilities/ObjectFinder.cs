using System.Collections.Generic;
using Core;
using Robotics;
using UnityEngine;

namespace Utilities
{
    /// <summary>
    /// Utility class for finding GameObjects within spatial proximity of a target position.
    /// Uses Physics.OverlapSphere for efficient collision-based queries.
    /// </summary>
    public class ObjectFinder : MonoBehaviour
    {
        public static ObjectFinder Instance { get; private set; }

        [Header("Search Settings")]
        [SerializeField]
        [Tooltip("Default search radius in meters")]
        private float _defaultSearchRadius = 2.0f;

        [SerializeField]
        [Tooltip("Layer mask for filtering searchable objects")]
        private LayerMask _searchLayerMask = ~0; // All layers by default

        [SerializeField]
        [Tooltip("Skip objects smaller than this threshold")]
        private float _minObjectSize = 0.01f;

        [SerializeField]
        [Tooltip("Skip robot parts in search results")]
        private bool _skipRobotParts = true;

        private const string _logPrefix = "[OBJECT_FINDER]";

        /// <summary>
        /// Unity Awake callback - implements singleton pattern
        /// </summary>
        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
            }
            else
            {
                Destroy(gameObject);
            }
        }

        /// <summary>
        /// Finds all GameObjects within a default radius of the target position
        /// </summary>
        /// <param name="position">Center position for search</param>
        /// <param name="radius">Search radius in meters</param>
        /// <returns>List of GameObjects found within radius</returns>
        public List<GameObject> FindObjectsNearPosition(Vector3 position)
        {
            List<GameObject> foundObjects = new List<GameObject>();

            // Find all colliders within radius using configured layer mask
            Collider[] colliders = Physics.OverlapSphere(
                position,
                _defaultSearchRadius,
                _searchLayerMask
            );

            Debug.Log(
                $"{_logPrefix} Found {colliders.Length} colliders within {_defaultSearchRadius}m of {position}"
            );

            foreach (Collider col in colliders)
            {
                GameObject obj = col.gameObject;

                // Skip robot parts if configured
                if (_skipRobotParts && IsRobotPart(obj))
                {
                    continue;
                }

                // Skip objects that are too small
                if (col.bounds.size.magnitude < _minObjectSize)
                {
                    continue;
                }

                foundObjects.Add(obj);
            }

            Debug.Log($"{_logPrefix} Filtered to {foundObjects.Count} valid objects");
            return foundObjects;
        }

        /// <summary>
        /// Finds graspable objects within the default radius of the target position
        /// </summary>
        /// <param name="position">Center position for search</param>
        /// <param name="radius">Search radius in meters</param>
        /// <returns>List of graspable GameObjects found within radius</returns>
        public List<GameObject> FindGraspableObjects(Vector3 position)
        {
            List<GameObject> foundObjects = new List<GameObject>();

            Collider[] colliders = Physics.OverlapSphere(
                position,
                _defaultSearchRadius,
                _searchLayerMask
            );

            foreach (Collider col in colliders)
            {
                GameObject obj = col.gameObject;

                // Skip robot parts
                if (_skipRobotParts && IsRobotPart(obj))
                {
                    continue;
                }

                // Check if object is graspable (has Rigidbody and is not kinematic)
                Rigidbody rb = obj.GetComponent<Rigidbody>();
                if (rb == null || rb.isKinematic)
                {
                    continue;
                }

                // Check size constraints
                if (
                    col.bounds.size.magnitude < _minObjectSize
                    || col.bounds.size.magnitude > SceneConstants.GRASPABLE_OBJECT_SIZE_THRESHOLD
                )
                {
                    continue;
                }

                foundObjects.Add(obj);
            }

            // Sort objects by distance from search position
            foundObjects.Sort((a, b) =>
            {
                float distanceA = Vector3.Distance(position, a.transform.position);
                float distanceB = Vector3.Distance(position, b.transform.position);
                return distanceA.CompareTo(distanceB);
            });

            Debug.Log($"{_logPrefix} Found {foundObjects.Count} graspable objects (sorted by distance)");
            return foundObjects;
        }

        /// <summary>
        /// Finds the closest GameObject to the target position within a radius
        /// </summary>
        /// <param name="position">Center position for search</param>
        /// <param name="radius">Search radius in meters</param>
        /// <returns>Closest GameObject or null if none found</returns>
        public GameObject FindClosestObject(Vector3 position, float radius)
        {
            List<GameObject> objects = FindObjectsNearPosition(position);

            if (objects.Count == 0)
            {
                return null;
            }

            GameObject closest = null;
            float minDistance = float.MaxValue;

            foreach (GameObject obj in objects)
            {
                float distance = Vector3.Distance(position, obj.transform.position);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    closest = obj;
                }
            }

            Debug.Log($"{_logPrefix} Closest object: {closest.name} at {minDistance}m distance");
            return closest;
        }

        /// <summary>
        /// Counts objects within a radius of the target position
        /// </summary>
        /// <param name="position">Center position for search</param>
        /// <param name="radius">Search radius in meters</param>
        /// <returns>Number of objects found</returns>
        public int CountObjectsNearPosition(Vector3 position, float radius)
        {
            return FindObjectsNearPosition(position).Count;
        }

        /// <summary>
        /// Checks if a GameObject is a robot part
        /// </summary>
        /// <param name="obj">GameObject to check</param>
        /// <returns>True if object is part of a robot</returns>
        private bool IsRobotPart(GameObject obj)
        {
            return obj.GetComponent<RobotController>() != null
                || obj.GetComponent<ArticulationBody>() != null
                || obj.GetComponent<GripperController>() != null;
        }

        /// <summary>
        /// Unity OnDestroy callback - cleanup singleton
        /// </summary>
        private void OnDestroy()
        {
            if (Instance == this)
            {
                Instance = null;
            }
        }

        // Public getters/setters for configuration
        public float DefaultSearchRadius
        {
            get => _defaultSearchRadius;
            set => _defaultSearchRadius = value;
        }

        public LayerMask SearchLayerMask
        {
            get => _searchLayerMask;
            set => _searchLayerMask = value;
        }

        public float MinObjectSize
        {
            get => _minObjectSize;
            set => _minObjectSize = value;
        }

        public bool SkipRobotParts
        {
            get => _skipRobotParts;
            set => _skipRobotParts = value;
        }
    }
}
