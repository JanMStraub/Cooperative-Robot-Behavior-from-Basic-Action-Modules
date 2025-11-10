using System;
using System.Collections.Generic;
using Core;
using Robotics;
using UnityEngine;

namespace Utilities
{
    /// <summary>
    /// Centralized service for registering and tracking scene objects.
    /// Eliminates duplication between MainLogger and AutoLogger.
    /// Provides filtering logic for robot parts, size thresholds, and material detection.
    /// </summary>
    public class ObjectRegistry : MonoBehaviour
    {
        public static ObjectRegistry Instance { get; private set; }

        private HashSet<GameObject> _registeredObjects = new HashSet<GameObject>();
        private Dictionary<GameObject, ObjectInfo> _objectInfo =
            new Dictionary<GameObject, ObjectInfo>();

        // Helper variables
        private const string _logPrefix = "[OBJECT_REGISTRY]";

        /// <summary>
        /// Event fired when an object is registered
        /// </summary>
        public event Action<GameObject, ObjectInfo> OnObjectRegistered;

        /// <summary>
        /// Information about a registered object
        /// </summary>
        public class ObjectInfo
        {
            public string ObjectType { get; set; }
            public bool IsGraspable { get; set; }
            public Vector3 InitialPosition { get; set; }
            public Quaternion InitialRotation { get; set; }
            public float Size { get; set; }
        }

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
        /// Registers a single GameObject with optional type and graspability
        /// </summary>
        /// <param name="gameObject">The GameObject to register</param>
        /// <param name="objectType">Optional object type identifier</param>
        /// <param name="isGraspable">Whether the object can be grasped</param>
        public void RegisterObject(
            GameObject gameObject,
            string objectType = null,
            bool isGraspable = false
        )
        {
            if (gameObject == null || _registeredObjects.Contains(gameObject))
                return;

            var info = new ObjectInfo
            {
                ObjectType = objectType ?? gameObject.name,
                IsGraspable = isGraspable,
                InitialPosition = gameObject.transform.position,
                InitialRotation = gameObject.transform.rotation,
                Size = GetObjectSize(gameObject),
            };

            _registeredObjects.Add(gameObject);
            _objectInfo[gameObject] = info;

            OnObjectRegistered?.Invoke(gameObject, info);
        }

        /// <summary>
        /// Automatically registers all scene objects based on filters.
        /// Finds objects with colliders and Trackpoint materials.
        /// </summary>
        /// <param name="includeColliders">Include objects with colliders</param>
        /// <param name="includeTrackpoints">Include objects with Trackpoint material</param>
        /// <returns>Number of objects registered</returns>
        public int RegisterSceneObjects(
            bool includeColliders = true,
            bool includeTrackpoints = true
        )
        {
            var newlyRegistered = new HashSet<GameObject>();

            if (includeColliders)
            {
                RegisterCollidersAsObjects(newlyRegistered);
            }

            if (includeTrackpoints)
            {
                RegisterTrackpointObjects(newlyRegistered);
            }

            Debug.Log(
                $"{_logPrefix} Registered {newlyRegistered.Count} new objects (total: {_registeredObjects.Count})"
            );
            return newlyRegistered.Count;
        }

        /// <summary>
        /// Registers objects with colliders (potential targets)
        /// </summary>
        private void RegisterCollidersAsObjects(HashSet<GameObject> newlyRegistered)
        {
            var colliders = FindObjectsByType<Collider>(FindObjectsSortMode.None);

            foreach (var collider in colliders)
            {
                var obj = collider.gameObject;

                // Skip if already registered
                if (_registeredObjects.Contains(obj))
                    continue;

                // Skip robot parts
                if (IsRobotPart(obj))
                    continue;

                // Skip too small objects
                if (collider.bounds.size.magnitude < SceneConstants.SMALL_OBJECT_SIZE_THRESHOLD)
                    continue;

                // Register object
                bool isGraspable =
                    obj.GetComponent<Rigidbody>() != null
                    && collider.bounds.size.magnitude
                        < SceneConstants.GRASPABLE_OBJECT_SIZE_THRESHOLD;

                RegisterObject(obj, null, isGraspable);
                newlyRegistered.Add(obj);
            }
        }

        /// <summary>
        /// Registers objects with Trackpoint materials
        /// </summary>
        private void RegisterTrackpointObjects(HashSet<GameObject> newlyRegistered)
        {
            var renderers = FindObjectsByType<Renderer>(FindObjectsSortMode.None);

            foreach (var renderer in renderers)
            {
                var obj = renderer.gameObject;

                // Skip if already registered
                if (_registeredObjects.Contains(obj))
                    continue;

                // Skip robot parts
                if (IsRobotPart(obj))
                    continue;

                // Check if any material is named "Trackpoint"
                bool hasTrackpointMaterial = false;
                foreach (var material in renderer.sharedMaterials)
                {
                    if (material != null && material.name.Contains("Trackpoint"))
                    {
                        hasTrackpointMaterial = true;
                        break;
                    }
                }

                if (hasTrackpointMaterial)
                {
                    // Trackpoint objects are typically not graspable (markers)
                    RegisterObject(obj, null, false);
                    newlyRegistered.Add(obj);
                }
            }
        }

        /// <summary>
        /// Checks if a GameObject is a robot part
        /// </summary>
        private bool IsRobotPart(GameObject obj)
        {
            return obj.GetComponent<RobotController>() != null
                || obj.GetComponent<ArticulationBody>() != null
                || obj.GetComponent<GripperController>() != null;
        }

        /// <summary>
        /// Gets the size magnitude of an object based on collider or renderer bounds
        /// </summary>
        private float GetObjectSize(GameObject obj)
        {
            var collider = obj.GetComponent<Collider>();
            if (collider != null)
                return collider.bounds.size.magnitude;

            var renderer = obj.GetComponent<Renderer>();
            if (renderer != null)
                return renderer.bounds.size.magnitude;

            return 0f;
        }

        /// <summary>
        /// Gets all registered objects
        /// </summary>
        public IEnumerable<GameObject> GetRegisteredObjects()
        {
            return _registeredObjects;
        }

        /// <summary>
        /// Gets information about a registered object
        /// </summary>
        public ObjectInfo GetObjectInfo(GameObject obj)
        {
            return _objectInfo.TryGetValue(obj, out var info) ? info : null;
        }

        /// <summary>
        /// Checks if an object is registered
        /// </summary>
        public bool IsRegistered(GameObject obj)
        {
            return _registeredObjects.Contains(obj);
        }

        /// <summary>
        /// Unregisters an object
        /// </summary>
        public void UnregisterObject(GameObject obj)
        {
            _registeredObjects.Remove(obj);
            _objectInfo.Remove(obj);
        }

        /// <summary>
        /// Clears all registered objects
        /// </summary>
        public void ClearAll()
        {
            _registeredObjects.Clear();
            _objectInfo.Clear();
            Debug.Log($"{_logPrefix} Cleared all registered objects");
        }

        /// <summary>
        /// Gets the count of registered objects
        /// </summary>
        public int Count => _registeredObjects.Count;

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
    }
}
