using System;
using System.Collections.Generic;
using UnityEngine;

namespace Vision
{
    /// <summary>
    /// Singleton manager for registering and retrieving cameras throughout the application.
    /// Provides a centralized registry for camera-related scripts to access cameras by ID.
    /// </summary>
    public class CameraManager : MonoBehaviour
    {
        // Singleton instance
        public static CameraManager Instance { get; private set; }

        // Events
        public event Action<string, Camera> OnCameraRegistered;
        public event Action<string> OnCameraUnregistered;

        private Dictionary<string, Camera> _cameras = new Dictionary<string, Camera>();

        // Helper variables
        private const string _logPrefix = "[CAMERA_MANAGER]";

        private void Awake()
        {
            // Singleton pattern
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                Debug.Log($"{_logPrefix} Initializing Camera Manager");
            }
            else
            {
                Destroy(gameObject);
            }
        }

        /// <summary>
        /// Registers a camera with a unique identifier
        /// </summary>
        /// <param name="cameraId">Unique identifier for the camera</param>
        /// <param name="camera">Camera component to register</param>
        /// <param name="silent">If true, suppress log output</param>
        /// <returns>True if registered successfully, false if already exists or invalid</returns>
        public bool RegisterCamera(string cameraId, Camera camera, bool silent = false)
        {
            if (string.IsNullOrEmpty(cameraId))
            {
                Debug.LogError($"{_logPrefix} Cannot register camera: cameraId is null or empty");
                return false;
            }

            if (camera == null)
            {
                Debug.LogError($"{_logPrefix} Cannot register camera '{cameraId}': camera is null");
                return false;
            }

            if (_cameras.ContainsKey(cameraId))
            {
                if (_cameras[cameraId] == camera)
                {
                    if (!silent)
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} Camera '{cameraId}' is already registered with the same instance"
                        );
                    }
                    return false;
                }
                else
                {
                    Debug.LogWarning(
                        $"{_logPrefix} Replacing existing camera registration for '{cameraId}'"
                    );
                }
            }

            _cameras[cameraId] = camera;

            if (!silent)
            {
                Debug.Log($"{_logPrefix} Registered camera: '{cameraId}' ({camera.name})");
            }

            OnCameraRegistered?.Invoke(cameraId, camera);
            return true;
        }

        /// <summary>
        /// Unregisters a camera by ID
        /// </summary>
        /// <param name="cameraId">Camera identifier to unregister</param>
        /// <returns>True if unregistered successfully, false if not found</returns>
        public bool UnregisterCamera(string cameraId)
        {
            if (string.IsNullOrEmpty(cameraId))
            {
                Debug.LogError($"{_logPrefix} Cannot unregister camera: cameraId is null or empty");
                return false;
            }

            if (_cameras.Remove(cameraId))
            {
                Debug.Log($"{_logPrefix} Unregistered camera: '{cameraId}'");
                OnCameraUnregistered?.Invoke(cameraId);
                return true;
            }

            Debug.LogWarning($"{_logPrefix} Camera '{cameraId}' not found for unregistration");
            return false;
        }

        /// <summary>
        /// Gets a camera by its unique identifier
        /// </summary>
        /// <param name="cameraId">Camera identifier</param>
        /// <returns>Camera component if found, null otherwise</returns>
        public Camera GetCamera(string cameraId)
        {
            if (string.IsNullOrEmpty(cameraId))
            {
                Debug.LogWarning($"{_logPrefix} Cannot get camera: cameraId is null or empty");
                return null;
            }

            if (_cameras.TryGetValue(cameraId, out Camera camera))
            {
                return camera;
            }

            // Try to find camera by GameObject name as fallback
            GameObject cameraObj = GameObject.Find(cameraId);
            if (cameraObj != null)
            {
                Camera foundCamera = cameraObj.GetComponent<Camera>();
                if (foundCamera != null)
                {
                    Debug.Log(
                        $"{_logPrefix} Found unregistered camera by GameObject name: '{cameraId}', auto-registering"
                    );
                    RegisterCamera(cameraId, foundCamera);
                    return foundCamera;
                }
            }

            Debug.LogWarning($"{_logPrefix} Camera '{cameraId}' not found");
            return null;
        }

        /// <summary>
        /// Tries to get a camera by ID without logging warnings
        /// </summary>
        /// <param name="cameraId">Camera identifier</param>
        /// <param name="camera">Output camera if found</param>
        /// <returns>True if camera found, false otherwise</returns>
        public bool TryGetCamera(string cameraId, out Camera camera)
        {
            camera = null;

            if (string.IsNullOrEmpty(cameraId))
            {
                return false;
            }

            return _cameras.TryGetValue(cameraId, out camera);
        }

        /// <summary>
        /// Gets all registered cameras
        /// </summary>
        /// <returns>Dictionary of camera IDs to Camera components</returns>
        public Dictionary<string, Camera> GetAllCameras()
        {
            return new Dictionary<string, Camera>(_cameras);
        }

        /// <summary>
        /// Gets all registered camera IDs
        /// </summary>
        /// <returns>Array of camera IDs</returns>
        public string[] GetAllCameraIds()
        {
            string[] ids = new string[_cameras.Count];
            _cameras.Keys.CopyTo(ids, 0);
            return ids;
        }

        /// <summary>
        /// Checks if a camera is registered
        /// </summary>
        /// <param name="cameraId">Camera identifier</param>
        /// <returns>True if camera is registered</returns>
        public bool IsCameraRegistered(string cameraId)
        {
            return !string.IsNullOrEmpty(cameraId) && _cameras.ContainsKey(cameraId);
        }

        /// <summary>
        /// Gets the number of registered cameras
        /// </summary>
        public int CameraCount => _cameras.Count;

        /// <summary>
        /// Clears all camera registrations
        /// </summary>
        public void ClearAllCameras()
        {
            int count = _cameras.Count;
            _cameras.Clear();
            Debug.Log($"{_logPrefix} Cleared {count} camera registration(s)");
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
