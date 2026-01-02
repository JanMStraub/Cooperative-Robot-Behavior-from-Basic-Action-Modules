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
                return true;
            }

            Debug.LogWarning($"{_logPrefix} Camera '{cameraId}' not found for unregistration");
            return false;
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
