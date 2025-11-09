using System;
using System.Collections;
using PythonCommunication;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Vision
{
#if UNITY_EDITOR
    [CustomEditor(typeof(StereoCameraController))]
    public class StereoCameraControllerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            var controller = (StereoCameraController)target;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Capture Controls", EditorStyles.boldLabel);

            // Capture button
            if (GUILayout.Button("Capture & Send for Depth Detection", GUILayout.Height(30)))
                controller.CaptureAndSend();

            if (!Application.isPlaying)
                return;

            bool isConnected =
                StereoImageSender.Instance != null && StereoImageSender.Instance.IsConnected;
            string statusText = isConnected ? "Connected to Stereo Detector" : "Disconnected";
            Color statusColor = isConnected ? new Color(0.6f, 1f, 0.6f) : new Color(1f, 0.6f, 0.6f);

            var originalColor = GUI.color;
            GUI.color = statusColor;
            EditorGUILayout.TextField("Stereo Server Status", statusText, EditorStyles.boldLabel);
            GUI.color = originalColor;

            // Processing indicator
            if (controller.IsProcessing)
            {
                GUI.color = new Color(1f, 0.9f, 0.4f);
                EditorGUILayout.TextField(
                    "Detection Status",
                    "Processing...",
                    EditorStyles.boldLabel
                );
                GUI.color = originalColor;
                Repaint();
            }
            else
            {
                EditorGUILayout.TextField("Detection Status", "Idle");
            }
        }
    }
#endif

    /// <summary>
    /// Metadata for stereo camera configuration
    /// </summary>
    [Serializable]
    public class StereoMetadata
    {
        public float baseline;
        public float fov;
        public float[] camera_position;
        public float[] camera_rotation;
    }

    /// <summary>
    /// Controls stereo camera pair for 3D depth detection.
    /// Captures images from left and right cameras and sends them to the stereo detector for 3D object localization.
    /// </summary>
    public class StereoCameraController : MonoBehaviour
    {
        [Header("Stereo Camera Configuration")]
        [Header("Image Settings")]
        [SerializeField]
        [Tooltip("Width of captured images in pixels")]
        private int _imageWidth = 1280; // Good balance of quality and performance

        [SerializeField]
        [Tooltip("Height of captured images in pixels")]
        private int _imageHeight = 960; // Good balance of quality and performance

        [SerializeField]
        [Tooltip("JPEG compression quality (1-100)")]
        [Range(1, 100)]
        private int _jpegQuality = 85;

        private float _stereoBaseline;
        private float _cameraFOV;

        // State
        private bool _isProcessing = false;
        private int _captureCounter = 0;

        // Events
        public event Action<DepthResult> OnDepthResultReceived;

        // Properties
        public bool IsProcessing => _isProcessing;
        public int CaptureCount => _captureCounter;

        // Helper variables

        private Camera _leftCamera;
        private Camera _rightCamera;
        private string _cameraPairId;

        /// <summary>
        /// Get the camera FOV - either from actual camera or manual override
        /// </summary>
        private float GetCameraFOV()
        {
            if (_leftCamera != null)
            {
                return _leftCamera.fieldOfView;
            }
            return _cameraFOV;
        }

        /// <summary>
        /// Get the stereo baseline - either manual or auto-calculated from camera positions
        /// </summary>
        private float GetStereoBaseline()
        {
            if (_leftCamera != null && _rightCamera != null)
            {
                // Calculate distance between cameras (typically horizontal X distance)
                float distance = Vector3.Distance(
                    _leftCamera.transform.position,
                    _rightCamera.transform.position
                );
                return distance;
            }
            return _stereoBaseline;
        }

        /// <summary>
        /// Finds cameras as child objects of this GameObject
        /// Expects left camera at index 0, right camera at index 1
        /// </summary>
        private void FindCameras()
        {
            // Try to find cameras as children of this GameObject
            if (transform.childCount >= 2)
            {
                Transform leftChild = transform.GetChild(0);
                Transform rightChild = transform.GetChild(1);

                _leftCamera = leftChild.GetComponent<Camera>();
                _rightCamera = rightChild.GetComponent<Camera>();

                if (_leftCamera != null && _rightCamera != null)
                {
                    Debug.Log(
                        $"[STEREO_CAMERA_CONTROLLER] Found cameras: {leftChild.name} and {rightChild.name}"
                    );
                    return;
                }
            }

            Debug.LogError(
                "[STEREO_CAMERA_CONTROLLER] Failed to find left and right cameras. "
                    + "Ensure cameras are children of this GameObject or named 'LeftCamera' and 'RightCamera'"
            );
        }

        /// <summary>
        /// Unity Start - validate configuration and subscribe to results
        /// </summary>
        private void Start()
        {
            FindCameras();

            // Validate cameras
            if (_leftCamera == null || _rightCamera == null)
            {
                Debug.LogError(
                    $"[STEREO_CAMERA_CONTROLLER] Both left and right cameras must be assigned!"
                );
                enabled = false;
                return;
            }

            _cameraPairId = name;

            // Register cameras with CameraManager
            if (CameraManager.Instance != null)
            {
                CameraManager.Instance.RegisterCamera(_cameraPairId + "_L", _leftCamera);
                CameraManager.Instance.RegisterCamera(_cameraPairId + "_R", _rightCamera);
            }
            else
            {
                Debug.LogWarning(
                    "[STEREO_CAMERA_CONTROLLER] CameraManager not found. Create a CameraManager GameObject in the scene."
                );
            }

            // Subscribe to depth results
            if (DepthResultsReceiver.Instance != null)
            {
                DepthResultsReceiver.Instance.OnDepthResultReceived += HandleDepthResult;
                float fov = GetCameraFOV();
                float baseline = GetStereoBaseline();
                Debug.Log(
                    $"[STEREO_CAMERA_CONTROLLER] Initialized: {_cameraPairId}, Baseline={baseline}m, FOV={fov}°"
                );

                // Log camera positions for debugging
                if (_leftCamera != null && _rightCamera != null)
                {
                    Vector3 leftPos = _leftCamera.transform.position;
                    Vector3 rightPos = _rightCamera.transform.position;
                    float actualDistance = Vector3.Distance(leftPos, rightPos);
                    Debug.Log(
                        $"[STEREO_CAMERA_CONTROLLER] Left camera: {leftPos}, Right camera: {rightPos}, Distance: {actualDistance}m"
                    );
                }
            }
            else
            {
                Debug.LogWarning(
                    "[STEREO_CAMERA_CONTROLLER] DepthResultsReceiver not found - results won't be received"
                );
            }
        }

        /// <summary>
        /// Handles depth detection results
        /// </summary>
        private void HandleDepthResult(DepthResult result)
        {
            if (result == null)
                return;

            // Check if this result is for our camera pair
            if (result.camera_id == _cameraPairId)
            {
                _isProcessing = false;

                // Forward to subscribers
                OnDepthResultReceived?.Invoke(result);
            }
        }

        /// <summary>
        /// Capture stereo pair and send for depth detection
        /// </summary>
        public void CaptureAndSend()
        {
            if (_isProcessing)
            {
                Debug.LogWarning("[STEREO_CAMERA_CONTROLLER] Already processing a capture");
                return;
            }

            if (!enabled)
            {
                Debug.LogWarning("[STEREO_CAMERA_CONTROLLER] Component is disabled");
                return;
            }

            StartCoroutine(CaptureAndSendCoroutine());
        }

        /// <summary>
        /// Coroutine to capture and send stereo pair
        /// </summary>
        private IEnumerator CaptureAndSendCoroutine()
        {
            _isProcessing = true;
            float startTime = Time.realtimeSinceStartup;

            try
            {
                // Check if sender is available
                if (StereoImageSender.Instance == null || !StereoImageSender.Instance.IsConnected)
                {
                    throw new Exception("StereoImageSender not available or not connected");
                }

                // Capture left camera
                byte[] leftImageBytes = CaptureImage(_leftCamera);
                if (leftImageBytes == null)
                    throw new Exception("Failed to capture left camera image");

                // Capture right camera
                byte[] rightImageBytes = CaptureImage(_rightCamera);
                if (rightImageBytes == null)
                    throw new Exception("Failed to capture right camera image");

                // Encode camera parameters as JSON metadata
                float fov = GetCameraFOV();
                float baseline = GetStereoBaseline();

                // Include left camera position and rotation for coordinate transformation
                Vector3 leftPos = _leftCamera.transform.position;
                Vector3 leftRot = _leftCamera.transform.eulerAngles;

                // Create metadata using JsonUtility for proper formatting
                StereoMetadata metadataObj = new StereoMetadata
                {
                    baseline = baseline,
                    fov = fov,
                    camera_position = new float[] { leftPos.x, leftPos.y, leftPos.z },
                    camera_rotation = new float[] { leftRot.x, leftRot.y, leftRot.z },
                };
                string metadata = JsonUtility.ToJson(metadataObj);

                // Send stereo pair
                bool success = StereoImageSender.Instance.SendStereoPair(
                    leftImageBytes,
                    rightImageBytes,
                    _cameraPairId,
                    _cameraPairId + "_L",
                    _cameraPairId + "_R",
                    metadata
                );

                if (!success)
                    throw new Exception("Failed to send stereo pair");

                float captureTime = Time.realtimeSinceStartup - startTime;
                _captureCounter++;

                Debug.Log(
                    $"[STEREO_CAMERA_CONTROLLER] Stereo pair sent successfully in {captureTime:F2}s "
                        + $"(L:{leftImageBytes.Length / 1024f:F1} KB, R:{rightImageBytes.Length / 1024f:F1} KB). "
                        + $"Waiting for depth detection..."
                );
            }
            catch (Exception ex)
            {
                Debug.LogError($"[STEREO_CAMERA_CONTROLLER] Capture failed: {ex.Message}");
                _isProcessing = false;
            }

            yield return null;
        }

        /// <summary>
        /// Capture image from a camera
        /// </summary>
        private byte[] CaptureImage(Camera camera)
        {
            if (camera == null)
                return null;

            RenderTexture rt = null;
            Texture2D texture = null;

            try
            {
                // Create render texture
                rt = new RenderTexture(_imageWidth, _imageHeight, 24);
                camera.targetTexture = rt;
                camera.Render();

                // Read pixels
                RenderTexture.active = rt;
                texture = new Texture2D(_imageWidth, _imageHeight, TextureFormat.RGB24, false);
                texture.ReadPixels(new Rect(0, 0, _imageWidth, _imageHeight), 0, 0);
                texture.Apply();

                // Cleanup render texture
                camera.targetTexture = null;
                RenderTexture.active = null;

                // Encode to JPEG
                byte[] bytes = texture.EncodeToJPG(_jpegQuality);
                return bytes;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[STEREO_CAMERA_CONTROLLER] Error capturing image: {ex.Message}");
                return null;
            }
            finally
            {
                // Cleanup resources
                if (rt != null)
                    Destroy(rt);
                if (texture != null)
                    Destroy(texture);
            }
        }

        /// <summary>
        /// Unity OnDestroy - cleanup
        /// </summary>
        private void OnDestroy()
        {
            // Unregister cameras from CameraManager
            if (CameraManager.Instance != null && !string.IsNullOrEmpty(_cameraPairId))
            {
                CameraManager.Instance.UnregisterCamera(_cameraPairId + "_L");
                CameraManager.Instance.UnregisterCamera(_cameraPairId + "_R");
            }

            // Unsubscribe from depth results
            if (DepthResultsReceiver.Instance != null)
            {
                DepthResultsReceiver.Instance.OnDepthResultReceived -= HandleDepthResult;
            }

            if (_captureCounter > 0)
            {
                Debug.Log($"[STEREO_CAMERA_CONTROLLER] Destroyed after {_captureCounter} captures");
            }
        }
    }
}
