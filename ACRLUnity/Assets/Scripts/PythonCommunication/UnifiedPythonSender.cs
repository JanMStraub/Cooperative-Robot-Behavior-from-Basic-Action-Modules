using System;
using PythonCommunication.Core;
using UnityEngine;
using Core;

namespace PythonCommunication
{
    /// <summary>
    /// Unified sender for all Python image communication.
    /// Handles both single camera images (port 5005) and stereo pairs (port 5006).
    /// Replaces ImageSender and StereoImageSender with a single, streamlined implementation.
    /// </summary>
    public class UnifiedPythonSender : MonoBehaviour
    {
        public static UnifiedPythonSender Instance { get; private set; }

        [Header("Image Encoding")]
        [Tooltip("JPEG quality (1-100, higher is better quality but larger file)")]
        [SerializeField]
        [Range(1, 100)]
        private int _jpegQuality = 85;

        [Header("Single Camera Streaming (Optional)")]
        [Tooltip("Enable continuous single camera streaming")]
        [SerializeField]
        private bool _enableSingleStreaming = false;

        [Tooltip("Camera to stream from (single mode)")]
        [SerializeField]
        private Camera _singleStreamCamera;

        [Tooltip("Camera identifier for streaming")]
        [SerializeField]
        private string _singleStreamCameraId = "Main";

        [Tooltip("Prompt for single camera streaming")]
        [SerializeField]
        private string _singleStreamPrompt = "";

        [Tooltip("Time between single camera frames (seconds)")]
        [SerializeField]
        private float _singleStreamInterval = 0.2f;

        [Header("Stereo Streaming (Optional)")]
        [Tooltip("Enable continuous stereo streaming")]
        [SerializeField]
        private bool _enableStereoStreaming = false;

        [Tooltip("Left camera for stereo streaming")]
        [SerializeField]
        private Camera _stereoLeftCamera;

        [Tooltip("Right camera for stereo streaming")]
        [SerializeField]
        private Camera _stereoRightCamera;

        [Tooltip("Stereo pair identifier")]
        [SerializeField]
        private string _stereoPairId = "stereo";

        [Tooltip("Left camera identifier")]
        [SerializeField]
        private string _stereoLeftId = "L";

        [Tooltip("Right camera identifier")]
        [SerializeField]
        private string _stereoRightId = "R";

        [Tooltip("Prompt for stereo streaming")]
        [SerializeField]
        private string _stereoStreamPrompt = "";

        [Tooltip("Time between stereo pairs (seconds)")]
        [SerializeField]
        private float _stereoStreamInterval = 0.5f;

        private SingleImageConnection _singleConnection;
        private StereoImageConnection _stereoConnection;

        private float _singleStreamTimer = 0f;
        private float _stereoStreamTimer = 0f;

        private const string _logPrefix = "[UNIFIED_PYTHON_SENDER]";

        #region Singleton

        /// <summary>
        /// Initialize singleton instance
        /// </summary>
        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);

                GameObject singleConnObj = new GameObject("SingleImageConnection");
                singleConnObj.transform.SetParent(transform);
                _singleConnection = singleConnObj.AddComponent<SingleImageConnection>();

                GameObject stereoConnObj = new GameObject("StereoImageConnection");
                stereoConnObj.transform.SetParent(transform);
                _stereoConnection = stereoConnObj.AddComponent<StereoImageConnection>();

                Debug.Log(
                    $"{_logPrefix} Initialized with two connection handlers (ports 5005, 5006)"
                );
            }
            else
            {
                Destroy(gameObject);
            }
        }

        #endregion

        #region Unity Lifecycle

        /// <summary>
        /// Update loop - handle streaming if enabled
        /// </summary>
        private void Update()
        {
            if (
                _enableSingleStreaming
                && _singleStreamCamera != null
                && _singleConnection != null
                && _singleConnection.IsConnected
            )
            {
                _singleStreamTimer += Time.deltaTime;
                if (_singleStreamTimer >= _singleStreamInterval)
                {
                    CaptureAndSend(_singleStreamCamera, _singleStreamCameraId, _singleStreamPrompt);
                    _singleStreamTimer = 0f;
                }
            }

            if (
                _enableStereoStreaming
                && _stereoLeftCamera != null
                && _stereoRightCamera != null
                && _stereoConnection != null
                && _stereoConnection.IsConnected
            )
            {
                _stereoStreamTimer += Time.deltaTime;
                if (_stereoStreamTimer >= _stereoStreamInterval)
                {
                    CaptureAndSendStereoPair(
                        _stereoLeftCamera,
                        _stereoRightCamera,
                        _stereoStreamPrompt,
                        _stereoPairId
                    );
                    _stereoStreamTimer = 0f;
                }
            }
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                // Gracefully disconnect all connections
                if (_singleConnection != null)
                {
                    _singleConnection.Disconnect();
                }

                if (_stereoConnection != null)
                {
                    _stereoConnection.Disconnect();
                }

                Instance = null;
            }
        }

        /// <summary>
        /// Called when Unity exits play mode - ensure clean shutdown
        /// </summary>
        private void OnApplicationQuit()
        {
            // Stop all connections before Unity closes
            if (_singleConnection != null)
            {
                _singleConnection.Disconnect();
            }

            if (_stereoConnection != null)
            {
                _stereoConnection.Disconnect();
            }
        }

        #endregion

        #region Public API - Single Camera (Port 5005)

        /// <summary>
        /// Send pre-encoded single camera image to Python StreamingServer (port 5005).
        /// </summary>
        /// <param name="imageBytes">Encoded image data (PNG/JPG)</param>
        /// <param name="cameraId">Camera identifier</param>
        /// <param name="prompt">Optional LLM prompt</param>
        /// <returns>True if sent successfully</returns>
        public bool SendImage(byte[] imageBytes, string cameraId, string prompt = "")
        {
            if (_singleConnection == null)
            {
                Debug.LogError($"{_logPrefix} Single image connection not initialized");
                return false;
            }

            return _singleConnection.SendImage(imageBytes, cameraId, prompt);
        }

        /// <summary>
        /// Capture image from camera and send to Python StreamingServer (port 5005).
        /// </summary>
        /// <param name="cam">Camera to capture from</param>
        /// <param name="cameraId">Camera identifier</param>
        /// <param name="prompt">Optional LLM prompt</param>
        /// <returns>True if captured and sent successfully</returns>
        public bool CaptureAndSend(Camera cam, string cameraId, string prompt = "")
        {
            if (_singleConnection == null)
            {
                Debug.LogError($"{_logPrefix} Single image connection not initialized");
                return false;
            }

            return _singleConnection.CaptureAndSend(cam, cameraId, prompt, _jpegQuality);
        }

        /// <summary>
        /// Check if single camera connection (port 5005) is active
        /// </summary>
        public bool IsSingleCameraConnected =>
            _singleConnection != null && _singleConnection.IsConnected;

        #endregion

        #region Public API - Stereo Pair (Port 5006)

        /// <summary>
        /// Send pre-encoded stereo image pair to Python StereoDetectionServer (port 5006).
        /// </summary>
        public bool SendStereoPair(
            byte[] leftImageBytes,
            byte[] rightImageBytes,
            string cameraPairId,
            string cameraLeftId,
            string cameraRightId,
            string prompt = ""
        )
        {
            if (_stereoConnection == null)
            {
                Debug.LogError($"{_logPrefix}Stereo connection not initialized");
                return false;
            }

            return _stereoConnection.SendStereoPair(
                leftImageBytes,
                rightImageBytes,
                cameraPairId,
                cameraLeftId,
                cameraRightId,
                prompt
            );
        }

        /// <summary>
        /// Capture and send stereo image pair from two cameras (port 5006).
        /// </summary>
        public bool CaptureAndSendStereoPair(
            Camera leftCam,
            Camera rightCam,
            string prompt = "",
            string cameraPairId = null
        )
        {
            if (_stereoConnection == null)
            {
                Debug.LogError($"{_logPrefix} Stereo connection not initialized");
                return false;
            }

            string pairId = cameraPairId ?? _stereoPairId;
            return _stereoConnection.CaptureAndSendStereoPair(
                leftCam,
                rightCam,
                _stereoLeftId,
                _stereoRightId,
                pairId,
                prompt,
                _jpegQuality
            );
        }

        /// <summary>
        /// Check if stereo connection (port 5006) is active
        /// </summary>
        public bool IsStereoConnected => _stereoConnection != null && _stereoConnection.IsConnected;

        #endregion

        #region Public API - Streaming Control

        /// <summary>
        /// Enable single camera streaming mode
        /// </summary>
        public void EnableSingleCameraStreaming(
            Camera cam,
            string cameraId,
            float interval = 0.2f,
            string prompt = ""
        )
        {
            _singleStreamCamera = cam;
            _singleStreamCameraId = cameraId;
            _singleStreamInterval = interval;
            _singleStreamPrompt = prompt;
            _enableSingleStreaming = true;
            _singleStreamTimer = 0f;

            Debug.Log($"{_logPrefix} Enabled single camera streaming: {cameraId} @ {interval}s");
        }

        /// <summary>
        /// Disable single camera streaming mode
        /// </summary>
        public void DisableSingleCameraStreaming()
        {
            _enableSingleStreaming = false;
            Debug.Log($"{_logPrefix} Disabled single camera streaming");
        }

        /// <summary>
        /// Enable stereo streaming mode
        /// </summary>
        public void EnableStereoStreaming(
            Camera leftCam,
            Camera rightCam,
            string pairId,
            float interval = 0.5f,
            string prompt = ""
        )
        {
            _stereoLeftCamera = leftCam;
            _stereoRightCamera = rightCam;
            _stereoPairId = pairId;
            _stereoStreamInterval = interval;
            _stereoStreamPrompt = prompt;
            _enableStereoStreaming = true;
            _stereoStreamTimer = 0f;

            Debug.Log($"{_logPrefix} Enabled stereo streaming: {pairId} @ {interval}s");
        }

        /// <summary>
        /// Disable stereo streaming mode
        /// </summary>
        public void DisableStereoStreaming()
        {
            _enableStereoStreaming = false;
            Debug.Log($"{_logPrefix} Disabled stereo streaming");
        }

        /// <summary>
        /// Disable all streaming modes
        /// </summary>
        public void DisableAllStreaming()
        {
            DisableSingleCameraStreaming();
            DisableStereoStreaming();
        }

        #endregion

        #region Image Capture Utilities

        /// <summary>
        /// Capture image from camera and encode to JPEG.
        /// Handles RenderTexture creation, rendering, and cleanup.
        /// </summary>
        /// <param name="cam">Camera to capture from</param>
        /// <param name="jpegQuality">JPEG quality (1-100)</param>
        /// <returns>Encoded JPEG bytes, or null on error</returns>
        private static byte[] CaptureFromCamera(Camera cam, int jpegQuality)
        {
            if (cam == null)
            {
                Debug.LogError("[CAPTURE_HELPER] Camera is null");
                return null;
            }

            RenderTexture rt = null;
            Texture2D texture = null;

            try
            {
                // Create temporary render texture
                rt = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 24);
                cam.targetTexture = rt;
                cam.Render();

                // Read pixels
                RenderTexture.active = rt;
                texture = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
                texture.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                texture.Apply();

                // Encode to JPEG
                return texture.EncodeToJPG(jpegQuality);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[CAPTURE_HELPER] Error capturing image: {ex.Message}");
                return null;
            }
            finally
            {
                // Cleanup
                if (cam != null)
                    cam.targetTexture = null;
                RenderTexture.active = null;
                if (rt != null)
                    UnityEngine.Object.Destroy(rt);
                if (texture != null)
                    UnityEngine.Object.Destroy(texture);
            }
        }

        #endregion

        #region Inner Connection Classes

        /// <summary>
        /// Handles single camera image sending (port 5005 - StreamingServer)
        /// </summary>
        private class SingleImageConnection : TCPClientBase
        {
            private const string _logPrefix = "[SINGLE_IMAGE_CONNECTION]";

            private void Awake()
            {
                _serverPort = CommunicationConstants.STREAMING_SERVER_PORT; // StreamingServer port
                _autoConnect = true;
            }

            /// <summary>
            /// Send pre-encoded image (protocol: [camera_id_len][camera_id][prompt_len][prompt][image_len][image_data])
            /// </summary>
            public bool SendImage(byte[] imageBytes, string cameraId, string prompt)
            {
                if (imageBytes == null || imageBytes.Length == 0)
                {
                    Debug.LogError($"{_logPrefix} Cannot send empty image data");
                    return false;
                }

                if (!IsConnected)
                {
                    Debug.LogWarning($"{_logPrefix} Not connected - attempting to connect");
                    Connect();
                    if (!IsConnected)
                    {
                        Debug.LogError($"{_logPrefix} Connection failed");
                        return false;
                    }
                }

                if (string.IsNullOrEmpty(cameraId))
                {
                    Debug.LogWarning($"{_logPrefix} Camera ID is empty, using 'Unknown'");
                    cameraId = "Unknown";
                }

                try
                {
                    // Validate with protocol
                    if (!UnityProtocol.IsValidImageSize(imageBytes))
                    {
                        Debug.LogError(
                            $"{_logPrefix} Image size {imageBytes.Length} exceeds protocol limit"
                        );
                        return false;
                    }

                    if (
                        !UnityProtocol.IsValidStringLength(cameraId)
                        || !UnityProtocol.IsValidStringLength(prompt)
                    )
                    {
                        Debug.LogError(
                            $"{_logPrefix} Camera ID or prompt exceeds protocol string length limit"
                        );
                        return false;
                    }

                    // Verify connection alive
                    if (!VerifyConnection())
                    {
                        Debug.LogError(
                            $"{_logPrefix} Connection verification failed - attempting reconnect"
                        );
                        Connect();
                        if (!VerifyConnection())
                        {
                            Debug.LogError($"{_logPrefix} Reconnection failed");
                            return false;
                        }
                    }

                    // Generate unique request ID for correlation (Protocol V2)
                    uint requestId = GenerateRequestId();

                    // Encode message using UnityProtocol (Protocol V2)
                    byte[] message = UnityProtocol.EncodeImageMessage(
                        cameraId,
                        prompt ?? "",
                        imageBytes,
                        requestId
                    );

                    // Send encoded message
                    _stream.Write(message, 0, message.Length);
                    _stream.Flush();

#if UNITY_EDITOR
                    string promptInfo = string.IsNullOrEmpty(prompt)
                        ? ""
                        : $" with prompt: '{prompt}'";
                    Debug.Log(
                        $"{_logPrefix} [req={requestId}] Sent {imageBytes.Length} bytes for camera '{cameraId}'{promptInfo}"
                    );
#endif

                    return true;
                }
                catch (System.IO.IOException ioEx)
                {
                    Debug.LogError($"{_logPrefix} Network error: {ioEx.Message}");
                    _isConnected = false;
                    return false;
                }
                catch (System.Net.Sockets.SocketException sockEx)
                {
                    Debug.LogError($"{_logPrefix} Socket error: {sockEx.Message}");
                    _isConnected = false;
                    return false;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"{_logPrefix} Unexpected error: {ex.Message}");
                    _isConnected = false;
                    return false;
                }
            }

            /// <summary>
            /// Capture and send camera image
            /// </summary>
            public bool CaptureAndSend(Camera cam, string cameraId, string prompt, int jpegQuality)
            {
                if (!IsConnected)
                {
                    Debug.LogWarning($"{_logPrefix} Cannot send - not connected");
                    return false;
                }

                byte[] imageData = UnifiedPythonSender.CaptureFromCamera(cam, jpegQuality);
                if (imageData == null)
                {
                    return false;
                }

                return SendImage(imageData, cameraId, prompt);
            }
        }

        /// <summary>
        /// Handles stereo image pair sending (port 5006 - StereoDetectionServer)
        /// </summary>
        private class StereoImageConnection : TCPClientBase
        {
            private const string _logPrefix = "[STEREO_IMAGE_CONNECTION]";

            private void Awake()
            {
                _serverPort = CommunicationConstants.STEREO_DETECTION_SERVER_PORT; // StereoDetectionServer port
                _autoConnect = false; // Only connect when actually used
            }

            /// <summary>
            /// Send stereo pair (protocol: [pair_id_len][pair_id][left_id_len][left_id][right_id_len][right_id][prompt_len][prompt][left_img_len][left_img][right_img_len][right_img])
            /// </summary>
            public bool SendStereoPair(
                byte[] leftImageBytes,
                byte[] rightImageBytes,
                string cameraPairId,
                string cameraLeftId,
                string cameraRightId,
                string prompt
            )
            {
                if (leftImageBytes == null || leftImageBytes.Length == 0)
                {
                    Debug.LogError($"{_logPrefix} Left image data is empty");
                    return false;
                }

                if (rightImageBytes == null || rightImageBytes.Length == 0)
                {
                    Debug.LogError($"{_logPrefix} Right image data is empty");
                    return false;
                }

                if (!IsConnected)
                {
                    Debug.LogWarning($"{_logPrefix} Not connected - attempting to connect");
                    Connect();
                    if (!IsConnected)
                    {
                        Debug.LogError($"{_logPrefix} Connection failed");
                        return false;
                    }
                }

                if (string.IsNullOrEmpty(cameraPairId))
                {
                    Debug.LogWarning($"{_logPrefix} Camera pair ID is empty, using 'stereo'");
                    cameraPairId = "stereo";
                }

                try
                {
                    // Validate sizes
                    if (
                        !UnityProtocol.IsValidImageSize(leftImageBytes)
                        || !UnityProtocol.IsValidImageSize(rightImageBytes)
                    )
                    {
                        Debug.LogError($"{_logPrefix} Image size exceeds protocol limit");
                        return false;
                    }

                    // Generate unique request ID for correlation (Protocol V2)
                    uint requestId = GenerateRequestId();

                    // Encode message using UnityProtocol (Protocol V2)
                    byte[] message = UnityProtocol.EncodeStereoImageMessage(
                        cameraPairId,
                        cameraLeftId,
                        cameraRightId,
                        prompt ?? "",
                        leftImageBytes,
                        rightImageBytes,
                        requestId
                    );

                    // Send encoded message
                    _stream.Write(message, 0, message.Length);
                    _stream.Flush();

#if UNITY_EDITOR
                    string promptInfo = string.IsNullOrEmpty(prompt)
                        ? ""
                        : $" with prompt: '{prompt}'";
                    Debug.Log(
                        $"{_logPrefix} [req={requestId}] Sent stereo pair '{cameraPairId}' (L:{leftImageBytes.Length}B, R:{rightImageBytes.Length}B){promptInfo}"
                    );
#endif

                    return true;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"{_logPrefix} Error sending stereo pair: {ex.Message}");
                    _isConnected = false;
                    return false;
                }
            }

            /// <summary>
            /// Capture and send stereo pair
            /// </summary>
            public bool CaptureAndSendStereoPair(
                Camera leftCam,
                Camera rightCam,
                string cameraLeftId,
                string cameraRightId,
                string cameraPairId,
                string prompt,
                int jpegQuality
            )
            {
                if (!IsConnected)
                {
                    Debug.LogWarning($"{_logPrefix} Not connected - attempting to connect");
                    Connect();
                    if (!IsConnected)
                    {
                        Debug.LogError($"{_logPrefix} Connection failed");
                        return false;
                    }
                }

                // Capture both cameras
                byte[] imageLeftData = UnifiedPythonSender.CaptureFromCamera(leftCam, jpegQuality);
                byte[] imageRightData = UnifiedPythonSender.CaptureFromCamera(rightCam, jpegQuality);

                if (imageLeftData == null || imageRightData == null)
                {
                    Debug.LogError($"{_logPrefix} Failed to capture left or right camera image");
                    return false;
                }

                // Send
                return SendStereoPair(
                    imageLeftData,
                    imageRightData,
                    cameraPairId,
                    cameraLeftId,
                    cameraRightId,
                    prompt
                );
            }
        }

        #endregion
    }
}
