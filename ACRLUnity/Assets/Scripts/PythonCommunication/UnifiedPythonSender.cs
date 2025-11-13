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

        // Internal connections (one per port)
        private SingleImageConnection _singleConnection; // Port 5005
        private StereoImageConnection _stereoConnection; // Port 5006

        // Streaming timers
        private float _singleStreamTimer = 0f;
        private float _stereoStreamTimer = 0f;

        // Helper variables
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

                // Create connection handlers
                GameObject singleConnObj = new GameObject("SingleImageConnection");
                singleConnObj.transform.SetParent(transform);
                _singleConnection = singleConnObj.AddComponent<SingleImageConnection>();

                GameObject stereoConnObj = new GameObject("StereoImageConnection");
                stereoConnObj.transform.SetParent(transform);
                _stereoConnection = stereoConnObj.AddComponent<StereoImageConnection>();

                Debug.Log(
                    $"{_logPrefix} ✓ Initialized with two connection handlers (ports 5005, 5006)"
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
            // Single camera streaming
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

            // Stereo streaming
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
                Instance = null;
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

            protected override void OnConnected() { }

            protected override void OnConnectionFailed(Exception exception) { }

            protected override void OnDisconnecting() { }

            protected override void OnDisconnected() { }

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

                    // Send data per streaming protocol
                    // [camera_id_len][camera_id][prompt_len][prompt][image_len][image_data]

                    // Camera ID
                    byte[] idBytes = System.Text.Encoding.UTF8.GetBytes(cameraId);
                    byte[] idLength = BitConverter.GetBytes(idBytes.Length);
                    _stream.Write(idLength, 0, idLength.Length);
                    _stream.Write(idBytes, 0, idBytes.Length);

                    // Prompt
                    byte[] promptBytes = System.Text.Encoding.UTF8.GetBytes(prompt ?? "");
                    byte[] promptLength = BitConverter.GetBytes(promptBytes.Length);
                    _stream.Write(promptLength, 0, promptLength.Length);
                    _stream.Write(promptBytes, 0, promptBytes.Length);

                    // Image
                    byte[] imageLength = BitConverter.GetBytes(imageBytes.Length);
                    _stream.Write(imageLength, 0, imageLength.Length);
                    _stream.Write(imageBytes, 0, imageBytes.Length);

                    _stream.Flush();

                    string promptInfo = string.IsNullOrEmpty(prompt)
                        ? ""
                        : $" with prompt: '{prompt}'";
                    Debug.Log(
                        $"{_logPrefix} Sent {imageBytes.Length} bytes for camera '{cameraId}'{promptInfo}"
                    );

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
                if (cam == null)
                {
                    Debug.LogError($"{_logPrefix} Camera is null");
                    return false;
                }

                if (!IsConnected)
                {
                    Debug.LogWarning($"{_logPrefix} Cannot send - not connected");
                    return false;
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
                    byte[] imageData = texture.EncodeToJPG(jpegQuality);

                    // Send
                    return SendImage(imageData, cameraId, prompt);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"{_logPrefix} Error capturing/sending: {ex.Message}");
                    return false;
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

            protected override void OnConnected() { }

            protected override void OnConnectionFailed(Exception exception) { }

            protected override void OnDisconnecting() { }

            protected override void OnDisconnected() { }

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

                    // Send stereo pair per protocol
                    // [cam_pair_id_len][cam_pair_id][camera_L_id_len][camera_L_id]
                    // [camera_R_id_len][camera_R_id][prompt_len][prompt]
                    // [image_L_len][image_L_data][image_R_len][image_R_data]

                    // Camera pair ID
                    byte[] pairIdBytes = System.Text.Encoding.UTF8.GetBytes(cameraPairId);
                    byte[] pairIdLength = BitConverter.GetBytes(pairIdBytes.Length);
                    _stream.Write(pairIdLength, 0, pairIdLength.Length);
                    _stream.Write(pairIdBytes, 0, pairIdBytes.Length);

                    // Camera Left ID
                    byte[] camLIdBytes = System.Text.Encoding.UTF8.GetBytes(cameraLeftId);
                    byte[] camLIdLength = BitConverter.GetBytes(camLIdBytes.Length);
                    _stream.Write(camLIdLength, 0, camLIdLength.Length);
                    _stream.Write(camLIdBytes, 0, camLIdBytes.Length);

                    // Camera Right ID
                    byte[] camRIdBytes = System.Text.Encoding.UTF8.GetBytes(cameraRightId);
                    byte[] camRIdLength = BitConverter.GetBytes(camRIdBytes.Length);
                    _stream.Write(camRIdLength, 0, camRIdLength.Length);
                    _stream.Write(camRIdBytes, 0, camRIdBytes.Length);

                    // Prompt
                    byte[] promptBytes = System.Text.Encoding.UTF8.GetBytes(prompt ?? "");
                    byte[] promptLength = BitConverter.GetBytes(promptBytes.Length);
                    _stream.Write(promptLength, 0, promptLength.Length);
                    _stream.Write(promptBytes, 0, promptBytes.Length);

                    // Left image
                    byte[] imageLLength = BitConverter.GetBytes(leftImageBytes.Length);
                    _stream.Write(imageLLength, 0, imageLLength.Length);
                    _stream.Write(leftImageBytes, 0, leftImageBytes.Length);

                    // Right image
                    byte[] imageRLength = BitConverter.GetBytes(rightImageBytes.Length);
                    _stream.Write(imageRLength, 0, imageRLength.Length);
                    _stream.Write(rightImageBytes, 0, rightImageBytes.Length);

                    _stream.Flush();

                    string promptInfo = string.IsNullOrEmpty(prompt)
                        ? ""
                        : $" with prompt: '{prompt}'";
                    Debug.Log(
                        $"{_logPrefix} Sent stereo pair '{cameraPairId}' (L:{leftImageBytes.Length}B, R:{rightImageBytes.Length}B){promptInfo}"
                    );

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
                if (leftCam == null || rightCam == null)
                {
                    Debug.LogError($"{_logPrefix} Left or right camera is null");
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

                RenderTexture rtLeft = null;
                RenderTexture rtRight = null;
                Texture2D textureLeft = null;
                Texture2D textureRight = null;

                try
                {
                    // Capture left
                    rtLeft = new RenderTexture(leftCam.pixelWidth, leftCam.pixelHeight, 24);
                    leftCam.targetTexture = rtLeft;
                    leftCam.Render();

                    RenderTexture.active = rtLeft;
                    textureLeft = new Texture2D(
                        rtLeft.width,
                        rtLeft.height,
                        TextureFormat.RGB24,
                        false
                    );
                    textureLeft.ReadPixels(new Rect(0, 0, rtLeft.width, rtLeft.height), 0, 0);
                    textureLeft.Apply();

                    // Capture right
                    rtRight = new RenderTexture(rightCam.pixelWidth, rightCam.pixelHeight, 24);
                    rightCam.targetTexture = rtRight;
                    rightCam.Render();

                    RenderTexture.active = rtRight;
                    textureRight = new Texture2D(
                        rtRight.width,
                        rtRight.height,
                        TextureFormat.RGB24,
                        false
                    );
                    textureRight.ReadPixels(new Rect(0, 0, rtRight.width, rtRight.height), 0, 0);
                    textureRight.Apply();

                    // Encode
                    byte[] imageLeftData = textureLeft.EncodeToJPG(jpegQuality);
                    byte[] imageRightData = textureRight.EncodeToJPG(jpegQuality);

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
                catch (Exception ex)
                {
                    Debug.LogError(
                        $"{_logPrefix} Error capturing/sending stereo pair: {ex.Message}"
                    );
                    return false;
                }
                finally
                {
                    // Cleanup
                    if (leftCam != null)
                        leftCam.targetTexture = null;
                    if (rightCam != null)
                        rightCam.targetTexture = null;
                    RenderTexture.active = null;
                    if (rtLeft != null)
                        UnityEngine.Object.Destroy(rtLeft);
                    if (rtRight != null)
                        UnityEngine.Object.Destroy(rtRight);
                    if (textureLeft != null)
                        UnityEngine.Object.Destroy(textureLeft);
                    if (textureRight != null)
                        UnityEngine.Object.Destroy(textureRight);
                }
            }
        }

        #endregion
    }
}
