using System;
using UnityEngine;
using LLMCommunication.Core;

namespace LLMCommunication
{
    /// <summary>
    /// Sends stereo image pairs to Python StereoDetectionServer via TCP.
    /// Enables 3D object detection with depth estimation from stereo disparity.
    /// </summary>
    public class StereoImageSender : TCPClientBase
    {
        public static StereoImageSender Instance { get; private set; }

        [Header("Stereo Camera Configuration")]
        [Tooltip("Left camera for stereo pair")]
        [SerializeField]
        private Camera _leftCamera;

        [Tooltip("Right camera for stereo pair")]
        [SerializeField]
        private Camera _rightCamera;

        [Tooltip("Camera pair identifier")]
        [SerializeField]
        private string _cameraPairId = "stereo";

        [Tooltip("Left camera identifier")]
        [SerializeField]
        private string _cameraLeftId = "L";

        [Tooltip("Right camera identifier")]
        [SerializeField]
        private string _cameraRightId = "R";

        [Header("Stereo Streaming Settings (Optional)")]
        [Tooltip("Enable continuous stereo streaming mode")]
        [SerializeField]
        private bool _enableStreaming = false;

        [Tooltip("Time between streamed stereo pairs in seconds")]
        [SerializeField]
        private float _sendInterval = 0.5f;

        [Tooltip("Prompt for streamed stereo images")]
        [SerializeField]
        private string _streamPrompt = "";

        // Streaming state
        private float _streamTimer = 0f;

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
            }
            else
            {
                Destroy(gameObject);
                return;
            }

            // Set default port for StereoImageSender
            if (_serverPort == 0)
            {
                _serverPort = 5009; // StereoDetectionServer default port
            }
        }

        #endregion

        #region Unity Lifecycle

        /// <summary>
        /// Update loop - handle streaming if enabled
        /// </summary>
        protected override void Update()
        {
            base.Update(); // Handle auto-reconnect

            // Handle continuous streaming if enabled
            if (_enableStreaming && IsConnected && _leftCamera != null && _rightCamera != null)
            {
                _streamTimer += Time.deltaTime;
                if (_streamTimer >= _sendInterval)
                {
                    CaptureAndSendStereoPair(_leftCamera, _rightCamera, _streamPrompt);
                    _streamTimer = 0f;
                }
            }
        }

        #endregion

        #region TCPClientBase Implementation

        protected override string LogPrefix => "StereoImageSender";

        protected override void OnConnected()
        {
            // No special action needed on connection
        }

        protected override void OnConnectionFailed(Exception exception)
        {
            // Connection failure already logged by base class
        }

        protected override void OnDisconnecting()
        {
            // No cleanup needed before disconnect
        }

        protected override void OnDisconnected()
        {
            // No cleanup needed after disconnect
        }

        #endregion

        #region Public API

        /// <summary>
        /// Capture and send stereo image pair from two cameras.
        ///
        /// Wire protocol:
        /// [cam_pair_id_len:4][cam_pair_id:N]
        /// [camera_L_id_len:4][camera_L_id:N]
        /// [camera_R_id_len:4][camera_R_id:N]
        /// [prompt_len:4][prompt:N]
        /// [image_L_len:4][image_L_data:N]
        /// [image_R_len:4][image_R_data:N]
        /// </summary>
        /// <param name="leftCam">Left camera</param>
        /// <param name="rightCam">Right camera</param>
        /// <param name="prompt">Optional prompt/metadata</param>
        /// <param name="cameraPairId">Camera pair identifier (uses default if null)</param>
        /// <returns>True if captured and sent successfully</returns>
        public bool CaptureAndSendStereoPair(
            Camera leftCam,
            Camera rightCam,
            string prompt = "",
            string cameraPairId = null
        )
        {
            if (leftCam == null || rightCam == null)
            {
                LogError("Left or right camera is null");
                return false;
            }

            if (!IsConnected)
            {
                LogWarning("Cannot send stereo pair - not connected");
                return false;
            }

            RenderTexture rtLeft = null;
            RenderTexture rtRight = null;
            Texture2D textureLeft = null;
            Texture2D textureRight = null;

            try
            {
                // Capture left camera
                rtLeft = new RenderTexture(leftCam.pixelWidth, leftCam.pixelHeight, 24);
                leftCam.targetTexture = rtLeft;
                leftCam.Render();

                RenderTexture.active = rtLeft;
                textureLeft = new Texture2D(rtLeft.width, rtLeft.height, TextureFormat.RGB24, false);
                textureLeft.ReadPixels(new Rect(0, 0, rtLeft.width, rtLeft.height), 0, 0);
                textureLeft.Apply();

                // Capture right camera
                rtRight = new RenderTexture(rightCam.pixelWidth, rightCam.pixelHeight, 24);
                rightCam.targetTexture = rtRight;
                rightCam.Render();

                RenderTexture.active = rtRight;
                textureRight = new Texture2D(rtRight.width, rtRight.height, TextureFormat.RGB24, false);
                textureRight.ReadPixels(new Rect(0, 0, rtRight.width, rtRight.height), 0, 0);
                textureRight.Apply();

                // Encode to PNG
                byte[] imageLeftData = textureLeft.EncodeToPNG();
                byte[] imageRightData = textureRight.EncodeToPNG();

                // Send stereo pair
                string pairId = cameraPairId ?? _cameraPairId;
                return SendStereoPair(imageLeftData, imageRightData, pairId, _cameraLeftId, _cameraRightId, prompt);
            }
            catch (Exception ex)
            {
                LogError($"Error capturing/sending stereo pair: {ex.Message}");
                return false;
            }
            finally
            {
                // Cleanup
                if (leftCam != null) leftCam.targetTexture = null;
                if (rightCam != null) rightCam.targetTexture = null;
                RenderTexture.active = null;

                if (rtLeft != null) Destroy(rtLeft);
                if (rtRight != null) Destroy(rtRight);
                if (textureLeft != null) Destroy(textureLeft);
                if (textureRight != null) Destroy(textureRight);
            }
        }

        /// <summary>
        /// Send pre-encoded stereo image pair to Python StereoDetectionServer.
        /// </summary>
        /// <param name="imageLeftBytes">Left image PNG/JPG data</param>
        /// <param name="imageRightBytes">Right image PNG/JPG data</param>
        /// <param name="cameraPairId">Camera pair identifier</param>
        /// <param name="cameraLeftId">Left camera identifier</param>
        /// <param name="cameraRightId">Right camera identifier</param>
        /// <param name="prompt">Optional prompt/metadata</param>
        /// <returns>True if sent successfully</returns>
        public bool SendStereoPair(
            byte[] imageLeftBytes,
            byte[] imageRightBytes,
            string cameraPairId,
            string cameraLeftId,
            string cameraRightId,
            string prompt = ""
        )
        {
            if (!IsConnected)
            {
                LogWarning("Cannot send stereo pair - not connected");
                return false;
            }

            if (imageLeftBytes == null || imageLeftBytes.Length == 0)
            {
                LogError("Left image data is empty");
                return false;
            }

            if (imageRightBytes == null || imageRightBytes.Length == 0)
            {
                LogError("Right image data is empty");
                return false;
            }

            if (string.IsNullOrEmpty(cameraPairId))
            {
                LogWarning("Camera pair ID is empty, using 'stereo'");
                cameraPairId = "stereo";
            }

            try
            {
                // Validate sizes
                if (!UnityProtocol.IsValidImageSize(imageLeftBytes))
                {
                    LogError($"Left image size {imageLeftBytes.Length} exceeds protocol limit");
                    return false;
                }

                if (!UnityProtocol.IsValidImageSize(imageRightBytes))
                {
                    LogError($"Right image size {imageRightBytes.Length} exceeds protocol limit");
                    return false;
                }

                // Send stereo pair according to protocol
                // [cam_pair_id_len][cam_pair_id][camera_L_id_len][camera_L_id]
                // [camera_R_id_len][camera_R_id][prompt_len][prompt]
                // [image_L_len][image_L_data][image_R_len][image_R_data]

                Debug.Log($"[StereoImageSender] Sending stereo pair '{cameraPairId}' (L:{cameraLeftId}, R:{cameraRightId})");

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
                byte[] imageLLength = BitConverter.GetBytes(imageLeftBytes.Length);
                _stream.Write(imageLLength, 0, imageLLength.Length);
                _stream.Write(imageLeftBytes, 0, imageLeftBytes.Length);

                // Right image
                byte[] imageRLength = BitConverter.GetBytes(imageRightBytes.Length);
                _stream.Write(imageRLength, 0, imageRLength.Length);
                _stream.Write(imageRightBytes, 0, imageRightBytes.Length);

                _stream.Flush();

                string promptInfo = string.IsNullOrEmpty(prompt) ? "" : $" with prompt: '{prompt}'";
                LogVerbose(
                    $"Sent stereo pair '{cameraPairId}' " +
                    $"(L:{imageLeftBytes.Length}B, R:{imageRightBytes.Length}B){promptInfo}"
                );

                return true;
            }
            catch (Exception ex)
            {
                LogError($"Error sending stereo pair: {ex.Message}");
                _isConnected = false;
                return false;
            }
        }

        #endregion
    }
}
