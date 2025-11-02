using System;
using UnityEngine;
using LLMCommunication.Core;

namespace LLMCommunication
{
    /// <summary>
    /// Sends camera images to Python StreamingServer via TCP for LLM vision processing.
    /// Refactored to use TCPClientBase and UnityProtocol.
    /// Supports both on-demand sending and continuous streaming.
    /// </summary>
    public class ImageSender : TCPClientBase
    {
        public static ImageSender Instance { get; private set; }

        [Header("Streaming Settings (Optional)")]
        [Tooltip("Enable continuous streaming mode")]
        [SerializeField]
        private bool _enableStreaming = false;

        [Header("Image Encoding Settings")]
        [Tooltip("JPEG quality (1-100, higher is better quality but larger file)")]
        [SerializeField]
        [Range(1, 100)]
        private int _jpegQuality = 85;

        [Header("Streaming Settings (Optional)")]
        [Tooltip("Time between streamed images in seconds")]
        [SerializeField]
        private float _sendInterval = 0.2f;

        [Tooltip("Camera to stream from (only used in streaming mode)")]
        [SerializeField]
        private Camera _streamCamera;

        [Tooltip("Camera identifier for streaming")]
        [SerializeField]
        private string _streamCameraId = "Main";

        [Tooltip("Prompt for streamed images")]
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

            // Set default port for ImageSender
            if (_serverPort == 0)
            {
                _serverPort = 5005; // StreamingServer default port
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
            if (_enableStreaming && IsConnected && _streamCamera != null)
            {
                _streamTimer += Time.deltaTime;
                if (_streamTimer >= _sendInterval)
                {
                    CaptureAndSendCamera(_streamCamera, _streamCameraId, _streamPrompt);
                    _streamTimer = 0f;
                }
            }
        }

        #endregion

        #region TCPClientBase Implementation

        protected override string LogPrefix => "ImageSender";

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
        /// Send pre-encoded image data to Python StreamingServer.
        /// Uses UnityProtocol for encoding.
        /// </summary>
        /// <param name="imageBytes">Encoded image data (PNG/JPG)</param>
        /// <param name="cameraId">Camera identifier</param>
        /// <param name="prompt">Optional LLM prompt</param>
        /// <returns>True if sent successfully</returns>
        public bool SendImageData(byte[] imageBytes, string cameraId, string prompt = "")
        {
            // Pre-flight validation
            if (imageBytes == null || imageBytes.Length == 0)
            {
                LogError("Cannot send empty image data");
                return false;
            }

            // Check connection and attempt reconnect if needed
            if (!IsConnected)
            {
                LogWarning("Not connected - attempting to connect");
                Connect();

                // Check again after connect attempt
                if (!IsConnected)
                {
                    LogError("Cannot send image - connection failed");
                    return false;
                }
            }

            if (string.IsNullOrEmpty(cameraId))
            {
                LogWarning("Camera ID is empty, using 'Unknown'");
                cameraId = "Unknown";
            }

            try
            {
                // Validate with protocol
                if (!UnityProtocol.IsValidImageSize(imageBytes))
                {
                    LogError($"Image size {imageBytes.Length} exceeds protocol limit");
                    return false;
                }

                if (!UnityProtocol.IsValidStringLength(cameraId))
                {
                    LogError($"Camera ID '{cameraId}' exceeds protocol string length limit");
                    return false;
                }

                if (!UnityProtocol.IsValidStringLength(prompt))
                {
                    LogError($"Prompt exceeds protocol string length limit");
                    return false;
                }

                // Verify connection is alive before writing
                if (!VerifyConnection())
                {
                    LogError("Connection verification failed - attempting reconnect");
                    Connect();

                    if (!VerifyConnection())
                    {
                        LogError("Reconnection failed");
                        return false;
                    }
                }

                // Send data piece by piece (streaming protocol)
                // Format: [camera_id_len][camera_id][prompt_len][prompt][image_len][image_data]

                // Send camera ID
                byte[] idBytes = System.Text.Encoding.UTF8.GetBytes(cameraId);
                byte[] idLength = BitConverter.GetBytes(idBytes.Length);
                _stream.Write(idLength, 0, idLength.Length);
                _stream.Write(idBytes, 0, idBytes.Length);

                // Send prompt
                byte[] promptBytes = System.Text.Encoding.UTF8.GetBytes(prompt ?? "");
                byte[] promptLength = BitConverter.GetBytes(promptBytes.Length);
                _stream.Write(promptLength, 0, promptLength.Length);
                _stream.Write(promptBytes, 0, promptBytes.Length);

                // Send image
                byte[] imageLength = BitConverter.GetBytes(imageBytes.Length);
                _stream.Write(imageLength, 0, imageLength.Length);
                _stream.Write(imageBytes, 0, imageBytes.Length);

                _stream.Flush();

                string promptInfo = string.IsNullOrEmpty(prompt) ? "" : $" with prompt: '{prompt}'";
                LogVerbose($"Sent {imageBytes.Length} bytes for camera '{cameraId}'{promptInfo}");

                return true;
            }
            catch (System.IO.IOException ioEx)
            {
                LogError($"Network error sending image: {ioEx.Message}");
                _isConnected = false;
                // Trigger reconnection
                return false;
            }
            catch (System.Net.Sockets.SocketException sockEx)
            {
                LogError($"Socket error sending image: {sockEx.Message}");
                _isConnected = false;
                // Trigger reconnection
                return false;
            }
            catch (Exception ex)
            {
                LogError($"Unexpected error sending image: {ex.Message}");
                _isConnected = false;
                return false;
            }
        }

        /// <summary>
        /// Capture image from camera and send to Python StreamingServer.
        /// </summary>
        /// <param name="cam">Camera to capture from</param>
        /// <param name="cameraId">Camera identifier</param>
        /// <param name="prompt">Optional LLM prompt</param>
        /// <returns>True if captured and sent successfully</returns>
        public bool CaptureAndSendCamera(Camera cam, string cameraId, string prompt = "")
        {
            if (cam == null)
            {
                LogError("Camera is null");
                return false;
            }

            if (!IsConnected)
            {
                LogWarning("Cannot send image - not connected");
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

                // Read pixels into Texture2D
                RenderTexture.active = rt;
                texture = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
                texture.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                texture.Apply();

                // Encode to JPEG
                byte[] imageData = texture.EncodeToJPG(_jpegQuality);

                // Send using protocol
                return SendImageData(imageData, cameraId, prompt);
            }
            catch (Exception ex)
            {
                LogError($"Error capturing/sending image: {ex.Message}");
                return false;
            }
            finally
            {
                // Cleanup
                if (cam != null)
                {
                    cam.targetTexture = null;
                }
                RenderTexture.active = null;

                if (rt != null)
                {
                    Destroy(rt);
                }
                if (texture != null)
                {
                    Destroy(texture);
                }
            }
        }

        #endregion
    }
}
