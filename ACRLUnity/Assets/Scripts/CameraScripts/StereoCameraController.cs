using System;
using System.Net.Sockets;
using Core;
using PythonCommunication;
using UnityEngine;

namespace Vision
{
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
        private int _imageWidth = 1920; // Good balance of quality and performance

        [SerializeField]
        [Tooltip("Height of captured images in pixels")]
        private int _imageHeight = 1080; // Good balance of quality and performance

        [SerializeField]
        [Tooltip("JPEG compression quality (1-100)")]
        [Range(1, 100)]
        private int _JPEGQuality = 85;

        [Header("Streaming Mode")]
        [SerializeField]
        [Tooltip("Enable continuous streaming mode (default: true for Python VisionProcessor)")]
        private bool _enableStreaming = true;

        [SerializeField]
        [Tooltip("Streaming rate in FPS (frames per second)")]
        [Range(1f, 30f)]
        private float _streamingFPS = 5.0f;

        private float _stereoBaseline;
        private float _cameraFOV;

        // State
        private bool _isProcessing = false;
        private int _captureCounter = 0;

        // Streaming state
        private float _streamingInterval;
        private float _timeSinceLastCapture = 0f;

        // Camera
        private Camera _leftCamera;
        private Camera _rightCamera;
        private string _cameraPairId;

        // Helper variables
        private const string _logPrefix = "[STEREO_CAMERA_CONTROLLER]";

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

                Debug.Log(
                    $"{_logPrefix} LeftChild: {leftChild.name}, rightChild: {rightChild.name}"
                );

                _leftCamera = leftChild.GetComponent<Camera>();
                _rightCamera = rightChild.GetComponent<Camera>();

                if (_leftCamera != null && _rightCamera != null)
                {
                    Debug.Log(
                        $"{_logPrefix} Found cameras: {leftChild.name} and {rightChild.name}"
                    );
                    return;
                }
            }

            Debug.LogError(
                $"{_logPrefix} Failed to find left and right cameras. "
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
                Debug.LogError($"{_logPrefix} Both left and right cameras must be assigned!");
                enabled = false;
                return;
            }

            _cameraPairId = name;

            // Subscribe to depth results via UnifiedPythonReceiver
            if (UnifiedPythonReceiver.Instance != null)
            {
                UnifiedPythonReceiver.Instance.OnDepthResultReceived += HandleDepthResult;
            }
            else
            {
                Debug.LogWarning(
                    $"{_logPrefix} UnifiedPythonReceiver not found - results won't be received"
                );
            }

            // Log initialization
            float fov = GetCameraFOV();
            float baseline = GetStereoBaseline();
            Debug.Log(
                $"{_logPrefix} Initialized: {_cameraPairId}, Baseline={baseline}m, FOV={fov}°"
            );

            // Log camera positions for debugging
            if (_leftCamera != null && _rightCamera != null)
            {
                Vector3 leftPos = _leftCamera.transform.position;
                Vector3 rightPos = _rightCamera.transform.position;
                float actualDistance = Vector3.Distance(leftPos, rightPos);
                Debug.Log(
                    $"{_logPrefix} Left camera: {leftPos}, Right camera: {rightPos}, Distance: {actualDistance}m"
                );
            }

            // Initialize streaming mode
            _streamingInterval = 1.0f / _streamingFPS;
            if (_enableStreaming)
            {
                Debug.Log(
                    $"{_logPrefix} Streaming enabled at {_streamingFPS} FPS (interval: {_streamingInterval:F3}s)"
                );
            }
        }

        /// <summary>
        /// Unity Update - handle streaming mode
        /// </summary>
        private void Update()
        {
            if (!_enableStreaming)
                return;

            _timeSinceLastCapture += Time.deltaTime;

            if (_timeSinceLastCapture >= _streamingInterval)
            {
                // Only capture if not already processing
                if (!_isProcessing)
                {
                    Debug.Log($"{_logPrefix} Capture another image");
                    CaptureAndSendToServer(_cameraPairId);
                }
                _timeSinceLastCapture = 0f;
            }
        }

        /// <summary>
        /// Handles depth detection results
        /// </summary>
        private void HandleDepthResult(DepthResult result)
        {
            if (result == null & _isProcessing)
                return;

            // Check if this result is for our camera pair
            if (result.camera_id == _cameraPairId)
            {
                Debug.Log($"{_logPrefix} is processing set to false");
                _isProcessing = false;
            }
        }

        /// <summary>
        /// Capture stereo images and send directly to Python StereoImageServer via TCP.
        /// Called by PythonCommandHandler for the capture_stereo_images command.
        /// </summary>
        /// <param name="cameraId">Camera pair ID to use in the message</param>
        public void CaptureAndSendToServer(string cameraId)
        {
            if (_leftCamera == null || _rightCamera == null)
            {
                Debug.LogError($"{_logPrefix} Cameras not initialized");
                return;
            }

            try
            {
                // Capture both images
                byte[] leftImage = CaptureImage(_leftCamera);
                byte[] rightImage = CaptureImage(_rightCamera);

                if (leftImage == null || rightImage == null)
                {
                    Debug.LogError($"{_logPrefix} Failed to capture stereo images");
                    return;
                }

                _isProcessing = true;

                // Send via TCP to StereoImageServer (port 5006)
                SendStereoImagesTCP(cameraId, leftImage, rightImage);

                _captureCounter++;
                Debug.Log(
                    $"{_logPrefix} Sent stereo images for '{cameraId}' (L: {leftImage.Length} bytes, R: {rightImage.Length} bytes)"
                );
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error in CaptureAndSendToServer: {ex.Message}");
            }
        }

        /// <summary>
        /// Send stereo images to Python StereoImageServer via TCP.
        /// Protocol: [type:1][request_id:4][camera_pair_id][cam_L_id][cam_R_id][prompt][img_L][img_R]
        /// </summary>
        private void SendStereoImagesTCP(string cameraPairId, byte[] leftImage, byte[] rightImage)
        {
            try
            {
                using (TcpClient client = new TcpClient())
                {
                    client.Connect(
                        CommunicationConstants.SERVER_HOST,
                        CommunicationConstants.STEREO_DETECTION_PORT
                    );
                    NetworkStream stream = client.GetStream();

                    // Build message
                    byte[] message = EncodeStereoMessage(cameraPairId, leftImage, rightImage);

                    // Send
                    stream.Write(message, 0, message.Length);
                    stream.Flush();

                    Debug.Log($"{_logPrefix} Sent {message.Length} bytes to StereoImageServer");
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} TCP send failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Encode stereo image message in Protocol V2 format with metadata.
        /// </summary>
        private byte[] EncodeStereoMessage(string cameraPairId, byte[] leftImage, byte[] rightImage)
        {
            // Protocol V2 format:
            // [type:1][request_id:4][camera_pair_id_len:4][camera_pair_id:N]
            // [cam_L_id_len:4][cam_L_id:N][cam_R_id_len:4][cam_R_id:N]
            // [prompt_len:4][prompt:N][img_L_len:4][img_L_data:N][img_R_len:4][img_R_data:N]
            // [metadata_len:4][metadata_json:N]

            string camLId = cameraPairId + "_L";
            string camRId = cameraPairId + "_R";
            string prompt = ""; // Could be removed but Python protocol needs to be updated as well

            // Build metadata with camera transform (use left camera position)
            var metadata = new StereoMetadata
            {
                baseline = GetStereoBaseline(),
                fov = GetCameraFOV(),
                camera_position = new float[]
                {
                    _leftCamera.transform.position.x,
                    _leftCamera.transform.position.y,
                    _leftCamera.transform.position.z,
                },
                camera_rotation = new float[]
                {
                    _leftCamera.transform.eulerAngles.x, // pitch
                    _leftCamera.transform.eulerAngles.y, // yaw
                    _leftCamera.transform.eulerAngles.z, // roll
                },
            };
            string metadataJson = JsonUtility.ToJson(metadata);

            byte[] cameraPairIdBytes = System.Text.Encoding.UTF8.GetBytes(cameraPairId);
            byte[] camLIdBytes = System.Text.Encoding.UTF8.GetBytes(camLId);
            byte[] camRIdBytes = System.Text.Encoding.UTF8.GetBytes(camRId);
            byte[] promptBytes = System.Text.Encoding.UTF8.GetBytes(prompt);
            byte[] metadataBytes = System.Text.Encoding.UTF8.GetBytes(metadataJson);

            // Calculate total size
            int totalSize =
                1
                + 4 // type + request_id
                + 4
                + cameraPairIdBytes.Length
                + 4
                + camLIdBytes.Length
                + 4
                + camRIdBytes.Length
                + 4
                + promptBytes.Length
                + 4
                + leftImage.Length
                + 4
                + rightImage.Length
                + 4
                + metadataBytes.Length;

            byte[] message = new byte[totalSize];
            int offset = 0;

            // Type (7 = STEREO_IMAGE in Protocol V2)
            message[offset++] = 0x07;

            // Request ID (little-endian per Protocol V2)
            uint requestId = (uint)(Time.time * 1000) % uint.MaxValue;
            WriteUInt32LittleEndian(message, offset, requestId);
            offset += 4;

            // Camera pair ID
            WriteUInt32LittleEndian(message, offset, (uint)cameraPairIdBytes.Length);
            offset += 4;
            Array.Copy(cameraPairIdBytes, 0, message, offset, cameraPairIdBytes.Length);
            offset += cameraPairIdBytes.Length;

            // Left camera ID
            WriteUInt32LittleEndian(message, offset, (uint)camLIdBytes.Length);
            offset += 4;
            Array.Copy(camLIdBytes, 0, message, offset, camLIdBytes.Length);
            offset += camLIdBytes.Length;

            // Right camera ID
            WriteUInt32LittleEndian(message, offset, (uint)camRIdBytes.Length);
            offset += 4;
            Array.Copy(camRIdBytes, 0, message, offset, camRIdBytes.Length);
            offset += camRIdBytes.Length;

            // Prompt
            WriteUInt32LittleEndian(message, offset, (uint)promptBytes.Length);
            offset += 4;
            Array.Copy(promptBytes, 0, message, offset, promptBytes.Length);
            offset += promptBytes.Length;

            // Left image
            WriteUInt32LittleEndian(message, offset, (uint)leftImage.Length);
            offset += 4;
            Array.Copy(leftImage, 0, message, offset, leftImage.Length);
            offset += leftImage.Length;

            // Right image
            WriteUInt32LittleEndian(message, offset, (uint)rightImage.Length);
            offset += 4;
            Array.Copy(rightImage, 0, message, offset, rightImage.Length);
            offset += rightImage.Length;

            // Metadata JSON
            WriteUInt32LittleEndian(message, offset, (uint)metadataBytes.Length);
            offset += 4;
            Array.Copy(metadataBytes, 0, message, offset, metadataBytes.Length);

            return message;
        }

        /// <summary>
        /// Write uint32 in little-endian format (Protocol V2 standard)
        /// </summary>
        private void WriteUInt32LittleEndian(byte[] buffer, int offset, uint value)
        {
            buffer[offset] = (byte)value;
            buffer[offset + 1] = (byte)(value >> 8);
            buffer[offset + 2] = (byte)(value >> 16);
            buffer[offset + 3] = (byte)(value >> 24);
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
                byte[] bytes = texture.EncodeToJPG(_JPEGQuality);
                return bytes;
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error capturing image: {ex.Message}");
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
            // Unsubscribe from depth results
            if (UnifiedPythonReceiver.Instance != null)
            {
                UnifiedPythonReceiver.Instance.OnDepthResultReceived -= HandleDepthResult;
            }

            if (_captureCounter > 0)
            {
                Debug.Log($"{_logPrefix} Destroyed after {_captureCounter} captures");
            }
        }
    }
}
