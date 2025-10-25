using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using LLMCommunication.Core;

namespace LLMCommunication
{
    /// <summary>
    /// TCP client that receives object detection results from Python DetectionServer (port 5007).
    /// Automatically converts pixel coordinates to Unity world coordinates using raycasting.
    /// </summary>
    public class DetectionResultsReceiver : TCPClientBase
    {
        // Singleton instance
        public static DetectionResultsReceiver Instance { get; private set; }

        [Header("Connection Settings")]
        [SerializeField]
        [Tooltip("Python server host (default: localhost)")]
        private new string _serverHost = "127.0.0.1";

        [SerializeField]
        [Tooltip("Python server port (default: 5007)")]
        private new int _serverPort = 5007;

        [Header("Camera Management")]
        [SerializeField]
        [Tooltip("Dictionary of camera IDs to Camera components for raycasting")]
        private CameraMapping[] _cameraMappings = Array.Empty<CameraMapping>();

        // Camera lookup dictionary
        private Dictionary<string, Camera> _cameraDict;

        // Background thread for receiving data
        private Thread _receiveThread;

        // Thread-safe queue for processing messages on main thread
        private readonly Queue<byte[]> _messageQueue = new Queue<byte[]>();
        private readonly object _queueLock = new object();

        // Events
        public event Action<DetectionResult> OnDetectionReceived;
        public event Action<DetectionResultWithWorld> OnDetectionWithWorldReceived;

        // Protocol constants (must match Python UnityProtocol)
        private const int MAX_JSON_LENGTH = 10 * 1024 * 1024; // 10MB max

        // Log prefix for TCPClientBase
        protected override string LogPrefix => "DetectionResultsReceiver";

        /// <summary>
        /// Camera mapping for inspector configuration
        /// </summary>
        [Serializable]
        public class CameraMapping
        {
            public string cameraId;
            public Camera camera;
        }

        private void Awake()
        {
            // Singleton pattern
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);

                // Build camera dictionary
                BuildCameraDictionary();
            }
            else
            {
                Destroy(gameObject);
            }
        }

        protected override void Start()
        {
            base.Start();
        }

        protected override void Update()
        {
            base.Update();

            // Process queued messages on main thread
            lock (_queueLock)
            {
                while (_messageQueue.Count > 0)
                {
                    byte[] message = _messageQueue.Dequeue();
                    HandleIncomingData(message);
                }
            }
        }

        protected override void OnDestroy()
        {
            base.OnDestroy();

            if (Instance == this)
            {
                Instance = null;
            }
        }

        /// <summary>
        /// Builds the camera lookup dictionary from inspector mappings
        /// </summary>
        private void BuildCameraDictionary()
        {
            _cameraDict = new Dictionary<string, Camera>();

            if (_cameraMappings != null)
            {
                foreach (var mapping in _cameraMappings)
                {
                    if (
                        !string.IsNullOrEmpty(mapping.cameraId)
                        && mapping.camera != null
                        && !_cameraDict.ContainsKey(mapping.cameraId)
                    )
                    {
                        _cameraDict[mapping.cameraId] = mapping.camera;
                    }
                }
            }

            Debug.Log(
                $"[DetectionResultsReceiver] Registered {_cameraDict.Count} camera(s) for detection"
            );
        }

        /// <summary>
        /// Registers a camera for detection results
        /// </summary>
        /// <param name="cameraId">Camera identifier</param>
        /// <param name="camera">Camera component</param>
        public void RegisterCamera(string cameraId, Camera camera)
        {
            if (string.IsNullOrEmpty(cameraId) || camera == null)
            {
                Debug.LogWarning("[DetectionResultsReceiver] Invalid camera registration");
                return;
            }

            if (_cameraDict == null)
            {
                _cameraDict = new Dictionary<string, Camera>();
            }

            _cameraDict[cameraId] = camera;
            Debug.Log($"[DetectionResultsReceiver] Registered camera: {cameraId}");
        }

        /// <summary>
        /// Gets a registered camera by ID
        /// </summary>
        private Camera GetCameraById(string cameraId)
        {
            if (_cameraDict != null && _cameraDict.TryGetValue(cameraId, out Camera camera))
            {
                return camera;
            }

            // Try to find camera by name if not in dictionary
            GameObject cameraObj = GameObject.Find(cameraId);
            if (cameraObj != null)
            {
                Camera foundCamera = cameraObj.GetComponent<Camera>();
                if (foundCamera != null)
                {
                    Debug.Log(
                        $"[DetectionResultsReceiver] Found camera by name: {cameraId}, registering it"
                    );
                    RegisterCamera(cameraId, foundCamera);
                    return foundCamera;
                }
            }

            Debug.LogWarning(
                $"[DetectionResultsReceiver] Camera not found for ID: {cameraId}. Register it first or add to inspector."
            );
            return null;
        }

        protected override void OnConnected()
        {
            Debug.Log(
                $"[DetectionResultsReceiver] Connected to DetectionServer at {_serverHost}:{_serverPort}"
            );

            // Start background thread to receive data
            _receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = "DetectionReceiver"
            };
            _receiveThread.Start();
        }

        protected override void OnConnectionFailed(Exception error)
        {
            Debug.LogWarning(
                $"[DetectionResultsReceiver] Connection failed: {error.Message}. Will retry..."
            );
        }

        protected override void OnDisconnecting()
        {
            Debug.Log("[DetectionResultsReceiver] Disconnecting from DetectionServer");

            // Stop receive thread
            if (_receiveThread != null && _receiveThread.IsAlive)
            {
                _receiveThread.Join(1000); // Wait up to 1 second
            }
        }

        protected override void OnDisconnected()
        {
            Debug.Log("[DetectionResultsReceiver] Disconnected from DetectionServer");
        }

        /// <summary>
        /// Background thread loop for receiving detection results
        /// </summary>
        private void ReceiveLoop()
        {
            try
            {
                while (IsConnected && _shouldRun)
                {
                    // Read message length (4 bytes, big-endian)
                    byte[] lengthBuffer = new byte[4];
                    int bytesRead = ReadExactly(_stream, lengthBuffer, 4);

                    if (bytesRead < 4)
                    {
                        Debug.LogWarning("[DetectionResultsReceiver] Connection closed by server");
                        break;
                    }

                    int messageLength = (lengthBuffer[0] << 24) | (lengthBuffer[1] << 16) |
                                       (lengthBuffer[2] << 8) | lengthBuffer[3];

                    if (messageLength <= 0 || messageLength > MAX_JSON_LENGTH)
                    {
                        Debug.LogError($"[DetectionResultsReceiver] Invalid message length: {messageLength}");
                        continue;
                    }

                    // Read message data
                    byte[] messageData = new byte[messageLength];
                    bytesRead = ReadExactly(_stream, messageData, messageLength);

                    if (bytesRead < messageLength)
                    {
                        Debug.LogWarning("[DetectionResultsReceiver] Incomplete message received");
                        break;
                    }

                    // Combine length + data for HandleIncomingData
                    byte[] fullMessage = new byte[4 + messageLength];
                    Array.Copy(lengthBuffer, 0, fullMessage, 0, 4);
                    Array.Copy(messageData, 0, fullMessage, 4, messageLength);

                    // Queue for processing on main thread
                    lock (_queueLock)
                    {
                        _messageQueue.Enqueue(fullMessage);
                    }
                }
            }
            catch (System.IO.IOException ex)
            {
                Debug.LogWarning($"[DetectionResultsReceiver] Connection lost: {ex.Message}");
                _isConnected = false;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[DetectionResultsReceiver] Error in receive loop: {ex.Message}");
                _isConnected = false;
            }
        }

        /// <summary>
        /// Handles incoming detection result data (called on main thread)
        /// </summary>
        private void HandleIncomingData(byte[] data)
        {
            try
            {
                // Decode result message using Unity protocol
                // Format: [json_length (4 bytes)][json_data]
                if (data.Length < 4)
                {
                    Debug.LogError("[DetectionResultsReceiver] Invalid data: too short");
                    return;
                }

                // Read JSON length (big-endian)
                int jsonLength =
                    (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];

                if (jsonLength <= 0 || jsonLength > MAX_JSON_LENGTH)
                {
                    Debug.LogError(
                        $"[DetectionResultsReceiver] Invalid JSON length: {jsonLength}"
                    );
                    return;
                }

                if (data.Length < 4 + jsonLength)
                {
                    Debug.LogError(
                        $"[DetectionResultsReceiver] Incomplete data: expected {4 + jsonLength} bytes, got {data.Length}"
                    );
                    return;
                }

                // Extract JSON string
                string jsonString = Encoding.UTF8.GetString(data, 4, jsonLength);

                // Parse JSON
                DetectionResult result = JsonUtility.FromJson<DetectionResult>(jsonString);

                if (result == null)
                {
                    Debug.LogError("[DetectionResultsReceiver] Failed to parse detection result");
                    return;
                }

                // Process result
                ProcessDetectionResult(result);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[DetectionResultsReceiver] Error handling data: {ex.Message}");
            }
        }

        /// <summary>
        /// Processes a detection result and converts pixel coordinates to world coordinates
        /// </summary>
        private void ProcessDetectionResult(DetectionResult result)
        {
            Debug.Log(
                $"[DetectionResultsReceiver] Received detection: {result.detections?.Length ?? 0} cube(s) from {result.camera_id}"
            );

            // Fire raw result event
            OnDetectionReceived?.Invoke(result);

            // Get camera for this result
            Camera sourceCamera = GetCameraById(result.camera_id);

            if (sourceCamera == null)
            {
                Debug.LogWarning(
                    $"[DetectionResultsReceiver] Cannot convert to world coordinates: camera '{result.camera_id}' not found"
                );
                return;
            }

            // Convert all detections to world coordinates
            List<DetectedCubeWithWorld> cubesWithWorld = new List<DetectedCubeWithWorld>();

            if (result.detections != null)
            {
                foreach (var detection in result.detections)
                {
                    Vector3 worldPos = Vector3.zero;
                    GameObject hitObject = null;
                    bool hasWorldPos = false;

                    // Check if stereo depth estimation provided world position
                    if (detection.world_position != null && detection.world_position.IsValid())
                    {
                        worldPos = detection.world_position.ToVector3();
                        hasWorldPos = true;
                        Debug.Log(
                            $"[DetectionResultsReceiver] {detection.color} cube: using stereo depth world position {worldPos}"
                        );
                    }
                    else
                    {
                        // Fall back to raycasting
                        hasWorldPos = detection.TryGetWorldPosition(
                            sourceCamera,
                            result.image_width,
                            result.image_height,
                            out worldPos,
                            out hitObject
                        );
                    }

                    cubesWithWorld.Add(
                        new DetectedCubeWithWorld(detection, worldPos, hitObject, hasWorldPos)
                    );

                    if (hasWorldPos)
                    {
                        Debug.Log(
                            $"  • {detection.color.ToUpper()} cube: pixel ({detection.center_px.x}, {detection.center_px.y}) → world {worldPos} (hit: {hitObject?.name ?? "none"})"
                        );
                    }
                    else
                    {
                        Debug.LogWarning(
                            $"  • {detection.color.ToUpper()} cube: pixel ({detection.center_px.x}, {detection.center_px.y}) → no raycast hit"
                        );
                    }
                }
            }

            // Create enhanced result with world coordinates
            DetectionResultWithWorld resultWithWorld = new DetectionResultWithWorld(
                result,
                sourceCamera,
                cubesWithWorld.ToArray()
            );

            // Fire enhanced result event
            OnDetectionWithWorldReceived?.Invoke(resultWithWorld);
        }

        /// <summary>
        /// Gets the most recent detection result for a specific camera
        /// </summary>
        public DetectionResult GetLatestDetectionForCamera(string cameraId)
        {
            // This would require storing results in a dictionary
            // For now, users should subscribe to events
            Debug.LogWarning(
                "[DetectionResultsReceiver] GetLatestDetectionForCamera not implemented yet. Use events instead."
            );
            return null;
        }

#if UNITY_EDITOR
        private void OnValidate()
        {
            // Validate camera mappings
            if (_cameraMappings != null)
            {
                bool hasDuplicates = _cameraMappings
                    .GroupBy(m => m.cameraId)
                    .Any(g => g.Count() > 1);

                if (hasDuplicates)
                {
                    Debug.LogWarning(
                        "[DetectionResultsReceiver] Duplicate camera IDs detected in mappings!"
                    );
                }
            }
        }
#endif
    }
}
