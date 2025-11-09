using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using UnityEngine;
using PythonCommunication.Core;
using Vision;

namespace PythonCommunication
{
    /// <summary>
    /// TCP client that receives object detection results from Python DetectionServer (port 5007).
    /// Automatically converts pixel coordinates to Unity world coordinates using raycasting.
    /// </summary>
    public class DetectionResultsReceiver : TCPClientBase
    {
        // Singleton instance
        public static DetectionResultsReceiver Instance { get; private set; }

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

        private void Awake()
        {
            // Singleton pattern
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);

                // Set default port if not configured
                if (_serverPort == 0)
                {
                    _serverPort = 5007; // DetectionServer default port
                }
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
        /// Gets a registered camera by ID from CameraManager
        /// </summary>
        private Camera GetCameraById(string cameraId)
        {
            if (CameraManager.Instance == null)
            {
                Debug.LogWarning(
                    "[DETECTION_RESULT_RECEIVER] CameraManager not initialized. Create a CameraManager GameObject in the scene."
                );
                return null;
            }

            return CameraManager.Instance.GetCamera(cameraId);
        }

        protected override void OnConnected()
        {
            Debug.Log(
                $"[DETECTION_RESULT_RECEIVER] Connected to DetectionServer at {_serverHost}:{_serverPort}"
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
                $"[DETECTION_RESULT_RECEIVER] Connection failed: {error.Message}. Will retry..."
            );
        }

        protected override void OnDisconnecting()
        {
            Debug.Log("[DETECTION_RESULT_RECEIVER] Disconnecting from DetectionServer");

            // Stop receive thread
            if (_receiveThread != null && _receiveThread.IsAlive)
            {
                _receiveThread.Join(1000); // Wait up to 1 second
            }
        }

        protected override void OnDisconnected()
        {
            Debug.Log("[DETECTION_RESULT_RECEIVER] Disconnected from DetectionServer");
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
                    // Read message length (4 bytes, little-endian to match Python protocol)
                    byte[] lengthBuffer = new byte[4];
                    int bytesRead = ReadExactly(_stream, lengthBuffer, 4);

                    if (bytesRead < 4)
                    {
                        Debug.LogWarning("[DETECTION_RESULT_RECEIVER] Connection closed by server");
                        break;
                    }

                    int messageLength = BitConverter.ToInt32(lengthBuffer, 0);

                    if (messageLength <= 0 || messageLength > MAX_JSON_LENGTH)
                    {
                        Debug.LogError($"[DETECTION_RESULT_RECEIVER] Invalid message length: {messageLength}");
                        continue;
                    }

                    // Read message data
                    byte[] messageData = new byte[messageLength];
                    bytesRead = ReadExactly(_stream, messageData, messageLength);

                    if (bytesRead < messageLength)
                    {
                        Debug.LogWarning("[DETECTION_RESULT_RECEIVER] Incomplete message received");
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
                Debug.LogWarning($"[DETECTION_RESULT_RECEIVER] Connection lost: {ex.Message}");
                _isConnected = false;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[DETECTION_RESULT_RECEIVER] Error in receive loop: {ex.Message}");
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
                    Debug.LogError("[DETECTION_RESULT_RECEIVER] Invalid data: too short");
                    return;
                }

                // Read JSON length (little-endian to match Python protocol)
                int jsonLength = BitConverter.ToInt32(data, 0);

                if (jsonLength <= 0 || jsonLength > MAX_JSON_LENGTH)
                {
                    Debug.LogError(
                        $"[DETECTION_RESULT_RECEIVER] Invalid JSON length: {jsonLength}"
                    );
                    return;
                }

                if (data.Length < 4 + jsonLength)
                {
                    Debug.LogError(
                        $"[DETECTION_RESULT_RECEIVER] Incomplete data: expected {4 + jsonLength} bytes, got {data.Length}"
                    );
                    return;
                }

                // Extract JSON string
                string jsonString = Encoding.UTF8.GetString(data, 4, jsonLength);

                // Parse JSON
                DetectionResult result = JsonUtility.FromJson<DetectionResult>(jsonString);

                if (result == null)
                {
                    Debug.LogError("[DETECTION_RESULT_RECEIVER] Failed to parse detection result");
                    return;
                }

                // Process result
                ProcessDetectionResult(result);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[DETECTION_RESULT_RECEIVER] Error handling data: {ex.Message}");
            }
        }

        /// <summary>
        /// Processes a detection result and converts pixel coordinates to world coordinates
        /// </summary>
        private void ProcessDetectionResult(DetectionResult result)
        {
            Debug.Log(
                $"[DETECTION_RESULT_RECEIVER] Received detection: {result.detections?.Length ?? 0} cube(s) from {result.camera_id}"
            );

            // Fire raw result event
            OnDetectionReceived?.Invoke(result);

            // Get camera for this result
            Camera sourceCamera = GetCameraById(result.camera_id);

            if (sourceCamera == null)
            {
                Debug.LogWarning(
                    $"[DETECTION_RESULT_RECEIVER] Cannot convert to world coordinates: camera '{result.camera_id}' not found"
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
                            $"[DETECTION_RESULT_RECEIVER] {detection.color} cube: using stereo depth world position {worldPos}"
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
                "[DETECTION_RESULT_RECEIVER] GetLatestDetectionForCamera not implemented yet. Use events instead."
            );
            return null;
        }

    }
}
