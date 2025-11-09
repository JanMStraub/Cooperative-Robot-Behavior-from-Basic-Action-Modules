using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using PythonCommunication.Core;

namespace PythonCommunication
{
    /// <summary>
    /// Data structure for stereo detection results with 3D depth information
    /// </summary>
    [System.Serializable]
    public class DepthResult
    {
        public bool success;
        public string camera_id;
        public string timestamp;
        public ObjectDetection[] detections;
        public DepthMetadata metadata;
    }

    [System.Serializable]
    public class ObjectDetection
    {
        public string color;
        public float confidence;
        public DetectionBoundingBox bbox;
        public DetectionPixelCoords pixel_center;
        public Detection3DPosition world_position;
        public float depth_m;
        public float disparity;
    }

    [System.Serializable]
    public class DetectionBoundingBox
    {
        public int x;
        public int y;
        public int width;
        public int height;
    }

    [System.Serializable]
    public class DetectionPixelCoords
    {
        public int x;
        public int y;
    }

    [System.Serializable]
    public class Detection3DPosition
    {
        public float x;
        public float y;
        public float z;
    }

    [System.Serializable]
    public class DepthMetadata
    {
        public float processing_time_seconds;
        public string prompt;
        public float camera_baseline_m;
        public float camera_fov_deg;
        public string detection_mode;
    }

    /// <summary>
    /// Receives stereo detection results with 3D depth information from Python StereoDetector via TCP.
    /// Connects to port 5007 (stereo detection results).
    /// </summary>
    public class DepthResultsReceiver : TCPClientBase
    {
        public static DepthResultsReceiver Instance { get; private set; }

        [Header("Result Processing")]
        [Tooltip("Log all received depth results to console")]
        [SerializeField]
        private bool _logResults = true;

        // Events
        public event Action<DepthResult> OnDepthResultReceived;

        // State
        private Thread _receiveThread;
        private Queue<DepthResult> _resultQueue = new Queue<DepthResult>();
        private readonly object _queueLock = new object();

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

                // Set default port if not configured
                if (_serverPort == 0)
                {
                    _serverPort = 5006; // ResultsServer default port (stereo depth results sent via ResultsServer)
                }
            }
            else
            {
                Destroy(gameObject);
                return;
            }
        }

        #endregion

        #region Unity Lifecycle

        /// <summary>
        /// Process queued results on main thread
        /// </summary>
        protected override void Update()
        {
            base.Update(); // Handle auto-reconnect

            // Process all queued results
            lock (_queueLock)
            {
                while (_resultQueue.Count > 0)
                {
                    DepthResult result = _resultQueue.Dequeue();
                    ProcessResult(result);
                }
            }
        }

        protected override void OnDestroy()
        {
            if (Instance == this)
            {
                base.OnDestroy();
                Instance = null;
            }
        }

        #endregion

        #region TCPClientBase Implementation

        protected override string LogPrefix => "DepthResultsReceiver";

        protected override void OnConnected()
        {
            // Start receive thread
            _receiveThread = new Thread(ReceiveLoop);
            _receiveThread.IsBackground = true;
            _receiveThread.Start();
        }

        protected override void OnConnectionFailed(Exception exception)
        {
            // Connection failure already logged by base class
            if (_autoReconnect)
            {
                StartCoroutine(ReconnectCoroutine());
            }
        }

        protected override void OnDisconnecting()
        {
            // Stop receive thread
            if (_receiveThread != null && _receiveThread.IsAlive)
            {
                _receiveThread.Join(1000); // Wait up to 1 second
            }
        }

        protected override void OnDisconnected()
        {
            // Clear result queue
            lock (_queueLock)
            {
                _resultQueue.Clear();
            }
        }

        #endregion

        #region Data Reception

        /// <summary>
        /// Background thread loop for receiving depth results
        /// </summary>
        private void ReceiveLoop()
        {
            byte[] lengthBuffer = new byte[UnityProtocol.INT_SIZE];

            try
            {
                while (_shouldRun && _isConnected)
                {
                    // Read message length (4 bytes)
                    int bytesRead = ReadExactly(_stream, lengthBuffer, UnityProtocol.INT_SIZE);
                    if (bytesRead < UnityProtocol.INT_SIZE)
                    {
                        LogWarning("Connection closed by server");
                        break;
                    }

                    int dataLength = BitConverter.ToInt32(lengthBuffer, 0);

                    // Validate length
                    if (dataLength <= 0 || dataLength > UnityProtocol.MAX_IMAGE_SIZE)
                    {
                        LogError($"Invalid data length received: {dataLength}");
                        break;
                    }

                    // Read message data
                    byte[] dataBuffer = new byte[dataLength];
                    bytesRead = ReadExactly(_stream, dataBuffer, dataLength);
                    if (bytesRead < dataLength)
                    {
                        LogWarning("Incomplete data received");
                        break;
                    }

                    // Decode using protocol
                    try
                    {
                        byte[] fullMessage = new byte[UnityProtocol.INT_SIZE + dataLength];
                        Buffer.BlockCopy(lengthBuffer, 0, fullMessage, 0, UnityProtocol.INT_SIZE);
                        Buffer.BlockCopy(dataBuffer, 0, fullMessage, UnityProtocol.INT_SIZE, dataLength);

                        string json = UnityProtocol.DecodeResultMessage(fullMessage);

                        // Parse JSON
                        DepthResult result = JsonUtility.FromJson<DepthResult>(json);

                        // Queue for processing on main thread
                        lock (_queueLock)
                        {
                            _resultQueue.Enqueue(result);
                        }
                    }
                    catch (Exception parseEx)
                    {
                        LogError($"Failed to parse depth result: {parseEx.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                if (_shouldRun && _isConnected)
                {
                    LogError($"Error in receive loop: {ex.Message}");
                }
            }
            finally
            {
                _isConnected = false;

                // Trigger reconnect if enabled
                if (_shouldRun && _autoReconnect)
                {
                    LogVerbose("Connection lost - will attempt to reconnect");
                }
            }
        }

        /// <summary>
        /// Coroutine to handle reconnection after connection loss
        /// </summary>
        private IEnumerator ReconnectCoroutine()
        {
            yield return new WaitForSeconds(_reconnectInterval);

            if (_shouldRun && !_isConnected)
            {
                Connect();
            }
        }

        #endregion

        #region Result Processing

        /// <summary>
        /// Process a received depth result (called on main thread)
        /// </summary>
        private void ProcessResult(DepthResult result)
        {
            if (result == null)
                return;

            if (_logResults)
            {
                string durationInfo = result.metadata != null ? $"{result.metadata.processing_time_seconds:F2}s" : "?";
                int detectionCount = result.detections != null ? result.detections.Length : 0;

                Log($"📏 Depth Result for {result.camera_id}: {detectionCount} objects detected in {durationInfo}");

                // Log each detection with 3D position
                if (result.detections != null)
                {
                    foreach (var detection in result.detections)
                    {
                        if (detection.world_position != null)
                        {
                            Log($"  - {detection.color.ToUpper()} cube at ({detection.world_position.x:F3}, {detection.world_position.y:F3}, {detection.world_position.z:F3})m, depth={detection.depth_m:F3}m, confidence={detection.confidence:F2}");
                        }
                        else
                        {
                            Log($"  - {detection.color.ToUpper()} cube at pixel ({detection.pixel_center.x}, {detection.pixel_center.y}), confidence={detection.confidence:F2}");
                        }
                    }
                }
            }

            // Fire event for subscribers
            try
            {
                OnDepthResultReceived?.Invoke(result);
            }
            catch (Exception ex)
            {
                LogError($"Error in OnDepthResultReceived event handler: {ex.Message}");
            }
        }

        #endregion
    }
}
