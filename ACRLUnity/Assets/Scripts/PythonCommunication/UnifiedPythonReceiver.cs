using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using Core;
using PythonCommunication.Core;
using UnityEngine;
using Vision;

namespace PythonCommunication
{
    /// <summary>
    /// Unified receiver for all Python result communication.
    /// Handles LLM results (port 5010) and Depth results (port 5007).
    /// Replaces LLMResultsReceiver, DepthResultsReceiver, and DetectionResultsReceiver.
    /// </summary>
    public class UnifiedPythonReceiver : MonoBehaviour
    {
        public static UnifiedPythonReceiver Instance { get; private set; }

        [Header("Result Processing")]
        [Tooltip("Log all received results to console")]
        [SerializeField]
        private bool _logResults = true;

        // Events - All existing events preserved
        public event Action<LLMResult> OnLLMResultReceived;
        public event Action<DepthResult> OnDepthResultReceived;

        // Internal connection handlers (one per port)
        private ResultsConnection _resultsConnection; // Port 5010 (LLM results)
        private DetectionConnection _detectionConnection; // Port 5007 (Depth results)

        // Helper variable
        private const string _logPrefix = "[UNIFIED_PYTHON_RECEIVER]";

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
                GameObject resultsConnObj = new GameObject("ResultsConnection");
                resultsConnObj.transform.SetParent(transform);
                _resultsConnection = resultsConnObj.AddComponent<ResultsConnection>();
                _resultsConnection.Initialize(this);

                GameObject detectionConnObj = new GameObject("DetectionConnection");
                detectionConnObj.transform.SetParent(transform);
                _detectionConnection = detectionConnObj.AddComponent<DetectionConnection>();
                _detectionConnection.Initialize(this);

                Debug.Log(
                    $"{_logPrefix} ✓ Initialized with two connection handlers (ports 5010, 5007)"
                );
            }
            else
            {
                Destroy(gameObject);
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

        #region Public API

        /// <summary>
        /// Check if results connection (port 5010) is active
        /// </summary>
        public bool IsResultsConnected =>
            _resultsConnection != null && _resultsConnection.IsConnected;

        /// <summary>
        /// Check if detection connection (port 5007) is active
        /// </summary>
        public bool IsDetectionConnected =>
            _detectionConnection != null && _detectionConnection.IsConnected;

        /// <summary>
        /// Check if both connections are active
        /// </summary>
        public bool IsFullyConnected => IsResultsConnected && IsDetectionConnected;

        #endregion

        #region Internal Event Routing

        /// <summary>
        /// Route LLM result from connection to external subscribers
        /// </summary>
        internal void RouteLLMResult(LLMResult result)
        {
            if (_logResults)
            {
                string modelInfo = result.metadata?.model ?? "unknown";
                string durationInfo =
                    result.metadata != null ? $"{result.metadata.duration_seconds:F2}s" : "?";
                Debug.Log(
                    $"{_logPrefix} 📥 LLM Result for {result.camera_id}:\n  Response: {result.response}\n  Model: {modelInfo}\n  Duration: {durationInfo}"
                );
            }

            try
            {
                OnLLMResultReceived?.Invoke(result);
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error in OnLLMResultReceived event handler: {ex.Message}"
                );
            }
        }

        /// <summary>
        /// Route depth result from connection to external subscribers
        /// </summary>
        internal void RouteDepthResult(DepthResult result)
        {
            if (_logResults)
            {
                string durationInfo =
                    result.metadata != null
                        ? $"{result.metadata.processing_time_seconds:F2}s"
                        : "?";
                int detectionCount = result.detections != null ? result.detections.Length : 0;

                Debug.Log(
                    $"{_logPrefix} 📏 Depth Result for {result.camera_id}: {detectionCount} objects detected in {durationInfo}"
                );

                if (result.detections != null)
                {
                    foreach (var detection in result.detections)
                    {
                        if (detection.world_position != null)
                        {
                            Debug.Log(
                                $"{_logPrefix}  - {detection.color.ToUpper()} cube at ({detection.world_position.x:F3}, {detection.world_position.y:F3}, {detection.world_position.z:F3})m, depth={detection.depth_m:F3}m, confidence={detection.confidence:F2}"
                            );
                        }
                        else
                        {
                            Debug.Log(
                                $"{_logPrefix}  - {detection.color.ToUpper()} cube at pixel ({detection.pixel_center.x}, {detection.pixel_center.y}), confidence={detection.confidence:F2}"
                            );
                        }
                    }
                }
            }

            try
            {
                OnDepthResultReceived?.Invoke(result);
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error in OnDepthResultReceived event handler: {ex.Message}"
                );
            }
        }

        #endregion

        #region Inner Connection Classes

        /// <summary>
        /// Handles results from port 5010 (LLM results only from RunAnalyzer)
        /// </summary>
        private class ResultsConnection : TCPClientBase
        {
            private const string _logPrefix = "[RESULTS_CONNECTION]";

            private UnifiedPythonReceiver _parent;
            private Thread _receiveThread;
            private Queue<string> _resultQueue = new Queue<string>();
            private readonly object _queueLock = new object();

            public void Initialize(UnifiedPythonReceiver parent)
            {
                _parent = parent;
                _serverPort = CommunicationConstants.RESULTS_SERVER_PORT; // Port 5010 - LLM results from RunAnalyzer
                _autoConnect = true;
            }

            protected override void Update()
            {
                base.Update(); // Handle auto-reconnect

                // Process all queued results on main thread
                lock (_queueLock)
                {
                    while (_resultQueue.Count > 0)
                    {
                        string json = _resultQueue.Dequeue();
                        ProcessResult(json);
                    }
                }
            }

            protected override void OnConnected()
            {
                // Start receive thread
                _receiveThread = new Thread(ReceiveLoop);
                _receiveThread.IsBackground = true;
                _receiveThread.Start();
            }

            protected override void OnConnectionFailed(Exception exception)
            {
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

            /// <summary>
            /// Background thread loop for receiving results
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
                            Debug.LogWarning($"{_logPrefix} Connection closed by server");
                            break;
                        }

                        int dataLength = BitConverter.ToInt32(lengthBuffer, 0);

                        // Validate length
                        if (dataLength <= 0 || dataLength > UnityProtocol.MAX_IMAGE_SIZE)
                        {
                            Debug.LogError(
                                $"{_logPrefix} Invalid data length received: {dataLength}"
                            );
                            break;
                        }

                        // Read message data
                        byte[] dataBuffer = new byte[dataLength];
                        bytesRead = ReadExactly(_stream, dataBuffer, dataLength);
                        if (bytesRead < dataLength)
                        {
                            Debug.LogWarning($"{_logPrefix} Incomplete data received");
                            break;
                        }

                        // Decode using protocol
                        try
                        {
                            byte[] fullMessage = new byte[UnityProtocol.INT_SIZE + dataLength];
                            Buffer.BlockCopy(
                                lengthBuffer,
                                0,
                                fullMessage,
                                0,
                                UnityProtocol.INT_SIZE
                            );
                            Buffer.BlockCopy(
                                dataBuffer,
                                0,
                                fullMessage,
                                UnityProtocol.INT_SIZE,
                                dataLength
                            );

                            string json = UnityProtocol.DecodeResultMessage(fullMessage);

                            // Queue for processing on main thread
                            lock (_queueLock)
                            {
                                _resultQueue.Enqueue(json);
                            }
                        }
                        catch (Exception parseEx)
                        {
                            Debug.LogError(
                                $"{_logPrefix} Failed to parse result: {parseEx.Message}"
                            );
                        }
                    }
                }
                catch (Exception ex)
                {
                    if (_shouldRun && _isConnected)
                    {
                        Debug.LogError($"{_logPrefix} Error in receive loop: {ex.Message}");
                    }
                }
                finally
                {
                    _isConnected = false;

                    // Trigger reconnect if enabled
                    if (_shouldRun && _autoReconnect)
                    {
                        Debug.Log($"{_logPrefix} Connection lost - will attempt to reconnect");
                    }
                }
            }

            /// <summary>
            /// Process received JSON and parse as LLM result
            /// </summary>
            private void ProcessResult(string json)
            {
                if (string.IsNullOrEmpty(json))
                    return;

                try
                {
                    // Parse as LLM result (port 5010 only receives LLM results)
                    LLMResult result = JsonUtility.FromJson<LLMResult>(json);

                    if (result != null)
                    {
                        _parent.RouteLLMResult(result);
                    }
                    else
                    {
                        Debug.LogError($"{_logPrefix} Failed to parse LLM result from JSON");
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogError($"{_logPrefix} Error processing LLM result: {ex.Message}");
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
        }

        /// <summary>
        /// Handles depth detection results from port 5007 (depth results from RunStereoDetector)
        /// </summary>
        private class DetectionConnection : TCPClientBase
        {
            private const string _logPrefix = "[DETECTION_CONNECTION]";

            private UnifiedPythonReceiver _parent;
            private Thread _receiveThread;
            private Queue<byte[]> _messageQueue = new Queue<byte[]>();
            private readonly object _queueLock = new object();

            private const int MAX_JSON_LENGTH = 10 * 1024 * 1024; // 10MB max

            public void Initialize(UnifiedPythonReceiver parent)
            {
                _parent = parent;
                _serverPort = CommunicationConstants.DETECTION_SERVER_PORT; // Port 5007 - Depth results from RunStereoDetector
                _autoConnect = true;
            }

            protected override void Update()
            {
                base.Update(); // Handle auto-reconnect

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

            protected override void OnConnected()
            {
                // Start background thread to receive data
                _receiveThread = new Thread(ReceiveLoop)
                {
                    IsBackground = true,
                    Name = "DetectionReceiver",
                };
                _receiveThread.Start();
            }

            protected override void OnConnectionFailed(Exception error)
            {
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
                // Clear message queue
                lock (_queueLock)
                {
                    _messageQueue.Clear();
                }
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
                        // Read message length (4 bytes)
                        byte[] lengthBuffer = new byte[4];
                        int bytesRead = ReadExactly(_stream, lengthBuffer, 4);

                        if (bytesRead < 4)
                        {
                            Debug.LogWarning($"{_logPrefix} Connection closed by server");
                            break;
                        }

                        int messageLength = BitConverter.ToInt32(lengthBuffer, 0);

                        if (messageLength <= 0 || messageLength > MAX_JSON_LENGTH)
                        {
                            Debug.LogError($"{_logPrefix} Invalid message length: {messageLength}");
                            continue;
                        }

                        // Read message data
                        byte[] messageData = new byte[messageLength];
                        bytesRead = ReadExactly(_stream, messageData, messageLength);

                        if (bytesRead < messageLength)
                        {
                            Debug.LogWarning($"{_logPrefix} Incomplete message received");
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
                    Debug.LogWarning($"{_logPrefix} Connection lost: {ex.Message}");
                    _isConnected = false;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"{_logPrefix} Error in receive loop: {ex.Message}");
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
                    // Format: [json_length (4 bytes)][json_data]
                    if (data.Length < 4)
                    {
                        Debug.LogError($"{_logPrefix} Invalid data: too short");
                        return;
                    }

                    int jsonLength = BitConverter.ToInt32(data, 0);

                    if (jsonLength <= 0 || jsonLength > MAX_JSON_LENGTH)
                    {
                        Debug.LogError($"{_logPrefix} Invalid JSON length: {jsonLength}");
                        return;
                    }

                    if (data.Length < 4 + jsonLength)
                    {
                        Debug.LogError(
                            $"{_logPrefix} Incomplete data: expected {4 + jsonLength} bytes, got {data.Length}"
                        );
                        return;
                    }

                    // Extract JSON string
                    string jsonString = Encoding.UTF8.GetString(data, 4, jsonLength);

                    // Parse JSON as DepthResult (stereo detection with 3D coordinates)
                    DepthResult result = JsonUtility.FromJson<DepthResult>(jsonString);

                    if (result == null)
                    {
                        Debug.LogError($"{_logPrefix} Failed to parse depth result");
                        return;
                    }

                    if (_parent._logResults)
                    {
                        Debug.Log(
                            $"{_logPrefix} Received DepthResult: camera={result.camera_id}, detections={result.detections?.Length ?? 0}"
                        );
                    }

                    // Route to parent's depth result handler
                    _parent.RouteDepthResult(result);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"{_logPrefix} Error handling data: {ex.Message}");
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
        }

        #endregion
    }
}
