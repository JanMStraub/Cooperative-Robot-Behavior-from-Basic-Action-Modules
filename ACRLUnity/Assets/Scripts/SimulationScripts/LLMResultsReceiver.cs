using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using LLMCommunication.Core;

namespace LLMCommunication
{
    /// <summary>
    /// Data structure for LLM analysis results received from Python
    /// </summary>
    [System.Serializable]
    public class LLMResult
    {
        public bool success;
        public string response;
        public string camera_id;
        public string timestamp;
        public LLMMetadata metadata;
    }

    [System.Serializable]
    public class LLMMetadata
    {
        public string model;
        public float duration_seconds;
        public int image_count;
        public string[] camera_ids;
        public string prompt;
        public string full_prompt;
    }

    /// <summary>
    /// Receives LLM analysis results from Python ResultsServer via TCP.
    /// Refactored to use TCPClientBase and UnityProtocol.
    /// </summary>
    public class LLMResultsReceiver : TCPClientBase
    {
        public static LLMResultsReceiver Instance { get; private set; }

        [Header("Result Processing")]
        [Tooltip("Log all received results to console")]
        [SerializeField]
        private bool _logResults = true;

        // Events
        public event Action<LLMResult> OnResultReceived;

        // State
        private Thread _receiveThread;
        private Queue<LLMResult> _resultQueue = new Queue<LLMResult>();
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
            }
            else
            {
                Destroy(gameObject);
                return;
            }

            // Set default port for LLMResultsReceiver
            if (_serverPort == 0)
            {
                _serverPort = 5006; // ResultsServer default port
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
                    LLMResult result = _resultQueue.Dequeue();
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

        protected override string LogPrefix => "LLMResultsReceiver";

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
                        LLMResult result = JsonUtility.FromJson<LLMResult>(json);

                        // Queue for processing on main thread
                        lock (_queueLock)
                        {
                            _resultQueue.Enqueue(result);
                        }
                    }
                    catch (Exception parseEx)
                    {
                        LogError($"Failed to parse result: {parseEx.Message}");
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
        /// Process a received result (called on main thread)
        /// </summary>
        private void ProcessResult(LLMResult result)
        {
            if (result == null)
                return;

            if (_logResults)
            {
                string modelInfo = result.metadata?.model ?? "unknown";
                string durationInfo = result.metadata != null ? $"{result.metadata.duration_seconds:F2}s" : "?";
                Log($"📥 LLM Result for {result.camera_id}:\n  Response: {result.response}\n  Model: {modelInfo}\n  Duration: {durationInfo}");
            }

            // Fire event for subscribers
            try
            {
                OnResultReceived?.Invoke(result);
            }
            catch (Exception ex)
            {
                LogError($"Error in OnResultReceived event handler: {ex.Message}");
            }
        }

        #endregion
    }
}
