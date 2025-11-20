using System;
using System.Collections.Generic;
using System.Threading;
using Core;
using PythonCommunication.Core;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// TCP client for RAG (Retrieval-Augmented Generation) queries to Python.
    /// Sends natural language queries and receives robot operation information.
    /// </summary>
    public class RAGClient : TCPClientBase
    {
        public static RAGClient Instance { get; private set; }

        [Header("RAG Client Settings")]
        [Tooltip("Log all queries and responses to console")]
        [SerializeField]
        private bool _logQueries = true;

        [Tooltip("Default number of results to return")]
        [SerializeField]
        private int _defaultTopK = 5;

        // Event for when RAG results are received
        public event Action<RagResult> OnRagResultReceived;

        // Background thread for receiving responses
        private Thread _receiveThread;
        private Queue<string> _responseQueue = new Queue<string>();
        private readonly object _queueLock = new object();

        // Helper variable
        private const string _logPrefix = "[RAG_CLIENT]";

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
                _serverPort = CommunicationConstants.RAG_SERVER_PORT; // Port 5011
                Debug.Log($"{_logPrefix} Initialized (port {_serverPort})");
            }
            else
            {
                Destroy(gameObject);
            }
        }

        #endregion

        #region TCP Client Override

        protected override void OnConnected()
        {
            Debug.Log($"{_logPrefix} Connected to {ConnectionInfo}");

            // Start background receive thread
            _receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = "RAGClient_ReceiveThread"
            };
            _receiveThread.Start();
        }

        protected override void OnConnectionFailed(Exception exception)
        {
            // Only log if auto-reconnect is disabled
            if (!_autoReconnect)
            {
                Debug.LogWarning($"{_logPrefix} Connection failed: {exception.Message}");
            }
        }

        protected override void OnDisconnecting()
        {
            // Removed redundant log - connection state changes are logged elsewhere
        }

        protected override void OnDisconnected()
        {
            // Wait for receive thread to finish
            if (_receiveThread != null && _receiveThread.IsAlive)
            {
                _receiveThread.Join(CommunicationConstants.THREAD_JOIN_TIMEOUT_MS);
            }
        }

        #endregion

        #region Unity Update Loop

        /// <summary>
        /// Process queued responses on main thread
        /// </summary>
        protected override void Update()
        {
            base.Update(); // Handle auto-reconnect

            // Process queued responses
            ProcessResponseQueue();
        }

        #endregion

        #region Public Query API

        /// <summary>
        /// Query the RAG system for robot operations.
        /// </summary>
        /// <param name="query">Natural language query (e.g., "move robot to position")</param>
        /// <param name="topK">Number of results to return (1-100, default from inspector)</param>
        /// <param name="filters">Optional filters for category/complexity/min_score</param>
        /// <returns>True if query was sent successfully</returns>
        public bool Query(string query, int topK = -1, RagQueryFilters filters = null)
        {
            if (!IsConnected)
            {
                Debug.LogWarning($"{_logPrefix} Cannot query - not connected to server");
                return false;
            }

            if (string.IsNullOrEmpty(query))
            {
                Debug.LogError($"{_logPrefix} Query cannot be null or empty");
                return false;
            }

            // Use default topK if not specified
            if (topK <= 0)
            {
                topK = _defaultTopK;
            }

            try
            {
                // Generate unique request ID for correlation (Protocol V2)
                uint requestId = GenerateRequestId();

                // Convert filters to JSON string
                string filtersJson = filters != null ? filters.ToJson() : null;

                // Encode query message (Protocol V2)
                byte[] message = UnityProtocol.EncodeRagQuery(query, topK, filtersJson, requestId);

                // Send to server
                bool success = WriteToStream(message);

                if (success && _logQueries)
                {
                    string filterInfo = filters != null ? $", filters={filtersJson}" : "";
                    Debug.Log($"{_logPrefix} [req={requestId}] Sent query: '{query}' (topK={topK}{filterInfo})");
                }

                return success;
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error sending query: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Query for navigation operations
        /// </summary>
        public bool QueryNavigation(string taskDescription)
        {
            var (query, filters) = RagQueryHelper.NavigationQuery(taskDescription);
            return Query(query, filters: filters);
        }

        /// <summary>
        /// Query for manipulation operations
        /// </summary>
        public bool QueryManipulation(string taskDescription)
        {
            var (query, filters) = RagQueryHelper.ManipulationQuery(taskDescription);
            return Query(query, filters: filters);
        }

        /// <summary>
        /// Query for perception operations
        /// </summary>
        public bool QueryPerception(string taskDescription)
        {
            var (query, filters) = RagQueryHelper.PerceptionQuery(taskDescription);
            return Query(query, filters: filters);
        }

        /// <summary>
        /// Query for basic operations only
        /// </summary>
        public bool QueryBasicOperations(string taskDescription)
        {
            var (query, filters) = RagQueryHelper.BasicOperationsQuery(taskDescription);
            return Query(query, filters: filters);
        }

        #endregion

        #region Background Receive Thread

        /// <summary>
        /// Background thread loop for receiving RAG responses (Protocol V2)
        /// </summary>
        private void ReceiveLoop()
        {
            Debug.Log($"{_logPrefix} Receive thread started");

            try
            {
                while (_shouldRun && IsConnected)
                {
                    try
                    {
                        // Read Protocol V2 header: [type:1][request_id:4]
                        byte[] headerBuffer = new byte[UnityProtocol.HEADER_SIZE];
                        int bytesRead = ReadExactly(_stream, headerBuffer, UnityProtocol.HEADER_SIZE);

                        if (bytesRead == 0)
                        {
                            Debug.Log($"{_logPrefix} Connection closed by server");
                            break;
                        }

                        if (bytesRead < UnityProtocol.HEADER_SIZE)
                        {
                            Debug.LogWarning(
                                $"{_logPrefix} Incomplete header (got {bytesRead} bytes)"
                            );
                            break;
                        }

                        // Decode header
                        MessageType messageType = (MessageType)headerBuffer[0];
                        uint requestId = BitConverter.ToUInt32(headerBuffer, 1);

                        // Validate message type (should be RAG_RESPONSE)
                        if (messageType != MessageType.RAG_RESPONSE)
                        {
                            Debug.LogError(
                                $"{_logPrefix} Unexpected message type: {messageType} (expected RAG_RESPONSE)"
                            );
                            break;
                        }

                        // Read JSON length (4 bytes)
                        byte[] lengthBuffer = new byte[UnityProtocol.INT_SIZE];
                        bytesRead = ReadExactly(_stream, lengthBuffer, UnityProtocol.INT_SIZE);

                        if (bytesRead < UnityProtocol.INT_SIZE)
                        {
                            Debug.LogWarning(
                                $"{_logPrefix} [req={requestId}] Incomplete length header"
                            );
                            break;
                        }

                        // Parse message length
                        int messageLength = BitConverter.ToInt32(lengthBuffer, 0);

                        if (
                            messageLength <= 0
                            || messageLength > UnityProtocol.MAX_IMAGE_SIZE
                        )
                        {
                            Debug.LogError(
                                $"{_logPrefix} [req={requestId}] Invalid message length: {messageLength}"
                            );
                            break;
                        }

                        // Read JSON data
                        byte[] jsonBuffer = new byte[messageLength];
                        bytesRead = ReadExactly(_stream, jsonBuffer, messageLength);

                        if (bytesRead < messageLength)
                        {
                            Debug.LogWarning(
                                $"{_logPrefix} [req={requestId}] Incomplete message (expected {messageLength}, got {bytesRead})"
                            );
                            break;
                        }

                        // Decode JSON string
                        string jsonResponse = System.Text.Encoding.UTF8.GetString(jsonBuffer);

                        Debug.Log($"{_logPrefix} [req={requestId}] Received RAG response ({messageLength} bytes)");

                        // Queue for main thread processing
                        lock (_queueLock)
                        {
                            _responseQueue.Enqueue(jsonResponse);
                        }
                    }
                    catch (System.Threading.ThreadAbortException)
                    {
                        Debug.Log($"{_logPrefix} Receive thread aborted");
                        break;
                    }
                    catch (Exception ex)
                    {
                        if (_shouldRun && IsConnected)
                        {
                            Debug.LogError($"{_logPrefix} Error in receive loop: {ex.Message}");
                        }
                        break;
                    }
                }
            }
            finally
            {
                Debug.Log($"{_logPrefix} Receive thread stopped");
            }
        }

        #endregion

        #region Response Processing

        /// <summary>
        /// Process queued responses on main thread
        /// </summary>
        private void ProcessResponseQueue()
        {
            // Process all queued responses
            while (true)
            {
                string jsonResponse = null;

                lock (_queueLock)
                {
                    if (_responseQueue.Count == 0)
                        break;
                    jsonResponse = _responseQueue.Dequeue();
                }

                ProcessResponse(jsonResponse);
            }
        }

        /// <summary>
        /// Process a single response
        /// </summary>
        private void ProcessResponse(string jsonResponse)
        {
            // Parse JSON using centralized parser
            if (!JsonParser.TryParseWithLogging<RagResult>(jsonResponse, out RagResult result, _logPrefix))
            {
                return;
            }

            if (_logQueries)
            {
                Debug.Log(
                    $"{_logPrefix} Received {result.num_results} operation(s) for query: '{result.query}'"
                );
            }

            // Fire event
            try
            {
                OnRagResultReceived?.Invoke(result);
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error in OnRagResultReceived event handler: {ex.Message}"
                );
            }
        }

        #endregion
    }
}
