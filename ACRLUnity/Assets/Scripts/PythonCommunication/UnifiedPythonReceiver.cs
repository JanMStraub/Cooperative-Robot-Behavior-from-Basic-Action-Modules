using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using Core;
using PythonCommunication.Core;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Unified receiver for all Python result communication.
    /// Routes all results through CommandServer (port 5010).
    /// Handles LLM results and robot commands.
    /// </summary>
    public class UnifiedPythonReceiver : MonoBehaviour
    {
        public static UnifiedPythonReceiver Instance { get; private set; }

        [Header("Result Processing")]
        [Tooltip("Log all received results to console")]
        [SerializeField]
        private bool _logResults = true;

        [Tooltip("Enable verbose logging (includes completion sends)")]
        [SerializeField]
        private bool _verboseLogging = false;

        // Events - All existing events preserved
        public event Action<LLMResult> OnLLMResultReceived;

        // Single connection handler for all results (port 5010)
        private ResultsConnection _resultsConnection;

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

                // Create single connection handler for all results (port 5010)
                GameObject resultsConnObj = new GameObject("ResultsConnection");
                resultsConnObj.transform.SetParent(transform);
                _resultsConnection = resultsConnObj.AddComponent<ResultsConnection>();
                _resultsConnection.Initialize(this);
                _resultsConnection.SetVerboseLogging(_verboseLogging);

                Debug.Log($"{_logPrefix} Initialized - all results route through port 5010");
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
                if (_resultsConnection != null)
                {
                    _resultsConnection.Disconnect();
                }

                Instance = null;
            }
        }

        /// <summary>
        /// Called when Unity exits play mode - ensure clean shutdown
        /// </summary>
        private void OnApplicationQuit()
        {
            if (_resultsConnection != null)
            {
                _resultsConnection.Disconnect();
            }
        }

        #endregion

        #region Public API

        /// <summary>
        /// Send a completion message back to Python on the results connection.
        /// Used by PythonCommandHandler to notify Python when commands complete.
        /// </summary>
        /// <param name="completionJson">JSON string containing completion data</param>
        /// <param name="requestId">Request ID for correlation</param>
        /// <returns>True if sent successfully</returns>
        public bool SendCompletion(string completionJson, uint requestId)
        {
            if (_resultsConnection == null || !_resultsConnection.IsConnected)
            {
                Debug.LogWarning(
                    $"{_logPrefix} Cannot send completion - results connection not available"
                );
                return false;
            }

            return _resultsConnection.SendCompletion(completionJson, requestId);
        }

        /// <summary>
        /// Enable or disable verbose logging at runtime
        /// </summary>
        /// <param name="verbose">True to enable verbose logging</param>
        public void SetVerboseLogging(bool verbose)
        {
            _verboseLogging = verbose;
            if (_resultsConnection != null)
            {
                _resultsConnection.SetVerboseLogging(verbose);
            }
        }

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
                    $"{_logPrefix} LLM Result for {result.camera_id}: response={result.response}, model={modelInfo}, duration={durationInfo}"
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


        #endregion

        #region Inner Connection Classes

        /// <summary>
        /// Handles results from port 5010 (LLM results only from RunAnalyzer)
        /// Also sends completion messages back on the same connection.
        /// </summary>
        private class ResultsConnection : TCPClientBase
        {
            private const string _logPrefix = "[RESULTS_CONNECTION]";

            private UnifiedPythonReceiver _parent;
            private Thread _receiveThread;
            private Queue<(uint requestId, string json)> _resultQueue =
                new Queue<(uint requestId, string json)>();
            private readonly object _queueLock = new object();
            private readonly object _writeLock = new object();
            private bool _verboseLogging;

            public void Initialize(UnifiedPythonReceiver parent)
            {
                _parent = parent;
                _serverPort = CommunicationConstants.LLM_RESULTS_PORT; // Port 5010 - LLM results from RunAnalyzer
                _autoConnect = true;
            }

            public void SetVerboseLogging(bool verbose)
            {
                _verboseLogging = verbose;
            }

            protected override void Update()
            {
                base.Update(); // Handle auto-reconnect

                // Process all queued results on main thread
                lock (_queueLock)
                {
                    while (_resultQueue.Count > 0)
                    {
                        var (requestId, json) = _resultQueue.Dequeue();
                        ProcessResult(requestId, json);
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
                    _receiveThread.Join(CommunicationConstants.THREAD_JOIN_TIMEOUT_MS);
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
            /// Background thread loop for receiving results (Protocol V2)
            /// </summary>
            private void ReceiveLoop()
            {
                byte[] headerBuffer = new byte[UnityProtocol.HEADER_SIZE]; // Protocol V2: 5 bytes

                try
                {
                    while (_shouldRun && _isConnected)
                    {
                        // Read Protocol V2 header: [type:1][request_id:4]
                        int bytesRead = ReadExactly(
                            _stream,
                            headerBuffer,
                            UnityProtocol.HEADER_SIZE
                        );
                        if (bytesRead < UnityProtocol.HEADER_SIZE)
                        {
                            Debug.LogWarning($"{_logPrefix} Connection closed by server");
                            break;
                        }

                        // Decode header
                        MessageType messageType = (MessageType)headerBuffer[0];
                        uint requestId = BitConverter.ToUInt32(headerBuffer, 1);

                        // Validate message type (should be RESULT for this connection)
                        if (messageType != MessageType.RESULT)
                        {
                            Debug.LogError(
                                $"{_logPrefix} Unexpected message type: {messageType} (expected RESULT)"
                            );
                            break;
                        }

                        // Read JSON length (4 bytes)
                        byte[] lengthBuffer = new byte[UnityProtocol.INT_SIZE];
                        bytesRead = ReadExactly(_stream, lengthBuffer, UnityProtocol.INT_SIZE);
                        if (bytesRead < UnityProtocol.INT_SIZE)
                        {
                            Debug.LogWarning(
                                $"{_logPrefix} Connection closed while reading JSON length"
                            );
                            break;
                        }

                        int dataLength = BitConverter.ToInt32(lengthBuffer, 0);

                        // Validate length (use MAX_IMAGE_SIZE for JSON messages)
                        if (dataLength <= 0 || dataLength > UnityProtocol.MAX_IMAGE_SIZE)
                        {
                            Debug.LogError(
                                $"{_logPrefix} Invalid JSON length received: {dataLength}"
                            );
                            break;
                        }

                        // Read JSON data
                        byte[] dataBuffer = new byte[dataLength];
                        bytesRead = ReadExactly(_stream, dataBuffer, dataLength);
                        if (bytesRead < dataLength)
                        {
                            Debug.LogWarning($"{_logPrefix} Incomplete data received");
                            break;
                        }

                        // Decode JSON
                        try
                        {
                            string json = Encoding.UTF8.GetString(dataBuffer);

                            // Queue for processing on main thread with request_id (Protocol V2)
                            lock (_queueLock)
                            {
                                _resultQueue.Enqueue((requestId, json));
                            }

#if UNITY_EDITOR
                            Debug.Log(
                                $"{_logPrefix} [req={requestId}] Received result ({dataLength} bytes)"
                            );
#endif
                        }
                        catch (Exception parseEx)
                        {
                            Debug.LogError(
                                $"{_logPrefix} [req={requestId}] Failed to parse result: {parseEx.Message}"
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
            /// Process received JSON and parse as appropriate result type (Protocol V2)
            /// Handles LLM results and robot commands through port 5010.
            /// </summary>
            private void ProcessResult(uint requestId, string json)
            {
                if (string.IsNullOrEmpty(json))
                    return;

                // Try to determine result type by checking for presence of specific fields
                // Robot commands have "command_type"
                // LLM results have "response" string
                if (json.Contains("\"command_type\""))
                {
                    // Parse as RobotCommand and route to PythonCommandHandler
                    if (
                        !JsonParser.TryParseWithLogging<RobotCommand>(
                            json,
                            out RobotCommand command,
                            _logPrefix
                        )
                    )
                    {
                        return;
                    }
                    command.request_id = requestId;

                    if (PythonCommandHandler.Instance != null)
                    {
                        PythonCommandHandler.Instance.HandleCommand(command);
                    }
                    else
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} PythonCommandHandler not available - command {command.command_type} not processed"
                        );
                    }
                }
                else
                {
                    // Parse as LLM result
                    if (
                        !JsonParser.TryParseWithLogging<LLMResult>(
                            json,
                            out LLMResult llmResult,
                            _logPrefix
                        )
                    )
                    {
                        return;
                    }
                    llmResult.request_id = requestId;
                    _parent.RouteLLMResult(llmResult);
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

            /// <summary>
            /// Send a completion message back to Python on the same connection.
            /// This is used by PythonCommandHandler to notify Python when commands complete.
            /// </summary>
            /// <param name="completionJson">JSON string containing completion data</param>
            /// <param name="requestId">Request ID for correlation</param>
            /// <returns>True if sent successfully</returns>
            public bool SendCompletion(string completionJson, uint requestId)
            {
                if (!_isConnected || _stream == null)
                {
                    Debug.LogWarning($"{_logPrefix} Cannot send completion - not connected");
                    return false;
                }

                try
                {
                    // Encode as STATUS_RESPONSE message (Protocol V2)
                    byte[] message = UnityProtocol.EncodeStatusResponse(completionJson, requestId);

                    lock (_writeLock)
                    {
                        _stream.Write(message, 0, message.Length);
                        _stream.Flush();
                    }

                    if (_verboseLogging)
                    {
                        Debug.Log(
                            $"{_logPrefix} [req={requestId}] Sent completion ({message.Length} bytes)"
                        );
                    }
                    return true;
                }
                catch (Exception ex)
                {
                    Debug.LogError(
                        $"{_logPrefix} [req={requestId}] Error sending completion: {ex.Message}"
                    );
                    return false;
                }
            }
        }

        #endregion
    }
}
