using System;
using System.Collections.Generic;
using System.Threading;
using Core;
using PythonCommunication.Core;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// TCP client for sending multi-command sequences to Python.
    /// Sends compound natural language commands and receives execution results.
    /// </summary>
    public class SequenceClient : TCPClientBase
    {
        public static SequenceClient Instance { get; private set; }

        [Header("Sequence Client Settings")]
        [Tooltip("Log all commands and responses to console")]
        [SerializeField]
        private bool _logCommands = true;

        [Tooltip("Default robot ID for commands")]
        [SerializeField]
        private string _defaultRobotId = "Robot1";

        /// <summary>
        /// Event fired when a sequence result is received
        /// </summary>
        public event Action<SequenceResult> OnSequenceResultReceived;

        // Background thread for receiving responses
        private Thread _receiveThread;
        private Queue<string> _responseQueue = new Queue<string>();
        private readonly object _queueLock = new object();

        // Helper variable
        private const string _logPrefix = "[SEQUENCE_CLIENT]";

        #region Singleton

        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                _serverPort = CommunicationConstants.SEQUENCE_SERVER_PORT; // Port 5013
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
                Name = "SequenceClient_ReceiveThread"
            };
            _receiveThread.Start();
        }

        protected override void OnConnectionFailed(Exception exception)
        {
            if (!_autoReconnect)
            {
                Debug.LogWarning($"{_logPrefix} Connection failed: {exception.Message}");
            }
        }

        protected override void OnDisconnecting()
        {
            // Connection state changes are logged elsewhere
        }

        protected override void OnDisconnected()
        {
            if (_receiveThread != null && _receiveThread.IsAlive)
            {
                _receiveThread.Join(CommunicationConstants.THREAD_JOIN_TIMEOUT_MS);
            }
        }

        #endregion

        #region Unity Update Loop

        protected override void Update()
        {
            base.Update(); // Handle auto-reconnect
            ProcessResponseQueue();
        }

        #endregion

        #region Public API

        /// <summary>
        /// Execute a compound command sequence.
        /// </summary>
        /// <param name="command">Natural language command (e.g., "move to (0.3, 0.2, 0.1) and close the gripper")</param>
        /// <param name="robotId">Robot ID to execute commands on (optional, uses default)</param>
        /// <returns>True if command was sent successfully</returns>
        public bool ExecuteSequence(string command, string robotId = null)
        {
            if (!IsConnected)
            {
                Debug.LogWarning($"{_logPrefix} Cannot execute - not connected to server");
                return false;
            }

            if (string.IsNullOrEmpty(command))
            {
                Debug.LogError($"{_logPrefix} Command cannot be null or empty");
                return false;
            }

            string robot = robotId ?? _defaultRobotId;

            try
            {
                // Generate unique request ID
                uint requestId = GenerateRequestId();

                // Encode message
                byte[] message = EncodeSequenceQuery(command, robot, requestId);

                // Send to server
                bool success = WriteToStream(message);

                if (success && _logCommands)
                {
                    Debug.Log($"{_logPrefix} [req={requestId}] Sent sequence: '{command}' (robot={robot})");
                }

                return success;
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error sending sequence: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Execute a move command followed by a gripper action.
        /// </summary>
        /// <param name="x">X coordinate</param>
        /// <param name="y">Y coordinate</param>
        /// <param name="z">Z coordinate</param>
        /// <param name="closeGripper">True to close gripper after move, false to open</param>
        /// <param name="robotId">Robot ID</param>
        public bool MoveAndGrip(float x, float y, float z, bool closeGripper, string robotId = null)
        {
            string gripperAction = closeGripper ? "close the gripper" : "open the gripper";
            string command = $"move to ({x}, {y}, {z}) and {gripperAction}";
            return ExecuteSequence(command, robotId);
        }

        /// <summary>
        /// Execute a pick operation: move to position, close gripper, move up.
        /// </summary>
        public bool Pick(float x, float y, float z, float liftHeight = 0.1f, string robotId = null)
        {
            float liftZ = z + liftHeight;
            string command = $"move to ({x}, {y}, {z}), then close the gripper, then move to ({x}, {y}, {liftZ})";
            return ExecuteSequence(command, robotId);
        }

        /// <summary>
        /// Execute a place operation: move to position, open gripper, move up.
        /// </summary>
        public bool Place(float x, float y, float z, float liftHeight = 0.1f, string robotId = null)
        {
            float liftZ = z + liftHeight;
            string command = $"move to ({x}, {y}, {z}), then open the gripper, then move to ({x}, {y}, {liftZ})";
            return ExecuteSequence(command, robotId);
        }

        #endregion

        #region Protocol Encoding

        /// <summary>
        /// Encode a sequence query message.
        /// Protocol: [request_id:4][command_len:4][command:N][robot_id_len:4][robot_id:N]
        /// </summary>
        private byte[] EncodeSequenceQuery(string command, string robotId, uint requestId)
        {
            byte[] commandBytes = System.Text.Encoding.UTF8.GetBytes(command);
            byte[] robotIdBytes = System.Text.Encoding.UTF8.GetBytes(robotId);

            int totalLength = 4 + 4 + commandBytes.Length + 4 + robotIdBytes.Length;
            byte[] message = new byte[totalLength];
            int offset = 0;

            // Request ID (4 bytes, big-endian)
            byte[] requestIdBytes = BitConverter.GetBytes(requestId);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(requestIdBytes);
            Array.Copy(requestIdBytes, 0, message, offset, 4);
            offset += 4;

            // Command length (4 bytes, big-endian)
            byte[] cmdLenBytes = BitConverter.GetBytes(commandBytes.Length);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(cmdLenBytes);
            Array.Copy(cmdLenBytes, 0, message, offset, 4);
            offset += 4;

            // Command text
            Array.Copy(commandBytes, 0, message, offset, commandBytes.Length);
            offset += commandBytes.Length;

            // Robot ID length (4 bytes, big-endian)
            byte[] robotIdLenBytes = BitConverter.GetBytes(robotIdBytes.Length);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(robotIdLenBytes);
            Array.Copy(robotIdLenBytes, 0, message, offset, 4);
            offset += 4;

            // Robot ID
            Array.Copy(robotIdBytes, 0, message, offset, robotIdBytes.Length);

            return message;
        }

        #endregion

        #region Background Receive Thread

        private void ReceiveLoop()
        {
            Debug.Log($"{_logPrefix} Receive thread started");

            try
            {
                while (_shouldRun && IsConnected)
                {
                    try
                    {
                        // Read request_id (4 bytes)
                        byte[] requestIdBuffer = new byte[4];
                        int bytesRead = ReadExactly(_stream, requestIdBuffer, 4);

                        if (bytesRead == 0)
                        {
                            Debug.Log($"{_logPrefix} Connection closed by server");
                            break;
                        }

                        if (bytesRead < 4)
                        {
                            Debug.LogWarning($"{_logPrefix} Incomplete request ID");
                            break;
                        }

                        // Parse request_id (big-endian)
                        if (BitConverter.IsLittleEndian)
                            Array.Reverse(requestIdBuffer);
                        uint requestId = BitConverter.ToUInt32(requestIdBuffer, 0);

                        // Read response length (4 bytes)
                        byte[] lengthBuffer = new byte[4];
                        bytesRead = ReadExactly(_stream, lengthBuffer, 4);

                        if (bytesRead < 4)
                        {
                            Debug.LogWarning($"{_logPrefix} [req={requestId}] Incomplete length header");
                            break;
                        }

                        // Parse length (big-endian)
                        if (BitConverter.IsLittleEndian)
                            Array.Reverse(lengthBuffer);
                        int messageLength = BitConverter.ToInt32(lengthBuffer, 0);

                        if (messageLength <= 0 || messageLength > CommunicationConstants.MAX_JSON_LENGTH)
                        {
                            Debug.LogError($"{_logPrefix} [req={requestId}] Invalid message length: {messageLength}");
                            break;
                        }

                        // Read JSON data
                        byte[] jsonBuffer = new byte[messageLength];
                        bytesRead = ReadExactly(_stream, jsonBuffer, messageLength);

                        if (bytesRead < messageLength)
                        {
                            Debug.LogWarning($"{_logPrefix} [req={requestId}] Incomplete message");
                            break;
                        }

                        // Decode JSON
                        string jsonResponse = System.Text.Encoding.UTF8.GetString(jsonBuffer);

                        if (_logCommands)
                        {
                            Debug.Log($"{_logPrefix} [req={requestId}] Received sequence result ({messageLength} bytes)");
                        }

                        // Queue for main thread processing
                        lock (_queueLock)
                        {
                            _responseQueue.Enqueue(jsonResponse);
                        }
                    }
                    catch (ThreadAbortException)
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

        private void ProcessResponseQueue()
        {
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

        private void ProcessResponse(string jsonResponse)
        {
            // Parse JSON
            if (!JsonParser.TryParseWithLogging<SequenceResult>(jsonResponse, out SequenceResult result, _logPrefix))
            {
                return;
            }

            if (_logCommands)
            {
                if (result.success)
                {
                    Debug.Log($"{_logPrefix} Sequence completed: {result.completed_commands}/{result.total_commands} commands in {result.total_duration_ms:F0}ms");
                }
                else
                {
                    Debug.LogWarning($"{_logPrefix} Sequence failed: {result.error}");
                }
            }

            // Fire event
            try
            {
                OnSequenceResultReceived?.Invoke(result);
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error in OnSequenceResultReceived handler: {ex.Message}");
            }
        }

        #endregion
    }
}
