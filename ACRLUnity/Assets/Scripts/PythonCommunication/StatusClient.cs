using System;
using System.Collections.Generic;
using System.Threading;
using Core;
using PythonCommunication.Core;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Data model for robot status result
    /// </summary>
    [System.Serializable]
    public class RobotStatusResult
    {
        public bool success;
        public string robot_id;
        public bool detailed;
        public RobotStatusData status;
        public ErrorInfo error;
    }

    [System.Serializable]
    public class RobotStatusData
    {
        public Vector3 position;
        public Quaternion rotation;
        public float[] joint_angles;
        public Vector3 target_position;
        public float distance_to_target;
        public bool is_moving;
        public string current_action;
    }

    [System.Serializable]
    public class ErrorInfo
    {
        public string code;
        public string message;
    }

    /// <summary>
    /// TCP client for robot status queries to Python.
    /// Sends status query requests and receives robot state information.
    /// Follows the RAGClient pattern for bidirectional communication.
    /// </summary>
    public class StatusClient : TCPClientBase
    {
        public static StatusClient Instance { get; private set; }

        [Header("Status Client Settings")]
        [Tooltip("Log all queries and responses to console")]
        [SerializeField]
        private bool _logQueries = true;

        // Event for when status results are received
        public event Action<RobotStatusResult> OnStatusResultReceived;

        // Background thread for receiving responses
        private Thread _receiveThread;
        private Queue<string> _responseQueue = new Queue<string>();
        private readonly object _queueLock = new object();

        // Helper variable
        private const string _logPrefix = "[STATUS_CLIENT]";

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
                _serverPort = CommunicationConstants.STATUS_SERVER_PORT; // Port 5012
                Debug.Log($"{_logPrefix} ✓ Initialized (port {_serverPort})");
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
            Debug.Log($"{_logPrefix} ✓ Connected to Status server at {ConnectionInfo}");

            // Start background receive thread
            _receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = "StatusClient_ReceiveThread",
            };
            _receiveThread.Start();
        }

        protected override void OnConnectionFailed(Exception exception)
        {
            Debug.LogWarning(
                $"{_logPrefix} ⚠️ Connection failed: {exception.Message}. Will retry in {_reconnectInterval}s"
            );
        }

        protected override void OnDisconnecting()
        {
            Debug.Log($"{_logPrefix} Disconnecting from Status server...");
        }

        protected override void OnDisconnected()
        {
            Debug.Log($"{_logPrefix} ✓ Disconnected from Status server");

            // Wait for receive thread to finish
            if (_receiveThread != null && _receiveThread.IsAlive)
            {
                _receiveThread.Join(1000); // Wait up to 1 second
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
        /// Query robot status from the status server.
        /// </summary>
        /// <param name="robotId">Robot identifier (e.g., "Robot1", "AR4_Robot")</param>
        /// <param name="detailed">If true, request detailed joint information</param>
        /// <returns>True if query was sent successfully</returns>
        public bool QueryStatus(string robotId, bool detailed = false)
        {
            if (!IsConnected)
            {
                Debug.LogWarning($"{_logPrefix} Cannot query - not connected to server");
                return false;
            }

            if (string.IsNullOrEmpty(robotId))
            {
                Debug.LogError($"{_logPrefix} Robot ID cannot be null or empty");
                return false;
            }

            try
            {
                // Encode query message
                byte[] message = UnityProtocol.EncodeStatusQuery(robotId, detailed);

                // Send to server
                bool success = WriteToStream(message);

                if (success && _logQueries)
                {
                    Debug.Log(
                        $"{_logPrefix} 📤 Sent status query: robot='{robotId}' detailed={detailed}"
                    );
                }

                return success;
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error sending query: {ex.Message}");
                return false;
            }
        }

        #endregion

        #region Background Receive Thread

        /// <summary>
        /// Background thread loop for receiving status responses
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
                        // Read response message length (4 bytes)
                        byte[] lengthBuffer = new byte[UnityProtocol.INT_SIZE];
                        int bytesRead = ReadExactly(_stream, lengthBuffer, UnityProtocol.INT_SIZE);

                        if (bytesRead == 0)
                        {
                            Debug.Log($"{_logPrefix} Connection closed by server");
                            break;
                        }

                        if (bytesRead < UnityProtocol.INT_SIZE)
                        {
                            Debug.LogWarning(
                                $"{_logPrefix} Incomplete length header (got {bytesRead} bytes)"
                            );
                            break;
                        }

                        // Parse message length
                        int messageLength = BitConverter.ToInt32(lengthBuffer, 0);

                        if (messageLength <= 0 || messageLength > UnityProtocol.MAX_IMAGE_SIZE)
                        {
                            Debug.LogError($"{_logPrefix} Invalid message length: {messageLength}");
                            break;
                        }

                        // Read complete message
                        byte[] messageBuffer = new byte[UnityProtocol.INT_SIZE + messageLength];
                        Buffer.BlockCopy(lengthBuffer, 0, messageBuffer, 0, UnityProtocol.INT_SIZE);

                        bytesRead = ReadExactly(
                            _stream,
                            messageBuffer,
                            messageLength,
                            UnityProtocol.INT_SIZE
                        );

                        if (bytesRead < messageLength)
                        {
                            Debug.LogWarning(
                                $"{_logPrefix} Incomplete message (expected {messageLength}, got {bytesRead})"
                            );
                            break;
                        }

                        // Decode response
                        string jsonResponse = UnityProtocol.DecodeStatusResponse(messageBuffer);

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

        /// <summary>
        /// Helper method to read exactly N bytes into buffer starting at offset
        /// </summary>
        private int ReadExactly(
            System.Net.Sockets.NetworkStream stream,
            byte[] buffer,
            int count,
            int offset = 0
        )
        {
            int totalRead = 0;
            while (totalRead < count)
            {
                int read = stream.Read(buffer, offset + totalRead, count - totalRead);
                if (read == 0)
                    return totalRead; // Connection closed
                totalRead += read;
            }
            return totalRead;
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
            try
            {
                // Parse JSON to RobotStatusResult
                RobotStatusResult result = JsonUtility.FromJson<RobotStatusResult>(jsonResponse);

                if (result == null)
                {
                    Debug.LogError($"{_logPrefix} Failed to parse status response");
                    return;
                }

                if (_logQueries)
                {
                    if (result.success && result.status != null)
                    {
                        Debug.Log(
                            $"{_logPrefix} 📥 Received status for '{result.robot_id}':\n"
                                + $"  Position: ({result.status.position.x:F3}, {result.status.position.y:F3}, {result.status.position.z:F3})\n"
                                + $"  Target: ({result.status.target_position.x:F3}, {result.status.target_position.y:F3}, {result.status.target_position.z:F3})\n"
                                + $"  Distance: {result.status.distance_to_target:F3}m\n"
                                + $"  Moving: {result.status.is_moving}\n"
                                + $"  Action: {result.status.current_action ?? "None"}"
                        );

                        if (result.detailed && result.status.joint_angles != null)
                        {
                            Debug.Log(
                                $"{_logPrefix}   Joint Angles: [{string.Join(", ", result.status.joint_angles)}]"
                            );
                        }
                    }
                    else if (!result.success && result.error != null)
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} ⚠️ Status query failed for '{result.robot_id}': {result.error.code} - {result.error.message}"
                        );
                    }
                }

                // Fire event
                try
                {
                    OnStatusResultReceived?.Invoke(result);
                }
                catch (Exception ex)
                {
                    Debug.LogError(
                        $"{_logPrefix} Error in OnStatusResultReceived event handler: {ex.Message}"
                    );
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error processing response: {ex.Message}");
            }
        }

        #endregion
    }
}
