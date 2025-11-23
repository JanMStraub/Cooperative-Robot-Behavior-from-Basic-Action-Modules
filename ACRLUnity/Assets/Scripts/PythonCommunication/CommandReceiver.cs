using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Core;
using PythonCommunication.Core;
using UnityEngine;
using System.Collections.Generic;

namespace PythonCommunication
{
    /// <summary>
    /// Receives commands from Python via ResultsServer (port 5010).
    /// Commands are sent by SequenceServer for execution on Unity robots.
    /// </summary>
    public class CommandReceiver : TCPClientBase
    {
        public static CommandReceiver Instance { get; private set; }

        [Header("Settings")]
        [Tooltip("Log all received commands")]
        [SerializeField]
        private bool _logCommands = true;

        /// <summary>
        /// Event fired when a command is received from Python
        /// </summary>
        public event Action<RobotCommand> OnCommandReceived;

        // Background thread for receiving
        private Thread _receiveThread;
        private Queue<string> _commandQueue = new Queue<string>();
        private readonly object _queueLock = new object();

        private const string _logPrefix = "[COMMAND_RECEIVER]";

        #region Singleton

        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                _serverPort = CommunicationConstants.LLM_RESULTS_PORT; // Port 5010
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

            _receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = "CommandReceiver_ReceiveThread"
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

        protected override void OnDisconnecting() { }

        protected override void OnDisconnected()
        {
            if (_receiveThread != null && _receiveThread.IsAlive)
            {
                _receiveThread.Join(CommunicationConstants.THREAD_JOIN_TIMEOUT_MS);
            }
        }

        #endregion

        #region Unity Update

        protected override void Update()
        {
            base.Update();
            ProcessCommandQueue();
        }

        #endregion

        #region Background Receive

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

                        // Read message length (4 bytes)
                        byte[] lengthBuffer = new byte[4];
                        bytesRead = ReadExactly(_stream, lengthBuffer, 4);

                        if (bytesRead < 4)
                        {
                            Debug.LogWarning($"{_logPrefix} [req={requestId}] Incomplete length");
                            break;
                        }

                        // Parse length (big-endian)
                        if (BitConverter.IsLittleEndian)
                            Array.Reverse(lengthBuffer);
                        int messageLength = BitConverter.ToInt32(lengthBuffer, 0);

                        if (messageLength <= 0 || messageLength > CommunicationConstants.MAX_JSON_LENGTH)
                        {
                            Debug.LogError($"{_logPrefix} [req={requestId}] Invalid length: {messageLength}");
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

                        string json = Encoding.UTF8.GetString(jsonBuffer);

                        if (_logCommands)
                        {
                            Debug.Log($"{_logPrefix} [req={requestId}] Received ({messageLength} bytes)");
                        }

                        // Queue for main thread
                        lock (_queueLock)
                        {
                            _commandQueue.Enqueue(json);
                        }
                    }
                    catch (ThreadAbortException)
                    {
                        break;
                    }
                    catch (Exception ex)
                    {
                        if (_shouldRun && IsConnected)
                        {
                            Debug.LogError($"{_logPrefix} Receive error: {ex.Message}");
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

        #region Queue Processing

        private void ProcessCommandQueue()
        {
            while (true)
            {
                string json = null;

                lock (_queueLock)
                {
                    if (_commandQueue.Count == 0)
                        break;
                    json = _commandQueue.Dequeue();
                }

                ProcessCommand(json);
            }
        }

        private void ProcessCommand(string json)
        {
            try
            {
                RobotCommand command = JsonUtility.FromJson<RobotCommand>(json);

                if (command != null && !string.IsNullOrEmpty(command.command_type))
                {
                    if (_logCommands)
                    {
                        Debug.Log($"{_logPrefix} Command: {command.command_type} for {command.robot_id}");
                    }

                    OnCommandReceived?.Invoke(command);
                }
                else if (_logCommands)
                {
                    Debug.Log($"{_logPrefix} Received non-command: {json.Substring(0, Math.Min(50, json.Length))}...");
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Parse error: {ex.Message}\nJSON: {json}");
            }
        }

        #endregion
    }
}
