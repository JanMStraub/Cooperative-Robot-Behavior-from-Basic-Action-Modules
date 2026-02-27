using System;
using System.Collections.Generic;
using System.Text;
using Core;
using PythonCommunication.Core;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// TCP client for sending multi-command sequences to Python.
    /// Refactored to use BidirectionalClientBase for robustness and Protocol V2 compliance.
    /// </summary>
    public class SequenceClient : BidirectionalClientBase<SequenceResult>
    {
        public static SequenceClient Instance { get; private set; }

        [Header("Query Settings")]
        [Tooltip("Natural language command sequence")]
        [TextArea(3, 10)]
        [SerializeField]
        private string _prompt = "move to (0.3, 0.2, 0.1) and close the gripper";

        [Tooltip("Default robot ID for commands")]
        [SerializeField]
        private string _defaultRobotId = "Robot1";

        [Tooltip("Camera ID to use for perception operations")]
        [SerializeField]
        private string _cameraId = "TableStereoCamera";

        [Header("Settings")]
        [Tooltip("Log all commands and responses to console")]
        [SerializeField]
        private bool _logCommands = true;

        [Tooltip("Automatically execute the operations")]
        [SerializeField]
        private bool _autoExecuteResult = true;

        // Recent sequence results
        private SequenceResult _lastResult;
        private List<string> _recentCommands = new List<string>();

        // Pre-allocated 4-byte buffer for reading response length prefix — avoids per-receive allocation
        private readonly byte[] _lenBuffer = new byte[4];

        protected override string LogPrefix => "[SEQUENCE_CLIENT]";

        /// <summary>
        /// Current prompt text (for Editor access)
        /// </summary>
        public string Prompt
        {
            get => _prompt;
            set => _prompt = value;
        }

        /// <summary>
        /// Last sequence result
        /// </summary>
        public SequenceResult LastResult => _lastResult;

        /// <summary>
        /// List of recent commands sent
        /// </summary>
        public List<string> RecentCommands => _recentCommands;

        /// <summary>
        /// Event fired when a sequence result is received
        /// </summary>
        public event Action<SequenceResult> OnSequenceResultReceived;

        #region Singleton & Init

        protected override void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                base.Awake(); // Call base to capture main thread context
                _serverPort = CommunicationConstants.SEQUENCE_SERVER_PORT; // Port 5013
                Debug.Log($"{LogPrefix} Initialized (port {_serverPort})");
            }
            else
            {
                Destroy(gameObject);
            }
        }

        #endregion

        #region Public API

        /// <summary>
        /// Send the current prompt as a sequence command.
        /// Called from the Inspector UI.
        /// </summary>
        public void SendSequence()
        {
            if (string.IsNullOrEmpty(_prompt))
            {
                Debug.LogWarning($"{LogPrefix} Prompt is empty");
                return;
            }

            bool success = ExecuteSequence(_prompt, _defaultRobotId);

            if (success)
            {
                // Add to recent commands
                _recentCommands.Insert(0, _prompt);
                if (_recentCommands.Count > 10)
                    _recentCommands.RemoveAt(_recentCommands.Count - 1);
            }
        }

        /// <summary>
        /// Clear the current prompt.
        /// </summary>
        public void ClearPrompt()
        {
            _prompt = "";
            Debug.Log($"{LogPrefix} Prompt cleared");
        }

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
                Debug.LogWarning($"{LogPrefix} Cannot execute - not connected to server");
                return false;
            }

            if (string.IsNullOrEmpty(command))
            {
                Debug.LogError($"{LogPrefix} Command cannot be null or empty");
                return false;
            }

            string robot = robotId ?? _defaultRobotId;
            uint requestId = GenerateRequestId();

            try
            {
                // Encode message using Protocol V2
                byte[] message = EncodeSequenceMessage(command, robot, requestId);

                // Send using base class method (handles locking internally)
                bool sent = SendRequest(message, requestId);

                if (sent && _logCommands)
                {
                    Debug.Log(
                        $"{LogPrefix} [req={requestId}] Sent sequence: '{command}' (robot={robot}, camera={_cameraId})"
                    );
                }

                return sent;
            }
            catch (Exception ex)
            {
                Debug.LogError($"{LogPrefix} Encode error: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Execute a move command followed by a gripper action.
        /// </summary>
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
            string command =
                $"move to ({x}, {y}, {z}), then close the gripper, then move to ({x}, {y}, {liftZ})";
            return ExecuteSequence(command, robotId);
        }

        /// <summary>
        /// Execute a place operation: move to position, open gripper, move up.
        /// </summary>
        public bool Place(float x, float y, float z, float liftHeight = 0.1f, string robotId = null)
        {
            float liftZ = z + liftHeight;
            string command =
                $"move to ({x}, {y}, {z}), then open the gripper, then move to ({x}, {y}, {liftZ})";
            return ExecuteSequence(command, robotId);
        }

        #endregion

        #region Protocol V2 Implementation (Overrides)

        /// <summary>
        /// Reads and decodes the response from the stream.
        /// This runs on the background thread.
        /// Protocol V2 Format: [Type:1][RequestId:4][JsonLen:4][Json:N]
        /// </summary>
        protected override SequenceResult ReceiveResponse()
        {
            // Check if data is available before blocking read (prevents idle timeout disconnects)
            if (!_stream.DataAvailable)
            {
                System.Threading.Thread.Sleep(10); // Brief sleep to avoid tight loop
                return null; // No data available, try again later
            }

            byte[] headerBuffer = new byte[UnityProtocol.HEADER_SIZE];
            ReadExactly(_stream, headerBuffer, UnityProtocol.HEADER_SIZE);

            UnityProtocol.DecodeHeader(headerBuffer, 0, out MessageType type, out uint requestId);

            if (type != MessageType.RESULT)
            {
                Debug.LogError($"{LogPrefix} Unexpected message type: {type} (expected RESULT)");
                throw new System.IO.IOException($"Protocol violation: Expected RESULT, got {type}");
            }

            ReadExactly(_stream, _lenBuffer, 4);
            int jsonLen = BitConverter.ToInt32(_lenBuffer, 0);

            if (jsonLen <= 0 || jsonLen > CommunicationConstants.MAX_JSON_LENGTH)
            {
                throw new System.IO.IOException($"Invalid JSON length: {jsonLen}");
            }

            byte[] jsonBytes = new byte[jsonLen];
            ReadExactly(_stream, jsonBytes, jsonLen);
            string json = Encoding.UTF8.GetString(jsonBytes);

            if (
                JsonParser.TryParseWithLogging<SequenceResult>(
                    json,
                    out SequenceResult result,
                    LogPrefix
                )
            )
            {
                result.request_id = requestId;
                return result;
            }

            return null;
        }

        /// <summary>
        /// Extract request_id from response for correlation.
        /// Overridden to use the request_id field added to SequenceResult.
        /// </summary>
        protected override uint GetResponseRequestId(SequenceResult response)
        {
            return response?.request_id ?? 0;
        }

        /// <summary>
        /// Handles the processed response on the main thread.
        /// </summary>
        protected override void OnResponseReceived(SequenceResult result)
        {
            if (result == null)
                return;

            _lastResult = result;

            if (_logCommands)
            {
                if (result.success)
                {
                    Debug.Log(
                        $"{LogPrefix} [req={result.request_id}] Success: {result.completed_commands}/{result.total_commands} commands in {result.total_duration_ms:F0}ms"
                    );
                }
                else
                {
                    Debug.LogWarning(
                        $"{LogPrefix} [req={result.request_id}] Failed: {result.error}"
                    );
                }
            }

            try
            {
                OnSequenceResultReceived?.Invoke(result);
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{LogPrefix} Error in OnSequenceResultReceived handler: {ex.Message}"
                );
            }
        }

        #endregion

        #region Encoding Helper

        /// <summary>
        /// Encode a sequence query message using Protocol V2.
        /// Format: [Type:1][ReqID:4] + [CmdLen:4][Cmd:N] + [RobotLen:4][Robot:N] + [CamLen:4][Cam:N] + [AutoExec:1]
        /// </summary>
        private byte[] EncodeSequenceMessage(string command, string robotId, uint requestId)
        {
            byte[] cmdBytes = Encoding.UTF8.GetBytes(command);
            byte[] robBytes = Encoding.UTF8.GetBytes(robotId);
            byte[] camBytes = Encoding.UTF8.GetBytes(_cameraId);

            int size =
                UnityProtocol.HEADER_SIZE
                + 4
                + cmdBytes.Length
                + 4
                + robBytes.Length
                + 4
                + camBytes.Length
                + 1;

            byte[] packet = new byte[size];
            int offset = 0;

            // Header: [type:1][request_id:4] — direct byte writes, zero allocation
            packet[0] = (byte)MessageType.SEQUENCE_QUERY;
            packet[1] = (byte)(requestId);
            packet[2] = (byte)(requestId >> 8);
            packet[3] = (byte)(requestId >> 16);
            packet[4] = (byte)(requestId >> 24);
            offset += UnityProtocol.HEADER_SIZE;

            // Body: all integers in little-endian to match Python protocol
            WriteInt32LE(packet, ref offset, cmdBytes.Length);
            Buffer.BlockCopy(cmdBytes, 0, packet, offset, cmdBytes.Length);
            offset += cmdBytes.Length;

            WriteInt32LE(packet, ref offset, robBytes.Length);
            Buffer.BlockCopy(robBytes, 0, packet, offset, robBytes.Length);
            offset += robBytes.Length;

            WriteInt32LE(packet, ref offset, camBytes.Length);
            Buffer.BlockCopy(camBytes, 0, packet, offset, camBytes.Length);
            offset += camBytes.Length;

            packet[offset] = _autoExecuteResult ? (byte)1 : (byte)0;

            return packet;
        }

        /// <summary>
        /// Write a 4-byte little-endian integer directly into buffer at offset, then advance offset by 4.
        /// Matches Python protocol: "All integers are little-endian unsigned 32-bit".
        /// Zero-allocation alternative to BitConverter.GetBytes(int).
        /// </summary>
        private void WriteInt32LE(byte[] buffer, ref int offset, int value)
        {
            buffer[offset]     = (byte)(value);
            buffer[offset + 1] = (byte)(value >> 8);
            buffer[offset + 2] = (byte)(value >> 16);
            buffer[offset + 3] = (byte)(value >> 24);
            offset += 4;
        }

        #endregion
    }
}
