using System;
using Core;
using PythonCommunication.Core;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Sends robot status responses back to Python StatusServer.
    /// This completes the bidirectional status query flow.
    ///
    /// Flow:
    /// 1. Unity receives "get_robot_status" command from Python (via ResultsServer port 5010)
    /// 2. PythonCommandHandler gathers robot state
    /// 3. PythonCommandHandler calls SendStatusResponse()
    /// 4. This class sends response to StatusServer (port 5012)
    /// </summary>
    public class StatusResponseSender : TCPClientBase
    {
        public static StatusResponseSender Instance { get; private set; }

        [Header("Status Response Settings")]
        [Tooltip("Log all status responses sent")]
        [SerializeField]
        private bool _logResponses = true;

        private const string _logPrefix = "[STATUS_RESPONSE_SENDER]";

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
                _autoConnect = true;
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
            Debug.Log($"{_logPrefix} ✓ Connected to StatusServer at {ConnectionInfo}");
        }

        protected override void OnConnectionFailed(Exception exception)
        {
            Debug.LogWarning(
                $"{_logPrefix} ⚠️ Connection failed: {exception.Message}. Will retry in {_reconnectInterval}s"
            );
        }

        protected override void OnDisconnecting()
        {
            Debug.Log($"{_logPrefix} Disconnecting from StatusServer...");
        }

        protected override void OnDisconnected()
        {
            Debug.Log($"{_logPrefix} ✓ Disconnected from StatusServer");
        }

        #endregion

        #region Public API

        /// <summary>
        /// Send a status response to Python StatusServer (Protocol V2).
        /// </summary>
        /// <param name="statusJson">JSON string containing robot status</param>
        /// <param name="requestId">Request ID from original query for correlation (Protocol V2)</param>
        /// <returns>True if sent successfully</returns>
        public bool SendStatusResponse(string statusJson, uint requestId = 0)
        {
            if (!IsConnected)
            {
                Debug.LogWarning($"{_logPrefix} Cannot send - not connected to StatusServer");
                return false;
            }

            if (string.IsNullOrEmpty(statusJson))
            {
                Debug.LogError($"{_logPrefix} Status JSON cannot be null or empty");
                return false;
            }

            try
            {
                // Encode status response using protocol (Protocol V2)
                byte[] message = UnityProtocol.EncodeStatusResponse(statusJson, requestId);

                // Send to StatusServer
                bool success = WriteToStream(message);

                if (success && _logResponses)
                {
                    Debug.Log(
                        $"{_logPrefix} [req={requestId}] 📤 Sent status response ({message.Length} bytes) to StatusServer"
                    );
                }

                return success;
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} [req={requestId}] Error sending status response: {ex.Message}");
                return false;
            }
        }

        #endregion
    }
}
