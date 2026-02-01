using System;
using Core;
using PythonCommunication.Core;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Dedicated TCP client for broadcasting world state updates to Python.
    /// Separates world state streaming from command/response traffic to prevent
    /// message correlation conflicts on port 5010.
    ///
    /// Architecture:
    /// - Port 5010 (ResultsClient): Strict request/response pattern for commands
    /// - Port 5014 (WorldStateClient): One-way broadcast stream for world state
    ///
    /// This separation ensures unsolicited world state updates don't interfere
    /// with command request/response correlation in Protocol V2.
    /// </summary>
    public class WorldStateClient : TCPClientBase
    {
        /// <summary>
        /// Singleton instance for global access
        /// </summary>
        public static WorldStateClient Instance { get; private set; }

        [Header("Settings")]
        [Tooltip("Log sent messages to console")]
        [SerializeField]
        private bool _verboseLogging = false;

        [Header("Statistics")]
        [SerializeField]
        [Tooltip("Total number of world state updates sent")]
        private int _updatesSent = 0;

        private const string _logPrefix = "[WORLD_STATE_CLIENT]";

        #region Unity Lifecycle

        /// <summary>
        /// Initialize singleton instance
        /// </summary>
        protected override void Awake()
        {
            // Singleton pattern
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
            }
            else
            {
                Debug.LogWarning($"{_logPrefix} Duplicate instance detected, destroying");
                Destroy(gameObject);
                return;
            }

            base.Awake();

            // Set port for world state streaming
            _serverPort = CommunicationConstants.WORLD_STATE_PORT;
            _autoConnect = true;
            _autoReconnect = true;
        }

        /// <summary>
        /// Clean up singleton on destroy
        /// </summary>
        protected override void OnDestroy()
        {
            if (Instance == this)
            {
                Instance = null;
            }
            base.OnDestroy();
        }

        #endregion

        #region World State Publishing

        /// <summary>
        /// Publish a world state update to Python.
        /// Uses Protocol V2 with requestId=0 to indicate unsolicited broadcast.
        ///
        /// Thread-safe: Can be called from any thread.
        /// </summary>
        /// <param name="worldStateJson">JSON string containing world state data</param>
        /// <returns>True if sent successfully, false otherwise</returns>
        public bool PublishWorldState(string worldStateJson)
        {
            if (string.IsNullOrEmpty(worldStateJson))
            {
                Debug.LogWarning($"{_logPrefix} Cannot publish empty world state");
                return false;
            }

            if (!IsConnected)
            {
                if (_verboseLogging)
                {
                    Debug.LogWarning($"{_logPrefix} Cannot publish - not connected");
                }
                return false;
            }

            try
            {
                // Use STATUS_RESPONSE message type with requestId=0 for unsolicited updates
                // RequestID 0 is reserved for broadcasts/events in Protocol V2
                byte[] message = UnityProtocol.EncodeStatusResponse(worldStateJson, requestId: 0);
                bool success = WriteToStream(message);

                if (success)
                {
                    _updatesSent++;
                    if (_verboseLogging)
                    {
                        Debug.Log(
                            $"{_logPrefix} Published update #{_updatesSent} ({message.Length} bytes)"
                        );
                    }
                }

                return success;
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error publishing world state: {ex.Message}");
                return false;
            }
        }

        #endregion

        #region Lifecycle Hooks

        /// <summary>
        /// Called when connection is established
        /// </summary>
        protected override void OnConnected()
        {
            base.OnConnected();
            Debug.Log($"{_logPrefix} World state streaming connected to port {_serverPort}");
        }

        /// <summary>
        /// Called when connection attempt fails
        /// </summary>
        protected override void OnConnectionFailed(Exception exception)
        {
            base.OnConnectionFailed(exception);
            if (_verboseLogging)
            {
                Debug.LogWarning(
                    $"{_logPrefix} Connection failed (Python WorldStateServer may not be running)"
                );
            }
        }

        /// <summary>
        /// Called when disconnecting
        /// </summary>
        protected override void OnDisconnecting()
        {
            base.OnDisconnecting();
            Debug.Log($"{_logPrefix} Disconnecting (sent {_updatesSent} updates this session)");
        }

        #endregion
    }
}
