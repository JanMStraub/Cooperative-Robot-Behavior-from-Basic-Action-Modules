using System;
using System.Net.Sockets;
using UnityEngine;

namespace PythonCommunication.Core
{
    /// <summary>
    /// Abstract base class for TCP client connections.
    /// Provides common functionality for connecting to servers, handling reconnection, and cleanup.
    /// Reduces duplicate code across ImageSender, LLMResultsReceiver, and future TCP clients.
    /// </summary>
    public abstract class TCPClientBase : MonoBehaviour
    {
        [Header("Connection Settings")]      

        [Tooltip("Auto-connect on Start")]
        [SerializeField]
        protected bool _autoConnect = true;

        [Tooltip("Auto-reconnect on connection loss")]
        [SerializeField]
        protected bool _autoReconnect = true;

        // Connection state
        protected TcpClient _client;
        protected NetworkStream _stream;
        protected bool _isConnected = false;
        protected bool _shouldRun = true;
        protected float _reconnectTimer = 0f;
        protected float _reconnectInterval = 2f;
        protected int _serverPort;
        protected string _serverHost = "127.0.0.1";

        // Helper variable
        private const string _logPrefix = "[TCP_CLIENT_BASE]";

        // Properties

        /// <summary>
        /// Check if the client is currently connected to the server.
        /// Fast check suitable for conditional logic.
        /// </summary>
        public bool IsConnected
        {
            get
            {
                if (!_isConnected || _client == null)
                    return false;

                try
                {
                    // Basic check - fast and reliable for most cases
                    return _client.Connected;
                }
                catch
                {
                    _isConnected = false;
                    return false;
                }
            }
        }
        /// <summary>
        /// Get the server host address
        /// </summary>
        public string ServerHost => _serverHost;

        /// <summary>
        /// Get the server port number
        /// </summary>
        public int ServerPort => _serverPort;

        /// <summary>
        /// Get formatted connection info (host:port)
        /// </summary>
        public string ConnectionInfo => $"{_serverHost}:{_serverPort}";

        #region Unity Lifecycle

        /// <summary>
        /// Called when component starts
        /// </summary>
        protected virtual void Start()
        {
            if (_autoConnect)
            {
                Connect();
            }
        }

        /// <summary>
        /// Called every frame - handles auto-reconnect
        /// </summary>
        protected virtual void Update()
        {
            // Handle auto-reconnect
            if (_autoReconnect && !IsConnected && _shouldRun)
            {
                _reconnectTimer += Time.deltaTime;
                if (_reconnectTimer >= _reconnectInterval)
                {
                    Connect();
                    _reconnectTimer = 0f;
                }
            }
        }

        /// <summary>
        /// Called when application quits
        /// </summary>
        protected virtual void OnApplicationQuit()
        {
            Disconnect();
        }

        /// <summary>
        /// Called when component is destroyed
        /// </summary>
        protected virtual void OnDestroy()
        {
            Disconnect();
        }

        #endregion

        #region Connection Management

        /// <summary>
        /// Establish connection to server
        /// </summary>
        public virtual void Connect()
        {
            if (IsConnected)
            {
                return;
            }

            try
            {
                _client = new TcpClient(_serverHost, _serverPort);
                _stream = _client.GetStream();
                _stream.ReadTimeout = System.Threading.Timeout.Infinite;

                _isConnected = true;

                OnConnected();
                Debug.Log(
                    $"{_logPrefix} Connected to {_serverHost}:{_serverPort} (read timeout: infinite)"
                );
            }
            catch (Exception ex)
            {
                _isConnected = false;
                OnConnectionFailed(ex);
                Debug.LogWarning($"{_logPrefix} Connection failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Disconnect from server
        /// </summary>
        public virtual void Disconnect()
        {
            _shouldRun = false;

            try
            {
                OnDisconnecting();

                if (_stream != null)
                {
                    _stream.Close();
                    _stream = null;
                }

                if (_client != null)
                {
                    _client.Close();
                    _client = null;
                }

                _isConnected = false;
                OnDisconnected();
                Debug.Log($"{_logPrefix} Disconnected");
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error during disconnect: {ex.Message}");
            }
        }

        /// <summary>
        /// Reconnect to server (disconnect then connect)
        /// </summary>
        public virtual void Reconnect()
        {
            Disconnect();
            _shouldRun = true;
            Connect();
        }

        #endregion

        #region Lifecycle Hooks (Optional Override)

        /// <summary>
        /// Called when connection is successfully established.
        /// Override to start background threads, initialize state, etc.
        /// </summary>
        protected virtual void OnConnected() { }

        /// <summary>
        /// Called when connection attempt fails.
        /// Override to handle specific error cases.
        /// </summary>
        /// <param name="exception">The exception that occurred</param>
        protected virtual void OnConnectionFailed(Exception exception) { }

        /// <summary>
        /// Called just before disconnecting.
        /// Override to stop background threads, save state, etc.
        /// </summary>
        protected virtual void OnDisconnecting() { }

        /// <summary>
        /// Called after disconnection is complete.
        /// Override to clean up resources.
        /// </summary>
        protected virtual void OnDisconnected() { }

        #endregion


        #region Utility Methods

        /// <summary>
        /// Check if the socket is truly connected (more thorough than IsConnected property).
        /// Use this before critical operations like sending data.
        /// </summary>
        protected bool VerifyConnection()
        {
            if (!_isConnected || _client == null)
                return false;

            try
            {
                // Quick poll check to detect dead connections
                if (
                    _client.Client != null
                    && _client.Client.Poll(0, System.Net.Sockets.SelectMode.SelectRead)
                )
                {
                    byte[] buff = new byte[1];
                    if (_client.Client.Receive(buff, System.Net.Sockets.SocketFlags.Peek) == 0)
                    {
                        // Connection is closed
                        _isConnected = false;
                        return false;
                    }
                }

                return _client.Connected;
            }
            catch
            {
                _isConnected = false;
                return false;
            }
        }

        /// <summary>
        /// Read exactly N bytes from the stream into buffer starting at offset.
        /// Returns the number of bytes read (may be less than requested if connection closed).
        /// </summary>
        /// <param name="stream">Network stream to read from</param>
        /// <param name="buffer">Buffer to read data into</param>
        /// <param name="count">Number of bytes to read</param>
        /// <param name="offset">Starting offset in buffer (default: 0)</param>
        /// <returns>Number of bytes actually read</returns>
        protected int ReadExactly(NetworkStream stream, byte[] buffer, int count, int offset = 0)
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

        /// <summary>
        /// Write data to the stream with error handling
        /// </summary>
        protected bool WriteToStream(byte[] data)
        {
            if (!IsConnected)
            {
                Debug.LogWarning($"{_logPrefix} Cannot write - not connected");
                return false;
            }

            try
            {
                _stream.Write(data, 0, data.Length);
                _stream.Flush();
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error writing to stream: {ex.Message}");
                _isConnected = false;
                return false;
            }
        }

        // Request ID counter for Protocol V2
        private static uint _nextRequestId = 1;
        private static readonly object _requestIdLock = new object();

        /// <summary>
        /// Generate a unique request ID for Protocol V2 message correlation.
        /// Thread-safe counter that wraps at uint.MaxValue.
        /// </summary>
        protected uint GenerateRequestId()
        {
            lock (_requestIdLock)
            {
                uint id = _nextRequestId;
                _nextRequestId++;
                if (_nextRequestId == 0) // Wrapped around
                    _nextRequestId = 1; // Skip 0 as it's used for "no request ID"
                return id;
            }
        }

        #endregion
    }
}
