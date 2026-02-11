using System;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using Core;
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

        [SerializeField]
        protected int _serverPort;

        [SerializeField]
        protected string _serverHost = CommunicationConstants.SERVER_HOST;

        // Helper variable
        private const string _logPrefix = "[TCP_CLIENT_BASE]";

        // Flags
        protected readonly object _connectionLock = new object();
        protected readonly object _writeLock = new object();
        private bool _disposed = false;
        private bool _isConnecting = false;
        private CancellationTokenSource _connectCancellation;

        // Unity main thread context for safe callback marshalling
        private SynchronizationContext _mainThreadContext;

        /// <summary>
        /// Check if the client is currently connected to the server.
        ///
        /// WARNING: This property reflects the state as of the last I/O operation.
        /// It does NOT actively ping the server. If the server crashes or network
        /// disconnects, this may still return true until the next read/write fails.
        ///
        /// The only reliable way to verify connection is to attempt I/O and handle exceptions.
        /// </summary>
        public bool IsConnected
        {
            get
            {
                lock (_connectionLock)
                {
                    if (!_isConnected || _client == null)
                        return false;
                    try
                    {
                        // TcpClient.Connected is updated after I/O operations, not in real-time
                        return _client.Connected;
                    }
                    catch
                    {
                        _isConnected = false;
                        return false;
                    }
                }
            }
        }

        /// <summary>
        /// Get formatted connection info (host:port)
        /// </summary>
        public string ConnectionInfo => $"{_serverHost}:{_serverPort}";

        #region Unity Lifecycle

        /// <summary>
        /// Called when script instance is being loaded
        /// </summary>
        protected virtual void Awake()
        {
            // Capture Unity main thread context for safe callback marshalling
            _mainThreadContext = SynchronizationContext.Current;
        }

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
            if (!_disposed)
            {
                _disposed = true;
                Disconnect();
            }
        }

        #endregion

        #region Connection Management

        /// <summary>
        /// Establish connection to server (async, non-blocking)
        /// </summary>
        public virtual void Connect()
        {
            if (IsConnected || _isConnecting)
            {
                return;
            }

            // Validate port
            if (_serverPort <= 0 || _serverPort > 65535)
            {
                Debug.LogError($"{_logPrefix} Invalid port: {_serverPort}. Connection aborted.");
                return;
            }

            _isConnecting = true;
            _connectCancellation = new CancellationTokenSource();

            // Run connection asynchronously to avoid blocking main thread
            Task.Run(async () => await ConnectAsync(_connectCancellation.Token));
        }

        /// <summary>
        /// Async connection logic (runs on background thread)
        /// </summary>
        private async Task ConnectAsync(CancellationToken cancellationToken)
        {
            TcpClient tempClient = null;
            NetworkStream tempStream = null;

            try
            {
                // Create client and connect asynchronously
                tempClient = new TcpClient();
                await tempClient.ConnectAsync(_serverHost, _serverPort);

                if (cancellationToken.IsCancellationRequested)
                {
                    tempClient?.Close();
                    return;
                }

                tempStream = tempClient.GetStream();
                tempStream.ReadTimeout = 30000; // 30 seconds

                // Atomically update connection state
                lock (_connectionLock)
                {
                    if (!cancellationToken.IsCancellationRequested)
                    {
                        _client = tempClient;
                        _stream = tempStream;
                        _isConnected = true;
                    }
                }

                // Marshal OnConnected callback to Unity main thread for safe API access
                if (_mainThreadContext != null)
                {
                    _mainThreadContext.Post(
                        _ =>
                        {
                            try
                            {
                                OnConnected();
                                Debug.Log($"{_logPrefix} Connected to {_serverHost}:{_serverPort}");
                            }
                            catch (Exception ex)
                            {
                                Debug.LogError(
                                    $"{_logPrefix} OnConnected callback error: {ex.Message}"
                                );
                            }
                        },
                        null
                    );
                }
                else
                {
                    // Fallback if context not captured (shouldn't happen in Unity)
                    OnConnected();
                    Debug.Log($"{_logPrefix} Connected to {_serverHost}:{_serverPort}");
                }
            }
            catch (Exception ex)
            {
                lock (_connectionLock)
                {
                    _isConnected = false;
                }

                // Cleanup partially-initialized connection
                if (tempStream != null)
                {
                    try
                    {
                        tempStream.Close();
                    }
                    catch { }
                }
                if (tempClient != null)
                {
                    try
                    {
                        tempClient.Close();
                    }
                    catch { }
                }

                // Marshal OnConnectionFailed to main thread
                if (_mainThreadContext != null)
                {
                    _mainThreadContext.Post(
                        _ =>
                        {
                            try
                            {
                                OnConnectionFailed(ex);
                                if (!cancellationToken.IsCancellationRequested)
                                {
                                    Debug.LogWarning(
                                        $"{_logPrefix} Connection failed: {ex.Message}"
                                    );
                                }
                            }
                            catch (Exception callbackEx)
                            {
                                Debug.LogError(
                                    $"{_logPrefix} OnConnectionFailed callback error: {callbackEx.Message}"
                                );
                            }
                        },
                        null
                    );
                }
                else
                {
                    OnConnectionFailed(ex);
                    if (!cancellationToken.IsCancellationRequested)
                    {
                        Debug.LogWarning($"{_logPrefix} Connection failed: {ex.Message}");
                    }
                }
            }
            finally
            {
                _isConnecting = false;
            }
        }

        /// <summary>
        /// Disconnect from server
        /// </summary>
        public virtual void Disconnect()
        {
            _shouldRun = false;

            // Cancel any pending connection attempts
            if (_connectCancellation != null)
            {
                _connectCancellation.Cancel();
                _connectCancellation.Dispose();
                _connectCancellation = null;
            }

            // Skip if already disconnected
            if (_client == null && _stream == null)
                return;

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

                lock (_connectionLock)
                {
                    _isConnected = false;
                }

                // MARSHAL OnDisconnected TO MAIN THREAD
                if (
                    _mainThreadContext != null
                    && SynchronizationContext.Current != _mainThreadContext
                )
                {
                    _mainThreadContext.Post(
                        _ =>
                        {
                            OnDisconnected();
                            Debug.Log($"{_logPrefix} Disconnected");
                        },
                        null
                    );
                }
                else
                {
                    // Already on the main thread, or context is missing
                    OnDisconnected();
                    Debug.Log($"{_logPrefix} Disconnected");
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"{_logPrefix} Error during disconnect: {ex.Message}");
            }
        }

        #endregion

        #region Lifecycle Hooks (Optional Override)

        /// <summary>
        /// Called when connection is successfully established.
        /// Override to start background threads, initialize state, etc.
        ///
        /// THREAD SAFETY: This is called on Unity's MAIN THREAD (via SynchronizationContext).
        /// You can safely access Unity API (GameObjects, UI, Transforms) from this method.
        /// </summary>
        protected virtual void OnConnected() { }

        /// <summary>
        /// Called when connection attempt fails.
        /// Override to handle specific error cases.
        ///
        /// THREAD SAFETY: This is called on Unity's MAIN THREAD (via SynchronizationContext).
        /// You can safely access Unity API (GameObjects, UI, Transforms) from this method.
        /// </summary>
        /// <param name="exception">The exception that occurred</param>
        protected virtual void OnConnectionFailed(Exception exception) { }

        /// <summary>
        /// Called just before disconnecting.
        /// Override to stop background threads, save state, etc.
        ///
        /// WARNING: This is called on the CALLING THREAD (typically main thread from OnDestroy/OnApplicationQuit).
        /// If you need to access Unity API, ensure you're on the main thread.
        /// </summary>
        protected virtual void OnDisconnecting() { }

        /// <summary>
        /// Called after disconnection is complete.
        /// Override to clean up resources.
        ///
        /// WARNING: This is called on the CALLING THREAD (typically main thread from OnDestroy/OnApplicationQuit).
        /// If you need to access Unity API, ensure you're on the main thread.
        /// </summary>
        protected virtual void OnDisconnected() { }

        #endregion


        #region Utility Methods

        /// <summary>
        /// Read exactly N bytes from the stream into buffer starting at offset.
        /// Throws IOException if connection is closed before reading all bytes.
        ///
        /// WARNING: This method is BLOCKING and will freeze the calling thread until
        /// all bytes are read or the connection closes. NEVER call this from Unity's
        /// main thread (Update, Start, etc.) or your game will freeze.
        ///
        /// MUST be called from a background thread (e.g., BidirectionalClientBase.ReceiveLoop).
        /// </summary>
        /// <param name="stream">Network stream to read from</param>
        /// <param name="buffer">Buffer to read data into</param>
        /// <param name="count">Number of bytes to read</param>
        /// <param name="offset">Starting offset in buffer (default: 0)</param>
        /// <exception cref="System.IO.IOException">Thrown if connection closes before reading count bytes</exception>
        protected void ReadExactly(NetworkStream stream, byte[] buffer, int count, int offset = 0)
        {
            int totalRead = 0;
            while (totalRead < count)
            {
                int read;
                try
                {
                    read = stream.Read(buffer, offset + totalRead, count - totalRead);
                }
                catch (System.IO.IOException ex)
                    when (ex.InnerException is SocketException sockEx
                        && sockEx.SocketErrorCode == SocketError.TimedOut)
                {
                    // ReadTimeout expired — rethrow as SocketException so callers can
                    // distinguish a timeout from a genuine connection close.
                    throw sockEx;
                }

                if (read == 0)
                {
                    // Connection closed gracefully
                    throw new System.IO.IOException(
                        $"Connection closed after reading {totalRead}/{count} bytes"
                    );
                }
                totalRead += read;
            }
        }

        /// <summary>
        /// Write data to the stream with error handling.
        /// Thread-safe: Uses lock to prevent concurrent writes from corrupting messages.
        ///
        /// CRITICAL: Lock covers entire write operation to prevent message interleaving.
        /// Without this, concurrent writes could result in: [MSG1_PART1][MSG2_PART1][MSG1_PART2]
        /// </summary>
        protected bool WriteToStream(byte[] data)
        {
            // Use separate write lock to allow reads during writes
            lock (_writeLock)
            {
                NetworkStream streamCopy;

                // Copy stream reference to prevent race with Disconnect()
                lock (_connectionLock)
                {
                    if (!_isConnected || _stream == null)
                    {
                        Debug.LogWarning($"{_logPrefix} Cannot write - not connected");
                        return false;
                    }
                    streamCopy = _stream;
                }

                try
                {
                    // Write and flush atomically to prevent message corruption
                    streamCopy.Write(data, 0, data.Length);
                    streamCopy.Flush();
                    return true;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"{_logPrefix} Error writing to stream: {ex.Message}");
                    lock (_connectionLock)
                    {
                        _isConnected = false;
                    }
                    return false;
                }
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
