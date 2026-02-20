using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace PythonCommunication.Core
{
    /// <summary>
    /// Base class for bidirectional TCP clients that send requests and receive responses.
    /// Provides common functionality for RAGClient, StatusClient, SequenceClient, etc.
    ///
    /// Features:
    /// - Thread-safe response queue
    /// - Background receive thread
    /// - Main thread callback dispatch
    /// - Request/response correlation via request_id
    /// </summary>
    /// <typeparam name="TResponse">Response data type</typeparam>
    public abstract class BidirectionalClientBase<TResponse> : TCPClientBase
        where TResponse : class
    {
        protected Queue<TResponse> _responseQueue = new Queue<TResponse>();
        protected readonly object _queueLock = new object();

        protected Thread _receiveThread;
        protected volatile bool _receiveShouldRun = false;

        protected Dictionary<uint, Action<TResponse>> _pendingRequests =
            new Dictionary<uint, Action<TResponse>>();
        protected readonly object _pendingLock = new object();

        private const int MAX_ITEMS_PER_FRAME = 50;

        protected abstract string LogPrefix { get; }

        #region Unity Lifecycle

        protected override void Update()
        {
            base.Update();
            ProcessResponseQueue();
        }

        #endregion

        #region Connection Lifecycle

        protected override void OnConnected()
        {
            base.OnConnected();

            _receiveShouldRun = true;
            _receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = $"{LogPrefix}_ReceiveThread",
            };
            _receiveThread.Start();

            Debug.Log($"{LogPrefix} Connected and receive thread started");
        }

        protected override void OnDisconnecting()
        {
            base.OnDisconnecting();

            _receiveShouldRun = false;

            lock (_queueLock)
            {
                _responseQueue.Clear();
            }
            lock (_pendingLock)
            {
                _pendingRequests.Clear();
            }

            Debug.Log($"{LogPrefix} Disconnecting cleanup done");
        }

        #endregion

        #region Receive Loop

        /// <summary>
        /// Background thread that receives responses from server.
        /// Overrides must implement ReceiveResponse() to handle specific protocol.
        /// Runs on background thread; uses locks and queues to safely marshal to main thread.
        /// </summary>
        protected virtual void ReceiveLoop()
        {
            while (_receiveShouldRun && IsConnected)
            {
                try
                {
                    TResponse response = ReceiveResponse();

                    if (response != null)
                    {
                        lock (_queueLock)
                        {
                            _responseQueue.Enqueue(response);
                        }
                    }
                }
                catch (SocketException sockEx)
                    when (sockEx.SocketErrorCode == SocketError.TimedOut
                       || sockEx.SocketErrorCode == SocketError.WouldBlock)
                {
                    // WouldBlock is the macOS equivalent of TimedOut on non-blocking sockets.
                    continue;
                }
                catch (System.IO.IOException ioEx)
                {
                    if (ioEx.InnerException is SocketException innerSockEx
                        && (innerSockEx.SocketErrorCode == SocketError.TimedOut
                            || innerSockEx.SocketErrorCode == SocketError.WouldBlock))
                    {
                        continue;
                    }

                    if (_receiveShouldRun)
                    {
                        // Distinguish a remote close (EOF / connection reset) from a
                        // true unexpected error. EOF means the server shut down cleanly.
                        bool isRemoteClose =
                            ioEx.Message.StartsWith("Connection closed") // graceful EOF from ReadExactly
                            || (ioEx.InnerException is SocketException sockEx2
                                && (sockEx2.SocketErrorCode == SocketError.ConnectionReset
                                    || sockEx2.SocketErrorCode == SocketError.ConnectionAborted));

                        if (isRemoteClose)
                            Debug.Log($"{LogPrefix} Server closed the connection.");
                        else
                            Debug.LogWarning($"{LogPrefix} Stream closed unexpectedly: {ioEx.Message}");

                        Disconnect();
                    }
                    break;
                }
                catch (Exception ex)
                {
                    if (_receiveShouldRun)
                    {
                        Debug.LogError($"{LogPrefix} Receive error: {ex.Message}");
                        Disconnect();
                    }
                    break;
                }
            }

            if (IsConnected)
            {
                Debug.LogWarning(
                    $"{LogPrefix} Receive thread died unexpectedly, triggering cleanup"
                );
                Disconnect();
            }
        }

        /// <summary>
        /// Receive and parse a single response from the server.
        /// Must be implemented by subclasses for their specific protocol.
        /// </summary>
        /// <returns>Parsed response or null if connection closed</returns>
        protected abstract TResponse ReceiveResponse();

        /// <summary>
        /// Extract request_id from response for correlation.
        /// Override if response has different structure.
        /// </summary>
        protected virtual uint GetResponseRequestId(TResponse response)
        {
            return 0;
        }

        #endregion

        #region Response Processing

        /// <summary>
        /// Process queued responses on main thread.
        /// Limits to MAX_ITEMS_PER_FRAME per update to prevent frame drops.
        /// </summary>
        protected virtual void ProcessResponseQueue()
        {
            int itemsProcessed = 0;

            while (itemsProcessed < MAX_ITEMS_PER_FRAME)
            {
                TResponse response;

                lock (_queueLock)
                {
                    if (_responseQueue.Count == 0)
                        break;

                    response = _responseQueue.Dequeue();
                }

                uint requestId = GetResponseRequestId(response);
                Action<TResponse> callback = null;

                if (requestId != 0)
                {
                    lock (_pendingLock)
                    {
                        if (_pendingRequests.TryGetValue(requestId, out callback))
                        {
                            _pendingRequests.Remove(requestId);
                        }
                    }
                }

                if (callback != null)
                {
                    try
                    {
                        callback(response);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"{LogPrefix} Callback error: {ex.Message}");
                    }
                }

                try
                {
                    OnResponseReceived(response);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"{LogPrefix} OnResponseReceived error: {ex.Message}");
                }

                itemsProcessed++;
            }
        }

        /// <summary>
        /// Called when a response is received (on main thread).
        /// Override to handle responses.
        /// </summary>
        protected virtual void OnResponseReceived(TResponse response) { }

        #endregion

        #region Request/Response

        /// <summary>
        /// Send a request and register callback for response.
        /// </summary>
        /// <param name="data">Request data to send</param>
        /// <param name="requestId">Request ID for correlation</param>
        /// <param name="callback">Callback when response received (optional)</param>
        /// <returns>True if sent successfully</returns>
        protected bool SendRequest(byte[] data, uint requestId, Action<TResponse> callback = null)
        {
            if (callback != null && requestId != 0)
            {
                lock (_pendingLock)
                {
                    _pendingRequests[requestId] = callback;
                }
            }

            if (!WriteToStream(data))
            {
                if (callback != null && requestId != 0)
                {
                    lock (_pendingLock)
                    {
                        _pendingRequests.Remove(requestId);
                    }
                }
                return false;
            }

            return true;
        }

        /// <summary>
        /// Get number of pending requests.
        /// </summary>
        public int PendingRequestCount
        {
            get
            {
                lock (_pendingLock)
                {
                    return _pendingRequests.Count;
                }
            }
        }

        /// <summary>
        /// Get number of queued responses.
        /// </summary>
        public int QueuedResponseCount
        {
            get
            {
                lock (_queueLock)
                {
                    return _responseQueue.Count;
                }
            }
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Read a 4-byte big-endian integer from stream.
        /// </summary>
        protected uint ReadUInt32BE()
        {
            byte[] buffer = new byte[4];
            ReadExactly(_stream, buffer, 4);

            if (BitConverter.IsLittleEndian)
                Array.Reverse(buffer);

            return BitConverter.ToUInt32(buffer, 0);
        }

        /// <summary>
        /// Read a length-prefixed UTF-8 string from stream.
        /// </summary>
        protected string ReadString()
        {
            uint length = ReadUInt32BE();
            if (length == 0)
                return string.Empty;

            const int MAX_STRING_LENGTH = 10 * 1024 * 1024;
            if (length > MAX_STRING_LENGTH)
            {
                throw new System.IO.IOException(
                    $"String length {length} exceeds maximum allowed {MAX_STRING_LENGTH}"
                );
            }

            byte[] buffer = new byte[length];
            ReadExactly(_stream, buffer, (int)length);

            return Encoding.UTF8.GetString(buffer);
        }

        /// <summary>
        /// Read a length-prefixed JSON string and parse it.
        /// </summary>
        protected string ReadJsonString()
        {
            return ReadString();
        }

        /// <summary>
        /// Write a 4-byte big-endian integer to buffer at offset.
        /// </summary>
        protected void WriteUInt32BE(byte[] buffer, int offset, uint value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);
            Array.Copy(bytes, 0, buffer, offset, 4);
        }

        #endregion
    }
}
