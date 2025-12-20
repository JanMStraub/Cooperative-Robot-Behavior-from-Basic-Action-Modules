using System;
using System.Collections.Generic;
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
        // Response queue for thread-safe communication
        protected Queue<TResponse> _responseQueue = new Queue<TResponse>();
        protected readonly object _queueLock = new object();

        // Receive thread
        protected Thread _receiveThread;
        protected bool _receiveShouldRun = false;

        // Pending requests for correlation
        protected Dictionary<uint, Action<TResponse>> _pendingRequests =
            new Dictionary<uint, Action<TResponse>>();
        protected readonly object _pendingLock = new object();

        // Log prefix for this client
        protected abstract string LogPrefix { get; }

        #region Unity Lifecycle

        protected override void Update()
        {
            base.Update();

            // Process queued responses on main thread
            ProcessResponseQueue();
        }

        #endregion

        #region Connection Lifecycle

        protected override void OnConnected()
        {
            base.OnConnected();

            // Start receive thread
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

            // Stop receive thread
            _receiveShouldRun = false;

            if (_receiveThread != null && _receiveThread.IsAlive)
            {
                _receiveThread.Join(1000);
                _receiveThread = null;
            }

            // Clear pending requests
            lock (_pendingLock)
            {
                _pendingRequests.Clear();
            }

            Debug.Log($"{LogPrefix} Disconnecting, receive thread stopped");
        }

        #endregion

        #region Receive Loop

        /// <summary>
        /// Background thread that receives responses from server.
        /// Override ParseResponse() to handle specific protocol.
        /// </summary>
        protected virtual void ReceiveLoop()
        {
            while (_receiveShouldRun && IsConnected)
            {
                try
                {
                    // Read response
                    TResponse response = ReceiveResponse();

                    if (response != null)
                    {
                        // Queue for main thread processing
                        lock (_queueLock)
                        {
                            _responseQueue.Enqueue(response);
                        }
                    }
                }
                catch (System.IO.IOException)
                {
                    // Connection closed
                    if (_receiveShouldRun)
                    {
                        Debug.Log($"{LogPrefix} Connection closed");
                        _isConnected = false;
                    }
                    break;
                }
                catch (Exception ex)
                {
                    if (_receiveShouldRun)
                    {
                        Debug.LogError($"{LogPrefix} Receive error: {ex.Message}");
                    }
                    break;
                }
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
            return 0; // Default: no correlation
        }

        #endregion

        #region Response Processing

        /// <summary>
        /// Process queued responses on main thread.
        /// </summary>
        protected virtual void ProcessResponseQueue()
        {
            while (true)
            {
                TResponse response;

                lock (_queueLock)
                {
                    if (_responseQueue.Count == 0)
                        break;

                    response = _responseQueue.Dequeue();
                }

                // Check for pending request callback
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

                // Call specific callback or general handler
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

                // Always call general response handler
                OnResponseReceived(response);
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
            if (!WriteToStream(data))
                return false;

            if (callback != null && requestId != 0)
            {
                lock (_pendingLock)
                {
                    _pendingRequests[requestId] = callback;
                }
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
            int read = ReadExactly(_stream, buffer, 4);
            if (read < 4)
                throw new System.IO.IOException("Connection closed");

            // Convert from big-endian
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

            byte[] buffer = new byte[length];
            int read = ReadExactly(_stream, buffer, (int)length);
            if (read < length)
                throw new System.IO.IOException("Connection closed");

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
