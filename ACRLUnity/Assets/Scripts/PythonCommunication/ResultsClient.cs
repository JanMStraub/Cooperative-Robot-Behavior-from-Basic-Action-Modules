using System;
using System.Text;
using Core;
using PythonCommunication.Core;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Wrapper for polymorphic JSON results (can be LLMResult or RobotCommand).
    /// Contains the raw JSON and request_id for routing.
    /// </summary>
    [Serializable]
    public class GenericResult
    {
        /// <summary>
        /// Raw JSON string (unparsed, to allow polymorphic routing)
        /// </summary>
        public string rawJson;

        public uint request_id;
    }

    /// <summary>
    /// TCP client for receiving results from Python CommandServer (port 5010).
    /// Receives both LLM results and robot commands as generic JSON, which are then
    /// routed by UnifiedPythonReceiver based on content.
    ///
    /// Also supports sending completion messages back to Python on the same connection.
    /// </summary>
    public class ResultsClient : BidirectionalClientBase<GenericResult>
    {
        /// <summary>
        /// Event fired when JSON is received (raw, unparsed)
        /// </summary>
        public event Action<string, uint> OnJsonReceived;

        [Header("Settings")]
        [Tooltip("Log received JSON to console")]
        [SerializeField]
        private bool _verboseLogging = false;

        protected override string LogPrefix => "[RESULTS_CLIENT]";

        protected override void Awake()
        {
            base.Awake();
            _serverPort = CommunicationConstants.COMMAND_SERVER_PORT;
            _autoConnect = true; // Auto-connect on start
        }

        /// <summary>
        /// Receive and decode a result message from the stream.
        /// Protocol V2: [Type:1][RequestId:4][JsonLen:4][Json:N]
        /// Runs on background thread.
        /// </summary>
        protected override GenericResult ReceiveResponse()
        {
            string json = ReadJsonMessage(MessageType.RESULT, UnityProtocol.MAX_IMAGE_SIZE, out uint requestId);
            if (json == null) return null;
            return new GenericResult { rawJson = json, request_id = requestId };
        }

        protected override uint GetResponseRequestId(GenericResult response)
        {
            return response?.request_id ?? 0;
        }

        /// <summary>
        /// Handle received response on main thread
        /// </summary>
        protected override void OnResponseReceived(GenericResult response)
        {
            if (response == null)
                return;

            if (_verboseLogging)
            {
                Debug.Log(
                    $"{LogPrefix} [req={response.request_id}] Received JSON: {response.rawJson}"
                );
            }

            try
            {
                OnJsonReceived?.Invoke(response.rawJson, response.request_id);
            }
            catch (Exception ex)
            {
                Debug.LogError($"{LogPrefix} Error in OnJsonReceived handler: {ex.Message}");
            }
        }

        /// <summary>
        /// Send a completion message back to Python on the same connection.
        /// Used by PythonCommandHandler to notify Python when commands complete.
        /// </summary>
        /// <param name="completionJson">JSON string containing completion data</param>
        /// <param name="requestId">Request ID for correlation</param>
        /// <returns>True if sent successfully</returns>
        public bool SendCompletion(string completionJson, uint requestId)
        {
            if (!IsConnected)
            {
                Debug.LogWarning($"{LogPrefix} Cannot send completion - not connected");
                return false;
            }

            try
            {
                byte[] message = UnityProtocol.EncodeStatusResponse(completionJson, requestId);
                bool success = WriteToStream(message);

                if (success && _verboseLogging)
                {
                    Debug.Log(
                        $"{LogPrefix} [req={requestId}] Sent completion ({message.Length} bytes)"
                    );
                }

                return success;
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{LogPrefix} [req={requestId}] Error sending completion: {ex.Message}"
                );
                return false;
            }
        }

        public void SetVerboseLogging(bool verbose)
        {
            _verboseLogging = verbose;
        }
    }
}
