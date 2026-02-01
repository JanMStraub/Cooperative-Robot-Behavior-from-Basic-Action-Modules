using System;
using PythonCommunication.Core;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Unified receiver for all Python result communication.
    /// Routes all results through CommandServer (port 5010).
    /// Handles LLM results and robot commands.
    ///
    /// This is now a lightweight manager that uses ResultsClient for all networking.
    /// </summary>
    public class UnifiedPythonReceiver : MonoBehaviour
    {
        public static UnifiedPythonReceiver Instance { get; private set; }

        [Header("Result Processing")]
        [Tooltip("Log all received results to console")]
        [SerializeField]
        private bool _logResults = true;

        [Tooltip("Enable verbose logging (includes completion sends)")]
        [SerializeField]
        private bool _verboseLogging = false;

        /// <summary>
        /// Event fired when an LLM result is received
        /// </summary>
        public event Action<LLMResult> OnLLMResultReceived;

        // Single robust client for all results (port 5010)
        private ResultsClient _client;

        private const string _logPrefix = "[UNIFIED_RECEIVER]";

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

                GameObject clientObj = new GameObject("ResultsClient");
                clientObj.transform.SetParent(transform);
                _client = clientObj.AddComponent<ResultsClient>();

                _client.OnJsonReceived += HandleJsonResult;
                _client.SetVerboseLogging(_verboseLogging);

                Debug.Log($"{_logPrefix} Initialized - using ResultsClient on port 5010");
            }
            else
            {
                Destroy(gameObject);
            }
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                if (_client != null)
                {
                    _client.OnJsonReceived -= HandleJsonResult;
                }

                Instance = null;
            }
        }

        #endregion

        #region JSON Routing

        /// <summary>
        /// Handle received JSON and route to appropriate handler.
        /// Runs on main thread (called by ResultsClient.OnResponseReceived).
        /// </summary>
        /// <param name="json">Raw JSON string</param>
        /// <param name="requestId">Request ID for correlation</param>
        private void HandleJsonResult(string json, uint requestId)
        {
            if (string.IsNullOrEmpty(json))
                return;

            if (json.Contains("\"command_type\""))
            {
                if (
                    JsonParser.TryParseWithLogging<RobotCommand>(
                        json,
                        out RobotCommand command,
                        _logPrefix
                    )
                )
                {
                    command.request_id = requestId;

                    if (PythonCommandHandler.Instance != null)
                    {
                        PythonCommandHandler.Instance.HandleCommand(command);
                    }
                    else
                    {
                        Debug.LogWarning(
                            $"{_logPrefix} PythonCommandHandler not available - command {command.command_type} not processed"
                        );
                    }
                }
            }
            else
            {
                if (
                    JsonParser.TryParseWithLogging<LLMResult>(
                        json,
                        out LLMResult result,
                        _logPrefix
                    )
                )
                {
                    result.request_id = requestId;
                    RouteLLMResult(result);
                }
            }
        }

        /// <summary>
        /// Route LLM result to external subscribers
        /// </summary>
        private void RouteLLMResult(LLMResult result)
        {
            if (_logResults)
            {
                string modelInfo = result.metadata?.model ?? "unknown";
                string durationInfo =
                    result.metadata != null ? $"{result.metadata.duration_seconds:F2}s" : "?";
                Debug.Log(
                    $"{_logPrefix} [req={result.request_id}] LLM Result for {result.camera_id}: response={result.response}, model={modelInfo}, duration={durationInfo}"
                );
            }

            try
            {
                OnLLMResultReceived?.Invoke(result);
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"{_logPrefix} Error in OnLLMResultReceived event handler: {ex.Message}"
                );
            }
        }

        #endregion

        #region Public API

        /// <summary>
        /// Send a completion message back to Python on the results connection.
        /// Used by PythonCommandHandler to notify Python when commands complete.
        /// </summary>
        /// <param name="completionJson">JSON string containing completion data</param>
        /// <param name="requestId">Request ID for correlation</param>
        /// <returns>True if sent successfully</returns>
        public bool SendCompletion(string completionJson, uint requestId)
        {
            if (_client == null || !_client.IsConnected)
            {
                Debug.LogWarning($"{_logPrefix} Cannot send completion - client not connected");
                return false;
            }

            return _client.SendCompletion(completionJson, requestId);
        }

        /// <summary>
        /// Enable or disable verbose logging at runtime
        /// </summary>
        /// <param name="verbose">True to enable verbose logging</param>
        public void SetVerboseLogging(bool verbose)
        {
            _verboseLogging = verbose;
            if (_client != null)
            {
                _client.SetVerboseLogging(verbose);
            }
        }

        /// <summary>
        /// Check if the results client is connected
        /// </summary>
        public bool IsConnected => _client != null && _client.IsConnected;

        #endregion
    }
}
