using System;
using System.Collections.Generic;
using System.Text;
using ConfigScripts;
using Core;
using PythonCommunication.Core;
using PythonCommunication.DataModels;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// TCP client for AutoRT (Autonomous Robot Task generation) integration.
    /// Connects to Python AutoRTServer (port 5015) and handles task generation,
    /// continuous loop control, and task approval workflow.
    ///
    /// Architecture:
    /// - Dedicated port 5015 for AutoRT communication
    /// - Tasks generated in Python, approved in Unity, executed back in Python
    /// - Custom inspector UI for human-in-the-loop task approval
    /// </summary>
    public class AutoRTManager : BidirectionalClientBase<AutoRTResponse>
    {
        public static AutoRTManager Instance { get; private set; }

        [Header("Configuration")]
        [Tooltip("AutoRT configuration asset")]
        [SerializeField]
        private AutoRTConfig _config;

        [Header("Runtime State")]
        [Tooltip("List of pending tasks awaiting approval")]
        [SerializeField]
        private List<ProposedTask> _pendingTasks = new List<ProposedTask>();

        [Tooltip("Is continuous loop currently running?")]
        [SerializeField]
        private bool _loopRunning = false;

        [Tooltip("Current AutoRT status")]
        [SerializeField]
        private string _statusMessage = "Idle";

        protected override string LogPrefix => "[AUTORT_MANAGER]";

        /// <summary>
        /// List of pending tasks (read-only)
        /// </summary>
        public List<ProposedTask> PendingTasks => _pendingTasks;

        /// <summary>
        /// Is continuous loop running?
        /// </summary>
        public bool LoopRunning => _loopRunning;

        /// <summary>
        /// Current status message
        /// </summary>
        public string StatusMessage => _statusMessage;

        /// <summary>
        /// AutoRT configuration
        /// </summary>
        public AutoRTConfig Config => _config;

        /// <summary>
        /// Event fired when tasks are received from Python
        /// </summary>
        public event Action<List<ProposedTask>> OnTasksReceived;

        /// <summary>
        /// Event fired when loop status changes
        /// </summary>
        public event Action<bool> OnLoopStatusChanged;

        #region Singleton & Init

        protected override void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                base.Awake(); // Call base to capture main thread context
                _serverPort = CommunicationConstants.AUTORT_SERVER_PORT; // Port 5015 (dedicated AutoRT port)
                Debug.Log($"{LogPrefix} Initialized (port {_serverPort})");
            }
            else
            {
                Destroy(gameObject);
            }
        }

        #endregion

        #region Public API - Task Generation

        /// <summary>
        /// Manually trigger task generation (one-shot).
        /// Tasks will appear in inspector UI for approval.
        /// </summary>
        /// <param name="numTasks">Number of tasks to generate (defaults to config)</param>
        /// <returns>True if request sent successfully</returns>
        public bool GenerateTasks(int? numTasks = null)
        {
            if (!IsConnected)
            {
                Debug.LogWarning($"{LogPrefix} Cannot generate tasks - not connected");
                _statusMessage = "Not connected";
                return false;
            }

            if (_config == null)
            {
                Debug.LogError($"{LogPrefix} AutoRTConfig is not assigned!");
                return false;
            }

            int taskCount = numTasks ?? _config.maxTaskCandidates;
            uint requestId = GenerateRequestId();

            try
            {
                // Build params JSON manually (simple approach, no external dependencies)
                string robotIdsJson =
                    "["
                    + string.Join(",", Array.ConvertAll(_config.robotIds, r => $"\"{r}\""))
                    + "]";
                string paramsJson =
                    $"{{\"num_tasks\":{taskCount},\"robot_ids\":{robotIdsJson},\"strategy\":\"{_config.strategy.ToString().ToLower()}\"}}";

                // Encode AUTORT_COMMAND message
                byte[] message = UnityProtocol.EncodeAutoRTCommand(
                    "generate",
                    paramsJson,
                    requestId
                );

                // Send to server
                bool sent = WriteToStream(message);

                if (sent)
                {
                    Debug.Log(
                        $"{LogPrefix} Sent task generation request (count={taskCount}, id={requestId})"
                    );
                    _statusMessage = "Generating tasks...";
                    return true;
                }
                else
                {
                    Debug.LogError($"{LogPrefix} Failed to send generation request");
                    _statusMessage = "Send failed";
                    return false;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"{LogPrefix} Error generating tasks: {e.Message}");
                _statusMessage = $"Error: {e.Message}";
                return false;
            }
        }

        #endregion

        #region Public API - Continuous Loop

        /// <summary>
        /// Start continuous task generation loop in Python backend.
        /// Tasks will be generated every loopDelay seconds and sent to Unity for approval.
        /// </summary>
        /// <param name="loopDelay">Delay between generations (defaults to config)</param>
        /// <returns>True if request sent successfully</returns>
        public bool StartLoop(float? loopDelay = null)
        {
            if (!IsConnected)
            {
                Debug.LogWarning($"{LogPrefix} Cannot start loop - not connected");
                _statusMessage = "Not connected";
                return false;
            }

            if (_config == null)
            {
                Debug.LogError($"{LogPrefix} AutoRTConfig is not assigned!");
                return false;
            }

            if (_loopRunning)
            {
                Debug.LogWarning($"{LogPrefix} Loop already running");
                return false;
            }

            float delay = loopDelay ?? _config.loopDelaySeconds;
            uint requestId = GenerateRequestId();

            try
            {
                // Build params JSON manually
                string robotIdsJson =
                    "["
                    + string.Join(",", Array.ConvertAll(_config.robotIds, r => $"\"{r}\""))
                    + "]";
                string paramsJson =
                    $"{{\"loop_delay\":{delay},\"robot_ids\":{robotIdsJson},\"strategy\":\"{_config.strategy.ToString().ToLower()}\"}}";

                // Encode AUTORT_COMMAND message
                byte[] message = UnityProtocol.EncodeAutoRTCommand(
                    "start_loop",
                    paramsJson,
                    requestId
                );

                // Send to server
                bool sent = WriteToStream(message);

                if (sent)
                {
                    Debug.Log(
                        $"{LogPrefix} Sent start loop request (delay={delay}s, id={requestId})"
                    );
                    _statusMessage = "Starting loop...";
                    return true;
                }
                else
                {
                    Debug.LogError($"{LogPrefix} Failed to send start loop request");
                    _statusMessage = "Send failed";
                    return false;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"{LogPrefix} Error starting loop: {e.Message}");
                _statusMessage = $"Error: {e.Message}";
                return false;
            }
        }

        /// <summary>
        /// Stop continuous task generation loop.
        /// </summary>
        /// <returns>True if request sent successfully</returns>
        public bool StopLoop()
        {
            if (!IsConnected)
            {
                Debug.LogWarning($"{LogPrefix} Cannot stop loop - not connected");
                return false;
            }

            if (!_loopRunning)
            {
                Debug.LogWarning($"{LogPrefix} Loop not running");
                return false;
            }

            uint requestId = GenerateRequestId();

            try
            {
                // Encode AUTORT_COMMAND message (empty params)
                byte[] message = UnityProtocol.EncodeAutoRTCommand("stop_loop", "{}", requestId);

                // Send to server
                bool sent = WriteToStream(message);

                if (sent)
                {
                    Debug.Log($"{LogPrefix} Sent stop loop request (id={requestId})");
                    _statusMessage = "Stopping loop...";
                    return true;
                }
                else
                {
                    Debug.LogError($"{LogPrefix} Failed to send stop loop request");
                    _statusMessage = "Send failed";
                    return false;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"{LogPrefix} Error stopping loop: {e.Message}");
                _statusMessage = $"Error: {e.Message}";
                return false;
            }
        }

        #endregion

        #region Public API - Task Approval

        /// <summary>
        /// Approve and execute a task (sends to Python for execution).
        /// </summary>
        /// <param name="task">Task to execute</param>
        /// <returns>True if request sent successfully</returns>
        public bool ExecuteTask(ProposedTask task)
        {
            if (!IsConnected)
            {
                Debug.LogWarning($"{LogPrefix} Cannot execute task - not connected");
                return false;
            }

            if (task == null)
            {
                Debug.LogError($"{LogPrefix} Task is null");
                return false;
            }

            uint requestId = GenerateRequestId();

            try
            {
                // Build params JSON manually
                string paramsJson = $"{{\"task_id\":\"{task.task_id}\"}}";

                // Encode AUTORT_COMMAND message
                byte[] message = UnityProtocol.EncodeAutoRTCommand(
                    "execute_task",
                    paramsJson,
                    requestId
                );

                // Send to server
                bool sent = WriteToStream(message);

                if (sent)
                {
                    Debug.Log($"{LogPrefix} Sent execute task request: {task.task_id}");
                    _statusMessage = $"Executing: {task.description}";
                    return true;
                }
                else
                {
                    Debug.LogError($"{LogPrefix} Failed to send execute request");
                    _statusMessage = "Send failed";
                    return false;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"{LogPrefix} Error executing task: {e.Message}");
                _statusMessage = $"Error: {e.Message}";
                return false;
            }
        }

        /// <summary>
        /// Reject a task (removes from pending list without execution).
        /// </summary>
        /// <param name="task">Task to reject</param>
        public void RejectTask(ProposedTask task)
        {
            if (task == null)
                return;

            _pendingTasks.Remove(task);
            Debug.Log($"{LogPrefix} Rejected task: {task.task_id}");
        }

        /// <summary>
        /// Clear all pending tasks.
        /// </summary>
        public void ClearPendingTasks()
        {
            int count = _pendingTasks.Count;
            _pendingTasks.Clear();
            Debug.Log($"{LogPrefix} Cleared {count} pending tasks");
        }

        #endregion

        #region Protocol Implementation

        /// <summary>
        /// Receive and decode AutoRT response from Python.
        /// Runs on background thread.
        /// Protocol V2 Format: [Type:1][RequestId:4][JsonLen:4][Json:N]
        /// </summary>
        protected override AutoRTResponse ReceiveResponse()
        {
            try
            {
                // Check if data is available before trying to read (prevents non-blocking socket errors)
                if (!_stream.DataAvailable)
                {
                    System.Threading.Thread.Sleep(10); // Brief sleep to avoid tight loop
                    return null; // No data available, return null and try again later
                }

                // Read header (5 bytes: type + request_id)
                byte[] headerBuffer = new byte[UnityProtocol.HEADER_SIZE];
                ReadExactly(_stream, headerBuffer, UnityProtocol.HEADER_SIZE);

                // Decode header
                UnityProtocol.DecodeHeader(
                    headerBuffer,
                    0,
                    out MessageType type,
                    out uint requestId
                );

                if (type != MessageType.AUTORT_RESPONSE)
                {
                    Debug.LogError(
                        $"{LogPrefix} Unexpected message type: {type} (expected AUTORT_RESPONSE)"
                    );
                    throw new System.IO.IOException(
                        $"Protocol violation: Expected AUTORT_RESPONSE, got {type}"
                    );
                }

                // Read JSON length (4 bytes)
                byte[] lenBuffer = new byte[4];
                ReadExactly(_stream, lenBuffer, 4);
                int jsonLen = BitConverter.ToInt32(lenBuffer, 0);

                if (jsonLen <= 0 || jsonLen > CommunicationConstants.MAX_JSON_LENGTH)
                {
                    throw new System.IO.IOException($"Invalid JSON length: {jsonLen}");
                }

                // Read JSON data
                byte[] jsonBytes = new byte[jsonLen];
                ReadExactly(_stream, jsonBytes, jsonLen);
                string json = Encoding.UTF8.GetString(jsonBytes);

                // Parse JSON to AutoRTResponse
                if (
                    JsonParser.TryParseWithLogging<AutoRTResponse>(
                        json,
                        out AutoRTResponse response,
                        LogPrefix
                    )
                )
                {
                    response.request_id = requestId;
                    return response;
                }

                return null;
            }
            catch (Exception e)
            {
                Debug.LogError($"{LogPrefix} Error receiving response: {e.Message}");
                return null;
            }
        }

        /// <summary>
        /// Extract request_id from response for correlation.
        /// </summary>
        protected override uint GetResponseRequestId(AutoRTResponse response)
        {
            return response?.request_id ?? 0;
        }

        /// <summary>
        /// Handle received response on main thread.
        /// Called by BidirectionalClientBase.ProcessResponseQueue().
        /// </summary>
        protected override void OnResponseReceived(AutoRTResponse response)
        {
            if (response == null)
                return;

            try
            {
                // Log all responses for debugging
                Debug.Log(
                    $"{LogPrefix} Received response: success={response.success}, "
                        + $"tasks={response.tasks?.Count ?? 0}, "
                        + $"status={response.status}, "
                        + $"loop_running={response.loop_running}, "
                        + $"error={response.error}"
                );

                // Update loop status
                if (response.loop_running != _loopRunning)
                {
                    _loopRunning = response.loop_running;
                    OnLoopStatusChanged?.Invoke(_loopRunning);
                    Debug.Log(
                        $"{LogPrefix} Loop status changed: {(_loopRunning ? "RUNNING" : "STOPPED")}"
                    );
                }

                // Handle error
                if (response.HasError)
                {
                    Debug.LogWarning($"{LogPrefix} Response error: {response.error}");
                    _statusMessage = $"Error: {response.error}";
                    return;
                }

                // Handle received tasks
                if (response.tasks != null && response.tasks.Count > 0)
                {
                    Debug.Log($"{LogPrefix} Received {response.tasks.Count} tasks");

                    // Add to pending list (limit max tasks)
                    foreach (var task in response.tasks)
                    {
                        if (_pendingTasks.Count < _config.maxDisplayTasks)
                        {
                            _pendingTasks.Add(task);
                        }
                        else
                        {
                            Debug.LogWarning(
                                $"{LogPrefix} Max pending tasks reached, dropping task"
                            );
                            break;
                        }
                    }

                    _statusMessage = $"Received {response.tasks.Count} tasks";
                    OnTasksReceived?.Invoke(response.tasks);
                }
                else
                {
                    _statusMessage = "No tasks generated";
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"{LogPrefix} Error processing response: {e.Message}");
                _statusMessage = $"Processing error: {e.Message}";
            }
        }

        #endregion
    }
}
