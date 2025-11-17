using System;
using System.Collections.Generic;
using Core;
using PythonCommunication.Core;
using Robotics;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Serializable command for move_to_coordinate operation
    /// </summary>
    [Serializable]
    public class MoveToCoordinateCommand
    {
        public string command_type = "move_to_coordinate";
        public string robot_id;
        public MoveParameters parameters;
        public long timestamp;

        [Serializable]
        public class MoveParameters
        {
            public Vector3Data target_position;
            public float speed_multiplier;
        }

        [Serializable]
        public class Vector3Data
        {
            public float x;
            public float y;
            public float z;
        }
    }

    /// <summary>
    /// Interactive Unity Inspector interface for querying the RAG/LLM system.
    /// Allows sending natural language prompts and executing returned operations.
    /// </summary>
    public class LLMQueryClient : MonoBehaviour
    {
        [Header("Query Settings")]
        [Tooltip("Natural language prompt for the LLM/RAG system")]
        [TextArea(3, 10)]
        [SerializeField]
        private string _prompt = "Move robot to position x=0.3, y=0.15, z=0.1";

        [Tooltip("Robot ID to use for operations")]
        [SerializeField]
        private string _robotId = "Robot1";

        [Tooltip("Number of RAG results to return")]
        [Range(1, 10)]
        [SerializeField]
        private int _topK = 3;

        [Header("Query Filters (Optional)")]
        [Tooltip("Filter by operation category")]
        [SerializeField]
        private CategoryFilter _categoryFilter = CategoryFilter.None;

        [Tooltip("Filter by complexity level")]
        [SerializeField]
        private ComplexityFilter _complexityFilter = ComplexityFilter.None;

        [Tooltip("Minimum similarity score (0.0 - 1.0)")]
        [Range(0.0f, 1.0f)]
        [SerializeField]
        private float _minScore = 0.5f;

        [Header("Auto-Execute Settings")]
        [Tooltip("Automatically execute the top-ranked operation")]
        [SerializeField]
        private bool _autoExecuteTopResult = true;

        [Tooltip("Skip auto-execute for operations that aren't implemented yet")]
        [SerializeField]
        private bool _skipUnimplemented = true;

        [Tooltip("Show detailed operation info in console")]
        [SerializeField]
        private bool _logDetailedResults = true;

        [Header("Status (Read-Only)")]
        [SerializeField]
        private string _lastQueryStatus = "Ready";

        [SerializeField]
        private string _lastOperationExecuted = "None";

        // Recent query results
        private RagResult _lastResult;
        private List<OperationInfo> _recentOperations = new List<OperationInfo>();

        // Helper variable
        private const string _logPrefix = "[LLM_QUERY_CLIENT]";

        // Enums for Inspector dropdown
        public enum CategoryFilter
        {
            None,
            Navigation,
            Manipulation,
            Perception,
            Coordination
        }

        public enum ComplexityFilter
        {
            None,
            Basic,
            Intermediate,
            Advanced,
            Expert
        }

        #region Unity Lifecycle

        /// <summary>
        /// Initialize and subscribe to RAG events
        /// </summary>
        private void OnEnable()
        {
            // Subscribe to RAG results
            if (RAGClient.Instance != null)
            {
                RAGClient.Instance.OnRagResultReceived += HandleRagResult;
                Debug.Log($"{_logPrefix} ✓ Subscribed to RAG results");
            }
            else
            {
                Debug.LogWarning($"{_logPrefix} ⚠️ RAGClient instance not found - will retry");
            }
        }

        /// <summary>
        /// Unsubscribe from events
        /// </summary>
        private void OnDisable()
        {
            if (RAGClient.Instance != null)
            {
                RAGClient.Instance.OnRagResultReceived -= HandleRagResult;
            }
        }

        /// <summary>
        /// Ensure RAGClient is connected
        /// </summary>
        private void Start()
        {
            // Ensure RAGClient is connected
            if (RAGClient.Instance == null)
            {
                Debug.LogError(
                    $"{_logPrefix} ⚠️ RAGClient not found! Please add RAGClient GameObject to scene."
                );
                _lastQueryStatus = "ERROR: RAGClient not found";
            }
            else if (!RAGClient.Instance.IsConnected)
            {
                Debug.LogWarning(
                    $"{_logPrefix} ⚠️ RAGClient not connected. Ensure Python RAGServer is running."
                );
                _lastQueryStatus = "WAITING: RAGServer not connected";
            }
            else
            {
                Debug.Log($"{_logPrefix} ✓ Ready to query RAG system");
                _lastQueryStatus = "Ready";
            }
        }

        #endregion

        #region Public API (Called by Custom Editor Buttons)

        /// <summary>
        /// Send query to RAG system (called by custom editor button)
        /// </summary>
        public void SendQuery()
        {
            if (string.IsNullOrEmpty(_prompt))
            {
                Debug.LogWarning($"{_logPrefix} ⚠️ Prompt is empty");
                _lastQueryStatus = "ERROR: Empty prompt";
                return;
            }

            if (RAGClient.Instance == null)
            {
                Debug.LogError($"{_logPrefix} ⚠️ RAGClient not found");
                _lastQueryStatus = "ERROR: RAGClient not found";
                return;
            }

            if (!RAGClient.Instance.IsConnected)
            {
                Debug.LogWarning($"{_logPrefix} ⚠️ RAGClient not connected");
                _lastQueryStatus = "ERROR: Not connected to RAGServer";
                return;
            }

            // Build filters
            RagQueryFilters filters = BuildFilters();

            // Send query
            Debug.Log($"{_logPrefix} 📤 Sending query: '{_prompt}'");
            bool success = RAGClient.Instance.Query(_prompt, _topK, filters);

            if (success)
            {
                _lastQueryStatus = $"SENT: Waiting for results...";
                Debug.Log($"{_logPrefix} ✓ Query sent successfully");
            }
            else
            {
                _lastQueryStatus = "ERROR: Failed to send query";
                Debug.LogError($"{_logPrefix} ❌ Failed to send query");
            }
        }

        /// <summary>
        /// Execute the top-ranked operation manually (called by custom editor button)
        /// </summary>
        public void ExecuteTopOperation()
        {
            if (_lastResult == null || _lastResult.operations == null || _lastResult.operations.Length == 0)
            {
                Debug.LogWarning($"{_logPrefix} ⚠️ No operations available to execute");
                _lastQueryStatus = "ERROR: No operations to execute";
                return;
            }

            ExecuteOperation(_lastResult.operations[0]);
        }

        /// <summary>
        /// Clear the prompt field
        /// </summary>
        public void ClearPrompt()
        {
            _prompt = "";
            _lastQueryStatus = "Ready";
            Debug.Log($"{_logPrefix} Prompt cleared");
        }

        #endregion

        #region RAG Result Handling

        /// <summary>
        /// Handle RAG results from Python
        /// </summary>
        private void HandleRagResult(RagResult result)
        {
            _lastResult = result;

            if (result == null)
            {
                Debug.LogError($"{_logPrefix} ❌ Received null result");
                _lastQueryStatus = "ERROR: Null result";
                return;
            }

            Debug.Log(
                $"{_logPrefix} 📥 Received {result.num_results} operation(s) for: '{result.query}'"
            );

            // Update status
            _lastQueryStatus = $"RECEIVED: {result.num_results} operation(s)";

            // Log detailed results
            if (_logDetailedResults && result.operations != null)
            {
                for (int i = 0; i < result.operations.Length; i++)
                {
                    var op = result.operations[i];
                    Debug.Log(
                        $"{_logPrefix}   [{i + 1}] {op.name} (score={op.similarity_score:F3}, category={op.category})"
                    );
                    Debug.Log($"{_logPrefix}       {op.description}");

                    if (op.parameters != null && op.parameters.Length > 0)
                    {
                        Debug.Log($"{_logPrefix}       Parameters:");
                        foreach (var param in op.parameters)
                        {
                            string req = param.required ? "required" : "optional";
                            Debug.Log(
                                $"{_logPrefix}         - {param.name}: {param.type} ({req}) - {param.description}"
                            );
                        }
                    }
                }
            }

            // Store operations
            _recentOperations.Clear();
            if (result.operations != null)
            {
                _recentOperations.AddRange(result.operations);
            }

            // Auto-execute top result if enabled
            if (_autoExecuteTopResult && result.operations != null && result.operations.Length > 0)
            {
                var topOp = result.operations[0];

                // Check if we should skip unimplemented operations
                if (_skipUnimplemented && !IsOperationImplemented(topOp.name))
                {
                    Debug.Log(
                        $"{_logPrefix} ⏭️ Skipping auto-execute for unimplemented operation: {topOp.name}"
                    );
                    Debug.Log(
                        $"{_logPrefix} 💡 Disable 'Skip Unimplemented' or click 'Execute Top Operation' to run anyway"
                    );
                }
                else
                {
                    Debug.Log($"{_logPrefix} 🚀 Auto-executing top operation: {topOp.name}");
                    ExecuteOperation(topOp);
                }
            }
        }

        /// <summary>
        /// Execute a specific operation
        /// </summary>
        private void ExecuteOperation(OperationInfo operation)
        {
            if (operation == null)
            {
                Debug.LogError($"{_logPrefix} ❌ Cannot execute null operation");
                return;
            }

            Debug.Log($"{_logPrefix} 🚀 Executing operation: {operation.name}");

            // Handle different operation types
            switch (operation.name)
            {
                case "move_to_coordinate":
                    ExecuteMoveToCoordinate(operation);
                    break;

                case "check_robot_status":
                    ExecuteCheckRobotStatus(operation);
                    break;

                default:
                    LogOperationNotImplemented(operation);
                    break;
            }
        }

        /// <summary>
        /// Execute move_to_coordinate operation by parsing prompt
        /// </summary>
        private void ExecuteMoveToCoordinate(OperationInfo operation)
        {
            // Parse coordinates from prompt using simple pattern matching
            Vector3? targetPos = TryParseCoordinatesFromPrompt(_prompt);

            if (!targetPos.HasValue)
            {
                Debug.LogWarning(
                    $"{_logPrefix} ⚠️ Could not parse coordinates from prompt. Using default position (0.3, 0.15, 0.1)"
                );
                targetPos = new Vector3(0.3f, 0.15f, 0.1f);
            }

            Debug.Log(
                $"{_logPrefix} 📍 Moving {_robotId} to ({targetPos.Value.x}, {targetPos.Value.y}, {targetPos.Value.z})"
            );

            // Create serializable command object for logging
            var command = new MoveToCoordinateCommand
            {
                robot_id = _robotId,
                parameters = new MoveToCoordinateCommand.MoveParameters
                {
                    target_position = new MoveToCoordinateCommand.Vector3Data
                    {
                        x = targetPos.Value.x,
                        y = targetPos.Value.y,
                        z = targetPos.Value.z
                    },
                    speed_multiplier = 1.0f
                },
                timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
            };

            string commandJson = JsonUtility.ToJson(command, true);
            Debug.Log($"{_logPrefix} 📤 Command JSON: {commandJson}");

            // Execute directly using RobotManager
            if (RobotManager.Instance == null)
            {
                Debug.LogError($"{_logPrefix} ❌ RobotManager.Instance is null!");
                _lastQueryStatus = "ERROR: RobotManager not found";
                return;
            }

            if (!RobotManager.Instance.RobotInstances.TryGetValue(_robotId, out RobotInstance robotInstance))
            {
                Debug.LogError(
                    $"{_logPrefix} ❌ Robot '{_robotId}' not found in RobotManager. "
                    + $"Available robots: {string.Join(", ", RobotManager.Instance.RobotInstances.Keys)}"
                );
                _lastQueryStatus = $"ERROR: Robot '{_robotId}' not found";
                return;
            }

            RobotController controller = robotInstance.controller;
            if (controller == null)
            {
                Debug.LogError($"{_logPrefix} ❌ Robot '{_robotId}' has no RobotController component");
                _lastQueryStatus = "ERROR: No RobotController";
                return;
            }

            // Execute movement
            controller.SetTarget(targetPos.Value);

            // Update status
            _lastOperationExecuted = $"move_to_coordinate({targetPos.Value.x:F2}, {targetPos.Value.y:F2}, {targetPos.Value.z:F2})";
            _lastQueryStatus = $"EXECUTED: {operation.name}";

            Debug.Log($"{_logPrefix} ✓ Movement command executed on {_robotId}");
        }

        /// <summary>
        /// Execute check_robot_status operation
        /// </summary>
        private void ExecuteCheckRobotStatus(OperationInfo operation)
        {
            Debug.Log($"{_logPrefix} 🔍 Checking status of {_robotId}");

            // Check if StatusClient is available
            if (StatusClient.Instance == null)
            {
                Debug.LogError($"{_logPrefix} ❌ StatusClient.Instance is null!");
                _lastQueryStatus = "ERROR: StatusClient not found";
                return;
            }

            // Parse detailed flag from prompt if available (default: false)
            bool detailed = _prompt.ToLower().Contains("detailed") || _prompt.ToLower().Contains("joints");

            // Send status query to Python via StatusClient
            bool success = StatusClient.Instance.QueryStatus(_robotId, detailed);

            if (success)
            {
                _lastOperationExecuted = $"check_robot_status({_robotId}, detailed={detailed})";
                _lastQueryStatus = $"EXECUTED: {operation.name}";
                Debug.Log($"{_logPrefix} ✓ Status query sent for {_robotId}");
            }
            else
            {
                _lastQueryStatus = "ERROR: Failed to send status query";
                Debug.LogError($"{_logPrefix} ❌ Failed to send status query");
            }
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Check if an operation is implemented in this client
        /// </summary>
        private bool IsOperationImplemented(string operationName)
        {
            switch (operationName)
            {
                case "move_to_coordinate":
                case "check_robot_status":
                    return true;

                // Add new implemented operations here
                default:
                    return false;
            }
        }

        /// <summary>
        /// Log information about an operation that isn't implemented yet
        /// Shows helpful details about the operation for future implementation
        /// </summary>
        private void LogOperationNotImplemented(OperationInfo operation)
        {
            Debug.LogWarning(
                $"{_logPrefix} ⚠️ Operation '{operation.name}' not yet implemented in LLMQueryClient"
            );

            // Show helpful information about the operation
            Debug.Log($"{_logPrefix} 📋 Operation Details:");
            Debug.Log($"{_logPrefix}    Category: {operation.category}");
            Debug.Log($"{_logPrefix}    Complexity: {operation.complexity}");
            Debug.Log($"{_logPrefix}    Description: {operation.description}");

            if (operation.parameters != null && operation.parameters.Length > 0)
            {
                Debug.Log($"{_logPrefix}    Parameters:");
                foreach (var param in operation.parameters)
                {
                    string req = param.required ? "required" : "optional";
                    string defVal = !string.IsNullOrEmpty(param.default_value)
                        ? $", default={param.default_value}"
                        : "";
                    Debug.Log(
                        $"{_logPrefix}      - {param.name}: {param.type} ({req}{defVal})"
                    );
                }
            }

            if (operation.usage_examples != null && operation.usage_examples.Length > 0)
            {
                Debug.Log($"{_logPrefix}    Example usage:");
                Debug.Log($"{_logPrefix}      {operation.usage_examples[0]}");
            }

            _lastOperationExecuted = $"{operation.name} (not implemented)";
            _lastQueryStatus = $"INFO: {operation.name} needs implementation";
        }

        /// <summary>
        /// Build query filters from inspector settings
        /// </summary>
        private RagQueryFilters BuildFilters()
        {
            var filters = new RagQueryFilters { min_score = _minScore };

            // Category filter
            switch (_categoryFilter)
            {
                case CategoryFilter.Navigation:
                    filters.category = RagQueryHelper.CATEGORY_NAVIGATION;
                    break;
                case CategoryFilter.Manipulation:
                    filters.category = RagQueryHelper.CATEGORY_MANIPULATION;
                    break;
                case CategoryFilter.Perception:
                    filters.category = RagQueryHelper.CATEGORY_PERCEPTION;
                    break;
                case CategoryFilter.Coordination:
                    filters.category = RagQueryHelper.CATEGORY_COORDINATION;
                    break;
            }

            // Complexity filter
            switch (_complexityFilter)
            {
                case ComplexityFilter.Basic:
                    filters.complexity = RagQueryHelper.COMPLEXITY_BASIC;
                    break;
                case ComplexityFilter.Intermediate:
                    filters.complexity = RagQueryHelper.COMPLEXITY_INTERMEDIATE;
                    break;
                case ComplexityFilter.Advanced:
                    filters.complexity = RagQueryHelper.COMPLEXITY_ADVANCED;
                    break;
                case ComplexityFilter.Expert:
                    filters.complexity = RagQueryHelper.COMPLEXITY_EXPERT;
                    break;
            }

            return filters;
        }

        /// <summary>
        /// Try to parse x, y, z coordinates from natural language prompt
        /// Supports patterns like: "x=0.3, y=0.15, z=0.1" or "position 0.3 0.15 0.1"
        /// </summary>
        private Vector3? TryParseCoordinatesFromPrompt(string prompt)
        {
            if (string.IsNullOrEmpty(prompt))
                return null;

            float x = 0, y = 0, z = 0;
            bool foundX = false, foundY = false, foundZ = false;

            // Pattern 1: "x=0.3, y=0.15, z=0.1"
            string[] parts = prompt.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
            foreach (string part in parts)
            {
                string trimmed = part.Trim().ToLower();

                if (trimmed.StartsWith("x="))
                {
                    if (float.TryParse(trimmed.Substring(2), out x))
                        foundX = true;
                }
                else if (trimmed.StartsWith("y="))
                {
                    if (float.TryParse(trimmed.Substring(2), out y))
                        foundY = true;
                }
                else if (trimmed.StartsWith("z="))
                {
                    if (float.TryParse(trimmed.Substring(2), out z))
                        foundZ = true;
                }
            }

            if (foundX && foundY && foundZ)
            {
                return new Vector3(x, y, z);
            }

            // Pattern 2: "position 0.3 0.15 0.1" or just "0.3 0.15 0.1"
            string[] words = prompt.Split(
                new[] { ' ', ',', '\t', '\n' },
                StringSplitOptions.RemoveEmptyEntries
            );
            List<float> numbers = new List<float>();

            foreach (string word in words)
            {
                if (float.TryParse(word.Trim(), out float num))
                {
                    numbers.Add(num);
                }
            }

            if (numbers.Count >= 3)
            {
                return new Vector3(numbers[0], numbers[1], numbers[2]);
            }

            return null;
        }

        #endregion

        #region Public Accessors (for Custom Editor)

        public string Prompt
        {
            get => _prompt;
            set => _prompt = value;
        }

        public RagResult LastResult => _lastResult;

        public List<OperationInfo> RecentOperations => _recentOperations;

        #endregion
    }
}
