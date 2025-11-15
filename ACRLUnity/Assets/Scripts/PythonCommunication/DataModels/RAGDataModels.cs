using System;

namespace PythonCommunication
{
    // ============================================================================
    // RAG QUERY AND RESPONSE DATA MODELS
    // ============================================================================

    /// <summary>
    /// Complete RAG query result from Python RAGServer
    /// Contains semantic search results for robot operations
    /// </summary>
    [Serializable]
    public class RagResult
    {
        public string query;
        public int num_results;
        public string summary;
        public OperationInfo[] operations;

        private const string _logPrefix = "[RAG_RESULT]";

        public override string ToString()
        {
            return $"{_logPrefix} Query='{query}', Results={num_results}";
        }
    }

    /// <summary>
    /// Information about a robot operation from the operations registry
    /// </summary>
    [Serializable]
    public class OperationInfo
    {
        public string operation_id;
        public string name;
        public string category;
        public string complexity;
        public string description;
        public float similarity_score;
        public OperationParameter[] parameters;
        public string[] usage_examples;
        public string[] preconditions;
        public string[] postconditions;
        public string[] failure_modes;

        private const string _logPrefix = "[OPERATION_INFO]";

        public override string ToString()
        {
            return $"{_logPrefix} {name} (score={similarity_score:F3}, category={category})";
        }
    }

    /// <summary>
    /// Parameter specification for a robot operation
    /// </summary>
    [Serializable]
    public class OperationParameter
    {
        public string name;
        public string type;
        public bool required;
        public string description;
        public string default_value;
        public ParameterValidation validation;

        private const string _logPrefix = "[OPERATION_PARAM]";

        public override string ToString()
        {
            string req = required ? "required" : "optional";
            return $"{_logPrefix} {name}: {type} ({req})";
        }
    }

    /// <summary>
    /// Validation constraints for operation parameters
    /// </summary>
    [Serializable]
    public class ParameterValidation
    {
        public float? min_value;
        public float? max_value;
        public string[] allowed_values;
        public string pattern;

        public bool HasMinValue => min_value.HasValue;
        public bool HasMaxValue => max_value.HasValue;
        public bool HasAllowedValues => allowed_values != null && allowed_values.Length > 0;
        public bool HasPattern => !string.IsNullOrEmpty(pattern);
    }

    /// <summary>
    /// Filters for RAG queries
    /// </summary>
    [Serializable]
    public class RagQueryFilters
    {
        public string category; // e.g., "navigation", "manipulation", "perception"
        public string complexity; // e.g., "basic", "intermediate", "advanced"
        public float min_score = 0.5f; // Minimum similarity score

        /// <summary>
        /// Convert to JSON string for protocol encoding
        /// </summary>
        public string ToJson()
        {
            var parts = new System.Collections.Generic.List<string>();

            if (!string.IsNullOrEmpty(category))
            {
                parts.Add($"\"category\": \"{category}\"");
            }

            if (!string.IsNullOrEmpty(complexity))
            {
                parts.Add($"\"complexity\": \"{complexity}\"");
            }

            parts.Add($"\"min_score\": {min_score}");

            return "{" + string.Join(", ", parts.ToArray()) + "}";
        }

        /// <summary>
        /// Create filters for a specific category
        /// </summary>
        public static RagQueryFilters ForCategory(string category, float minScore = 0.5f)
        {
            return new RagQueryFilters
            {
                category = category,
                min_score = minScore
            };
        }

        /// <summary>
        /// Create filters for a specific complexity level
        /// </summary>
        public static RagQueryFilters ForComplexity(string complexity, float minScore = 0.5f)
        {
            return new RagQueryFilters
            {
                complexity = complexity,
                min_score = minScore
            };
        }

        /// <summary>
        /// Create empty filters (no restrictions)
        /// </summary>
        public static RagQueryFilters None()
        {
            return new RagQueryFilters();
        }
    }

    /// <summary>
    /// Helper class for common RAG query patterns
    /// </summary>
    public static class RagQueryHelper
    {
        // Operation categories (must match Python side)
        public const string CATEGORY_NAVIGATION = "navigation";
        public const string CATEGORY_MANIPULATION = "manipulation";
        public const string CATEGORY_PERCEPTION = "perception";
        public const string CATEGORY_COORDINATION = "coordination";

        // Complexity levels (must match Python side)
        public const string COMPLEXITY_BASIC = "basic";
        public const string COMPLEXITY_INTERMEDIATE = "intermediate";
        public const string COMPLEXITY_ADVANCED = "advanced";
        public const string COMPLEXITY_EXPERT = "expert";

        /// <summary>
        /// Create query for navigation operations
        /// </summary>
        public static (string query, RagQueryFilters filters) NavigationQuery(string taskDescription)
        {
            return (taskDescription, RagQueryFilters.ForCategory(CATEGORY_NAVIGATION));
        }

        /// <summary>
        /// Create query for manipulation operations
        /// </summary>
        public static (string query, RagQueryFilters filters) ManipulationQuery(string taskDescription)
        {
            return (taskDescription, RagQueryFilters.ForCategory(CATEGORY_MANIPULATION));
        }

        /// <summary>
        /// Create query for perception operations
        /// </summary>
        public static (string query, RagQueryFilters filters) PerceptionQuery(string taskDescription)
        {
            return (taskDescription, RagQueryFilters.ForCategory(CATEGORY_PERCEPTION));
        }

        /// <summary>
        /// Create query for basic operations only
        /// </summary>
        public static (string query, RagQueryFilters filters) BasicOperationsQuery(string taskDescription)
        {
            return (taskDescription, RagQueryFilters.ForComplexity(COMPLEXITY_BASIC));
        }
    }
}
