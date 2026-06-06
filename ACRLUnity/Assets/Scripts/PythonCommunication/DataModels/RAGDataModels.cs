using System;

namespace PythonCommunication
{
    /// <summary>
    /// Complete RAG query result from Python SequenceServer
    /// Contains semantic search results for robot operations
    /// (RAG is now integrated into SequenceServer, not a separate server)
    /// </summary>
    [Serializable]
    public class RagResult
    {
        public string query;
        public int num_results;
        public OperationInfo[] operations;
    }

    /// <summary>
    /// Information about a robot operation from the operations registry
    /// </summary>
    [Serializable]
    public class OperationInfo
    {
        public string name;
        public string category;
        public string description;
        public float similarity_score;
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
    }
}
