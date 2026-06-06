using System;

namespace PythonCommunication
{
    /// <summary>
    /// Data structure for LLM analysis results received from Python
    /// </summary>
    [Serializable]
    public class LLMResult
    {
        public string response;
        public string camera_id;
        public LLMMetadata metadata;
        public uint request_id; // Protocol V2: for request/response correlation
    }

    [Serializable]
    public class LLMMetadata
    {
        public string model;
        public float duration_seconds;
    }
}
