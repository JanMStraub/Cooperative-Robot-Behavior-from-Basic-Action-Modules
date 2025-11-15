using System;
using System.Text;

namespace PythonCommunication.Core
{
    /// <summary>
    /// Wire protocol for Unity ↔ Python LLM communication.
    /// Matches the Python implementation in core/UnityProtocol.py.
    /// Provides consistent encoding/decoding for all message types.
    /// </summary>
    public static class UnityProtocol
    {
        // Protocol constants (must match Python version)
        public const int VERSION = 1;
        public const int INT_SIZE = 4;
        public const int MAX_STRING_LENGTH = 256;
        public const int MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB

        // Helper variables
        private const string _logPrefix = "[UNITY_PROTOCOL]";

        #region Image Messages (Unity → Python)

        /// <summary>
        /// Encode an image message for sending to Python StreamingServer.
        /// Format: [camera_id_len][camera_id][prompt_len][prompt][image_len][image_data]
        /// </summary>
        /// <param name="cameraId">Camera identifier</param>
        /// <param name="prompt">LLM prompt (can be empty)</param>
        /// <param name="imageBytes">Encoded image data (PNG/JPG)</param>
        /// <returns>Encoded message bytes</returns>
        public static byte[] EncodeImageMessage(string cameraId, string prompt, byte[] imageBytes)
        {
            // Validate inputs
            if (string.IsNullOrEmpty(cameraId))
            {
                throw new ArgumentException($"{_logPrefix} Camera ID cannot be null or empty");
            }

            if (imageBytes == null || imageBytes.Length == 0)
            {
                throw new ArgumentException($"{_logPrefix} Image bytes cannot be null or empty");
            }

            if (imageBytes.Length > MAX_IMAGE_SIZE)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Image size {imageBytes.Length} exceeds maximum {MAX_IMAGE_SIZE}"
                );
            }

            // Ensure prompt is not null
            if (prompt == null)
            {
                prompt = "";
            }

            // Encode strings to UTF-8
            byte[] cameraIdBytes = Encoding.UTF8.GetBytes(cameraId);
            byte[] promptBytes = Encoding.UTF8.GetBytes(prompt);

            // Validate string lengths
            if (cameraIdBytes.Length > MAX_STRING_LENGTH)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Camera ID too long: {cameraIdBytes.Length} > {MAX_STRING_LENGTH}"
                );
            }

            if (promptBytes.Length > MAX_STRING_LENGTH)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Prompt too long: {promptBytes.Length} > {MAX_STRING_LENGTH}"
                );
            }

            // Calculate total message size
            int totalSize =
                INT_SIZE * 3 + cameraIdBytes.Length + promptBytes.Length + imageBytes.Length;
            byte[] message = new byte[totalSize];

            int offset = 0;

            // Write camera ID length and data
            Buffer.BlockCopy(
                BitConverter.GetBytes(cameraIdBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(cameraIdBytes, 0, message, offset, cameraIdBytes.Length);
            offset += cameraIdBytes.Length;

            // Write prompt length and data
            Buffer.BlockCopy(
                BitConverter.GetBytes(promptBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(promptBytes, 0, message, offset, promptBytes.Length);
            offset += promptBytes.Length;

            // Write image length and data
            Buffer.BlockCopy(
                BitConverter.GetBytes(imageBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(imageBytes, 0, message, offset, imageBytes.Length);

            return message;
        }

        #endregion

        #region Result Messages (Python → Unity)

        /// <summary>
        /// Decode a result message received from Python ResultsServer.
        /// Format: [json_len][json_data]
        /// </summary>
        /// <param name="data">Raw message bytes</param>
        /// <returns>JSON string</returns>
        public static string DecodeResultMessage(byte[] data)
        {
            if (data == null || data.Length < INT_SIZE)
            {
                throw new ArgumentException($"{_logPrefix} Invalid result message: too short");
            }

            // Read JSON length
            int jsonLength = BitConverter.ToInt32(data, 0);

            if (jsonLength <= 0 || jsonLength > MAX_IMAGE_SIZE)
            {
                throw new ArgumentException($"{_logPrefix} Invalid JSON length: {jsonLength}");
            }

            if (data.Length < INT_SIZE + jsonLength)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Incomplete message: expected {INT_SIZE + jsonLength}, got {data.Length}"
                );
            }

            // Extract JSON data
            byte[] jsonBytes = new byte[jsonLength];
            Buffer.BlockCopy(data, INT_SIZE, jsonBytes, 0, jsonLength);

            return Encoding.UTF8.GetString(jsonBytes);
        }

        /// <summary>
        /// Encode a result message for sending (used for testing).
        /// Format: [json_len][json_data]
        /// </summary>
        /// <param name="json">JSON string to encode</param>
        /// <returns>Encoded message bytes</returns>
        public static byte[] EncodeResultMessage(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                throw new ArgumentException($"{_logPrefix} JSON cannot be null or empty");
            }

            byte[] jsonBytes = Encoding.UTF8.GetBytes(json);

            if (jsonBytes.Length > MAX_IMAGE_SIZE)
            {
                throw new ArgumentException(
                    $"{_logPrefix} JSON too large: {jsonBytes.Length} > {MAX_IMAGE_SIZE}"
                );
            }

            byte[] message = new byte[INT_SIZE + jsonBytes.Length];

            // Write JSON length
            Buffer.BlockCopy(BitConverter.GetBytes(jsonBytes.Length), 0, message, 0, INT_SIZE);

            // Write JSON data
            Buffer.BlockCopy(jsonBytes, 0, message, INT_SIZE, jsonBytes.Length);

            return message;
        }

        #endregion

        #region RAG Query Messages (Unity → Python)

        /// <summary>
        /// Encode a RAG query message for sending to Python RAGServer.
        /// Format: [query_len][query_text][top_k][filters_json_len][filters_json]
        /// </summary>
        /// <param name="query">Natural language query text</param>
        /// <param name="topK">Number of results to return (1-100)</param>
        /// <param name="filtersJson">Optional filters JSON (can be null or empty)</param>
        /// <returns>Encoded message bytes</returns>
        public static byte[] EncodeRagQuery(string query, int topK = 5, string filtersJson = null)
        {
            // Validate inputs
            if (string.IsNullOrEmpty(query))
            {
                throw new ArgumentException($"{_logPrefix} RAG query cannot be null or empty");
            }

            if (topK < 1 || topK > 100)
            {
                throw new ArgumentException($"{_logPrefix} topK must be between 1 and 100, got {topK}");
            }

            // Encode strings to UTF-8
            byte[] queryBytes = Encoding.UTF8.GetBytes(query);

            if (queryBytes.Length > MAX_STRING_LENGTH)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Query too long: {queryBytes.Length} > {MAX_STRING_LENGTH}"
                );
            }

            // Handle filters (can be null or empty)
            if (string.IsNullOrEmpty(filtersJson))
            {
                filtersJson = "{}"; // Empty JSON object
            }

            byte[] filtersBytes = Encoding.UTF8.GetBytes(filtersJson);

            // Calculate total message size
            int totalSize = INT_SIZE * 3 + queryBytes.Length + filtersBytes.Length;
            byte[] message = new byte[totalSize];

            int offset = 0;

            // Write query length and data
            Buffer.BlockCopy(BitConverter.GetBytes(queryBytes.Length), 0, message, offset, INT_SIZE);
            offset += INT_SIZE;
            Buffer.BlockCopy(queryBytes, 0, message, offset, queryBytes.Length);
            offset += queryBytes.Length;

            // Write top_k
            Buffer.BlockCopy(BitConverter.GetBytes(topK), 0, message, offset, INT_SIZE);
            offset += INT_SIZE;

            // Write filters length and data
            Buffer.BlockCopy(BitConverter.GetBytes(filtersBytes.Length), 0, message, offset, INT_SIZE);
            offset += INT_SIZE;
            Buffer.BlockCopy(filtersBytes, 0, message, offset, filtersBytes.Length);

            return message;
        }

        #endregion

        #region RAG Response Messages (Python → Unity)

        /// <summary>
        /// Decode a RAG response message received from Python RAGServer.
        /// Format: [json_len][operation_context_json]
        /// This is identical to DecodeResultMessage but kept separate for clarity.
        /// </summary>
        /// <param name="data">Raw message bytes</param>
        /// <returns>JSON string with operation context</returns>
        public static string DecodeRagResponse(byte[] data)
        {
            return DecodeResultMessage(data);
        }

        #endregion

        #region Status Query Messages (Unity → Python)

        /// <summary>
        /// Encode a status query message for sending to Python StatusServer.
        /// Format: [robot_id_len][robot_id][detailed]
        /// </summary>
        /// <param name="robotId">Robot identifier (e.g., "Robot1", "AR4_Robot")</param>
        /// <param name="detailed">If true, request detailed joint information</param>
        /// <returns>Encoded message bytes</returns>
        public static byte[] EncodeStatusQuery(string robotId, bool detailed = false)
        {
            // Validate inputs
            if (string.IsNullOrEmpty(robotId))
            {
                throw new ArgumentException($"{_logPrefix} Robot ID cannot be null or empty");
            }

            // Encode robot_id to UTF-8
            byte[] robotIdBytes = Encoding.UTF8.GetBytes(robotId);

            if (robotIdBytes.Length > MAX_STRING_LENGTH)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Robot ID too long: {robotIdBytes.Length} > {MAX_STRING_LENGTH}"
                );
            }

            // Calculate total message size: [robot_id_len:4][robot_id:N][detailed:1]
            int totalSize = INT_SIZE + robotIdBytes.Length + 1;
            byte[] message = new byte[totalSize];

            int offset = 0;

            // Write robot ID length and data
            Buffer.BlockCopy(
                BitConverter.GetBytes(robotIdBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(robotIdBytes, 0, message, offset, robotIdBytes.Length);
            offset += robotIdBytes.Length;

            // Write detailed flag (1 byte: 0 or 1)
            message[offset] = (byte)(detailed ? 1 : 0);

            return message;
        }

        #endregion

        #region Status Response Messages (Python → Unity)

        /// <summary>
        /// Decode a status response message received from Python StatusServer.
        /// Format: [json_len][robot_status_json]
        /// This is identical to DecodeResultMessage but kept separate for clarity.
        /// </summary>
        /// <param name="data">Raw message bytes</param>
        /// <returns>JSON string with robot status</returns>
        public static string DecodeStatusResponse(byte[] data)
        {
            return DecodeResultMessage(data);
        }

        /// <summary>
        /// Encode a status response message for sending to Python (Unity → Python status response).
        /// Format: [json_len][robot_status_json]
        /// </summary>
        /// <param name="statusJson">JSON string containing robot status</param>
        /// <returns>Encoded message bytes</returns>
        public static byte[] EncodeStatusResponse(string statusJson)
        {
            return EncodeResultMessage(statusJson);
        }

        #endregion

        #region Validation Helpers

        /// <summary>
        /// Validate that a string is within the protocol's length limits
        /// </summary>
        public static bool IsValidStringLength(string str)
        {
            if (str == null)
                return true; // Null is treated as empty string
            return Encoding.UTF8.GetByteCount(str) <= MAX_STRING_LENGTH;
        }

        /// <summary>
        /// Validate that image data is within the protocol's size limits
        /// </summary>
        public static bool IsValidImageSize(byte[] imageBytes)
        {
            if (imageBytes == null)
                return false;
            return imageBytes.Length > 0 && imageBytes.Length <= MAX_IMAGE_SIZE;
        }

        #endregion
    }
}
