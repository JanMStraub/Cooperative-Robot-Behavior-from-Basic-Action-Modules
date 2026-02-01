using System;
using System.Text;

namespace PythonCommunication.Core
{
    /// <summary>
    /// Message type enumeration for Protocol V2.
    /// Must match Python MessageType enum exactly.
    /// </summary>
    public enum MessageType : byte
    {
        IMAGE = 0x01,
        RESULT = 0x02,
        RAG_QUERY = 0x03,
        RAG_RESPONSE = 0x04,
        STATUS_QUERY = 0x05,
        STATUS_RESPONSE = 0x06,
        STEREO_IMAGE = 0x07,
        SEQUENCE_QUERY = 0x08,
    }

    /// <summary>
    /// Wire protocol for Unity ↔ Python LLM communication (Protocol V2).
    /// Matches the Python implementation in core/UnityProtocol.py.
    ///
    /// ALL messages now include a 5-byte header:
    /// - message_type (1 byte): Identifies the message type
    /// - request_id (4 bytes): Unsigned integer for request/response correlation
    ///
    /// This enables robust request tracking, timeout handling, and proper message routing.
    /// </summary>
    public static class UnityProtocol
    {
        public const int VERSION = 2;
        public const int INT_SIZE = 4;
        public const int TYPE_SIZE = 1;
        public const int HEADER_SIZE = TYPE_SIZE + INT_SIZE;
        public const int MAX_IMAGE_SIZE = 10 * 1024 * 1024;

        private const string _logPrefix = "[UNITY_PROTOCOL_V2]";

        #region Header Encoding/Decoding

        /// <summary>
        /// Encode message header (type + request_id).
        /// </summary>
        /// <param name="messageType">Message type from MessageType enum</param>
        /// <param name="requestId">Unsigned 32-bit request ID for correlation</param>
        /// <returns>5-byte header</returns>
        private static byte[] EncodeHeader(MessageType messageType, uint requestId)
        {
            byte[] header = new byte[HEADER_SIZE];
            header[0] = (byte)messageType;
            Buffer.BlockCopy(BitConverter.GetBytes(requestId), 0, header, TYPE_SIZE, INT_SIZE);
            return header;
        }

        /// <summary>
        /// Decode message header from data.
        /// </summary>
        /// <param name="data">Byte array containing header</param>
        /// <param name="offset">Starting offset (default 0)</param>
        /// <param name="messageType">Decoded message type</param>
        /// <param name="requestId">Decoded request ID</param>
        /// <returns>New offset after header</returns>
        public static int DecodeHeader(
            byte[] data,
            int offset,
            out MessageType messageType,
            out uint requestId
        )
        {
            if (data == null)
            {
                throw new ArgumentException($"{_logPrefix} Data cannot be null");
            }

            if (data.Length - offset < HEADER_SIZE)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Not enough data for header (need {HEADER_SIZE}, have {data.Length - offset})"
                );
            }

            messageType = (MessageType)data[offset];
            offset += TYPE_SIZE;

            requestId = BitConverter.ToUInt32(data, offset);
            offset += INT_SIZE;

            return offset;
        }

        #endregion

        #region Image Messages (Unity → Python)

        /// <summary>
        /// Encode an image message for sending to Python ImageServer.
        /// Format: [type:1][request_id:4][camera_id_len:4][camera_id:N][prompt_len:4][prompt:N][image_len:4][image_data:N]
        /// </summary>
        /// <param name="cameraId">Camera identifier</param>
        /// <param name="prompt">LLM prompt (can be empty)</param>
        /// <param name="imageBytes">Encoded image data (PNG/JPG)</param>
        /// <param name="requestId">Request ID for correlation (default 0)</param>
        /// <returns>Encoded message bytes</returns>
        public static byte[] EncodeImageMessage(
            string cameraId,
            string prompt,
            byte[] imageBytes,
            uint requestId = 0
        )
        {
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

            if (prompt == null)
            {
                prompt = "";
            }

            byte[] cameraIdBytes = Encoding.UTF8.GetBytes(cameraId);
            byte[] promptBytes = Encoding.UTF8.GetBytes(prompt);

            int totalSize =
                HEADER_SIZE
                + INT_SIZE * 3
                + cameraIdBytes.Length
                + promptBytes.Length
                + imageBytes.Length;
            byte[] message = new byte[totalSize];

            int offset = 0;

            byte[] header = EncodeHeader(MessageType.IMAGE, requestId);
            Buffer.BlockCopy(header, 0, message, offset, HEADER_SIZE);
            offset += HEADER_SIZE;

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

        /// <summary>
        /// Encode stereo image pair message for Python stereo detection server (Protocol V2).
        /// Format: [type:1][request_id:4][pair_id_len:4][pair_id:N][cam_L_id_len:4][cam_L_id:N]
        /// [cam_R_id_len:4][cam_R_id:N][prompt_len:4][prompt:N][img_L_len:4][img_L:N][img_R_len:4][img_R:N]
        /// </summary>
        /// <param name="cameraPairId">Camera pair identifier</param>
        /// <param name="cameraLeftId">Left camera identifier</param>
        /// <param name="cameraRightId">Right camera identifier</param>
        /// <param name="prompt">LLM prompt (can be empty)</param>
        /// <param name="leftImageBytes">Encoded left image data (PNG/JPG)</param>
        /// <param name="rightImageBytes">Encoded right image data (PNG/JPG)</param>
        /// <param name="requestId">Request ID for correlation (default 0)</param>
        /// <returns>Encoded message bytes</returns>
        public static byte[] EncodeStereoImageMessage(
            string cameraPairId,
            string cameraLeftId,
            string cameraRightId,
            string prompt,
            byte[] leftImageBytes,
            byte[] rightImageBytes,
            uint requestId = 0
        )
        {
            if (string.IsNullOrEmpty(cameraPairId))
            {
                throw new ArgumentException($"{_logPrefix} Camera pair ID cannot be null or empty");
            }

            if (string.IsNullOrEmpty(cameraLeftId))
            {
                throw new ArgumentException($"{_logPrefix} Camera left ID cannot be null or empty");
            }

            if (string.IsNullOrEmpty(cameraRightId))
            {
                throw new ArgumentException(
                    $"{_logPrefix} Camera right ID cannot be null or empty"
                );
            }

            if (leftImageBytes == null || leftImageBytes.Length == 0)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Left image bytes cannot be null or empty"
                );
            }

            if (rightImageBytes == null || rightImageBytes.Length == 0)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Right image bytes cannot be null or empty"
                );
            }

            if (leftImageBytes.Length > MAX_IMAGE_SIZE)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Left image size {leftImageBytes.Length} exceeds maximum {MAX_IMAGE_SIZE}"
                );
            }

            if (rightImageBytes.Length > MAX_IMAGE_SIZE)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Right image size {rightImageBytes.Length} exceeds maximum {MAX_IMAGE_SIZE}"
                );
            }

            if (prompt == null)
            {
                prompt = "";
            }

            byte[] pairIdBytes = Encoding.UTF8.GetBytes(cameraPairId);
            byte[] leftIdBytes = Encoding.UTF8.GetBytes(cameraLeftId);
            byte[] rightIdBytes = Encoding.UTF8.GetBytes(cameraRightId);
            byte[] promptBytes = Encoding.UTF8.GetBytes(prompt);

            int totalSize =
                HEADER_SIZE
                + INT_SIZE * 7
                + pairIdBytes.Length
                + leftIdBytes.Length
                + rightIdBytes.Length
                + promptBytes.Length
                + leftImageBytes.Length
                + rightImageBytes.Length;
            byte[] message = new byte[totalSize];

            int offset = 0;

            byte[] header = EncodeHeader(MessageType.STEREO_IMAGE, requestId);
            Buffer.BlockCopy(header, 0, message, offset, HEADER_SIZE);
            offset += HEADER_SIZE;

            Buffer.BlockCopy(
                BitConverter.GetBytes(pairIdBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(pairIdBytes, 0, message, offset, pairIdBytes.Length);
            offset += pairIdBytes.Length;

            Buffer.BlockCopy(
                BitConverter.GetBytes(leftIdBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(leftIdBytes, 0, message, offset, leftIdBytes.Length);
            offset += leftIdBytes.Length;

            Buffer.BlockCopy(
                BitConverter.GetBytes(rightIdBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(rightIdBytes, 0, message, offset, rightIdBytes.Length);
            offset += rightIdBytes.Length;

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

            Buffer.BlockCopy(
                BitConverter.GetBytes(leftImageBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(leftImageBytes, 0, message, offset, leftImageBytes.Length);
            offset += leftImageBytes.Length;

            Buffer.BlockCopy(
                BitConverter.GetBytes(rightImageBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(rightImageBytes, 0, message, offset, rightImageBytes.Length);

            return message;
        }

        /// <summary>
        /// Decode an image message (for testing).
        /// </summary>
        /// <param name="data">Raw message bytes</param>
        /// <param name="requestId">Decoded request ID</param>
        /// <param name="cameraId">Decoded camera ID</param>
        /// <param name="prompt">Decoded prompt</param>
        /// <param name="imageBytes">Decoded image data</param>
        public static void DecodeImageMessage(
            byte[] data,
            out uint requestId,
            out string cameraId,
            out string prompt,
            out byte[] imageBytes
        )
        {
            int offset = DecodeHeader(data, 0, out MessageType msgType, out requestId);

            if (msgType != MessageType.IMAGE)
            {
                throw new ArgumentException($"{_logPrefix} Expected IMAGE message, got {msgType}");
            }

            int cameraIdLen = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;
            cameraId = Encoding.UTF8.GetString(data, offset, cameraIdLen);
            offset += cameraIdLen;

            int promptLen = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;
            prompt = Encoding.UTF8.GetString(data, offset, promptLen);
            offset += promptLen;

            int imageLen = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;
            imageBytes = new byte[imageLen];
            Buffer.BlockCopy(data, offset, imageBytes, 0, imageLen);
        }

        #endregion

        #region Result Messages (Python → Unity)

        /// <summary>
        /// Decode a result message received from Python CommandServer or SequenceServer.
        /// Format: [type:1][request_id:4][json_len:4][json_data:N]
        /// </summary>
        /// <param name="data">Raw message bytes</param>
        /// <param name="requestId">Decoded request ID for correlation</param>
        /// <returns>JSON string</returns>
        public static string DecodeResultMessage(byte[] data, out uint requestId)
        {
            if (data == null)
            {
                throw new ArgumentException($"{_logPrefix} Data cannot be null");
            }

            if (data.Length < HEADER_SIZE + INT_SIZE)
            {
                throw new ArgumentException($"{_logPrefix} Invalid result message: too short");
            }

            int offset = DecodeHeader(data, 0, out MessageType msgType, out requestId);

            if (msgType != MessageType.RESULT)
            {
                throw new ArgumentException($"{_logPrefix} Expected RESULT message, got {msgType}");
            }

            int jsonLength = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;

            if (jsonLength <= 0 || jsonLength > MAX_IMAGE_SIZE)
            {
                throw new ArgumentException($"{_logPrefix} Invalid JSON length: {jsonLength}");
            }

            if (data.Length < offset + jsonLength)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Incomplete message: expected {offset + jsonLength}, got {data.Length}"
                );
            }

            return Encoding.UTF8.GetString(data, offset, jsonLength);
        }

        /// <summary>
        /// Encode a result message for sending to Python (used for testing).
        /// Format: [type:1][request_id:4][json_len:4][json_data:N]
        /// </summary>
        /// <param name="json">JSON string to encode</param>
        /// <param name="requestId">Request ID for correlation (default 0)</param>
        /// <returns>Encoded message bytes</returns>
        public static byte[] EncodeResultMessage(string json, uint requestId = 0)
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

            byte[] message = new byte[HEADER_SIZE + INT_SIZE + jsonBytes.Length];
            int offset = 0;

            byte[] header = EncodeHeader(MessageType.RESULT, requestId);
            Buffer.BlockCopy(header, 0, message, offset, HEADER_SIZE);
            offset += HEADER_SIZE;

            Buffer.BlockCopy(BitConverter.GetBytes(jsonBytes.Length), 0, message, offset, INT_SIZE);
            offset += INT_SIZE;

            Buffer.BlockCopy(jsonBytes, 0, message, offset, jsonBytes.Length);

            return message;
        }

        #endregion

        #region RAG Query Messages (Unity → Python)

        /// <summary>
        /// Encode a RAG query message for sending to Python SequenceServer (integrated RAG).
        /// Format: [type:1][request_id:4][query_len:4][query_text:N][top_k:4][filters_json_len:4][filters_json:N]
        /// </summary>
        /// <param name="query">Natural language query text</param>
        /// <param name="topK">Number of results to return (1-100)</param>
        /// <param name="filtersJson">Optional filters JSON (can be null or empty)</param>
        /// <param name="requestId">Request ID for correlation (default 0)</param>
        /// <returns>Encoded message bytes</returns>
        public static byte[] EncodeRagQuery(
            string query,
            int topK = 5,
            string filtersJson = null,
            uint requestId = 0
        )
        {
            if (string.IsNullOrEmpty(query))
            {
                throw new ArgumentException($"{_logPrefix} RAG query cannot be null or empty");
            }

            if (topK < 1 || topK > 100)
            {
                throw new ArgumentException(
                    $"{_logPrefix} topK must be between 1 and 100, got {topK}"
                );
            }

            byte[] queryBytes = Encoding.UTF8.GetBytes(query);

            if (string.IsNullOrEmpty(filtersJson))
            {
                filtersJson = "{}";
            }

            byte[] filtersBytes = Encoding.UTF8.GetBytes(filtersJson);

            int totalSize = HEADER_SIZE + INT_SIZE * 3 + queryBytes.Length + filtersBytes.Length;
            byte[] message = new byte[totalSize];

            int offset = 0;

            byte[] header = EncodeHeader(MessageType.RAG_QUERY, requestId);
            Buffer.BlockCopy(header, 0, message, offset, HEADER_SIZE);
            offset += HEADER_SIZE;

            Buffer.BlockCopy(
                BitConverter.GetBytes(queryBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(queryBytes, 0, message, offset, queryBytes.Length);
            offset += queryBytes.Length;

            Buffer.BlockCopy(BitConverter.GetBytes(topK), 0, message, offset, INT_SIZE);
            offset += INT_SIZE;

            Buffer.BlockCopy(
                BitConverter.GetBytes(filtersBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(filtersBytes, 0, message, offset, filtersBytes.Length);

            return message;
        }

        /// <summary>
        /// Decode a RAG query message (for testing).
        /// </summary>
        public static void DecodeRagQuery(
            byte[] data,
            out uint requestId,
            out string query,
            out int topK,
            out string filtersJson
        )
        {
            int offset = DecodeHeader(data, 0, out MessageType msgType, out requestId);

            if (msgType != MessageType.RAG_QUERY)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Expected RAG_QUERY message, got {msgType}"
                );
            }

            int queryLen = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;
            query = Encoding.UTF8.GetString(data, offset, queryLen);
            offset += queryLen;

            topK = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;

            int filtersLen = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;
            filtersJson = Encoding.UTF8.GetString(data, offset, filtersLen);
        }

        #endregion

        #region RAG Response Messages (Python → Unity)

        /// <summary>
        /// Decode a RAG response message received from Python SequenceServer (integrated RAG).
        /// Format: [type:1][request_id:4][json_len:4][operation_context_json:N]
        /// </summary>
        /// <param name="data">Raw message bytes</param>
        /// <param name="requestId">Decoded request ID for correlation</param>
        /// <returns>JSON string with operation context</returns>
        public static string DecodeRagResponse(byte[] data, out uint requestId)
        {
            if (data == null)
            {
                throw new ArgumentException($"{_logPrefix} Data cannot be null");
            }

            if (data.Length < HEADER_SIZE + INT_SIZE)
            {
                throw new ArgumentException($"{_logPrefix} Invalid RAG response: too short");
            }

            int offset = DecodeHeader(data, 0, out MessageType msgType, out requestId);

            if (msgType != MessageType.RAG_RESPONSE)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Expected RAG_RESPONSE message, got {msgType}"
                );
            }

            int jsonLength = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;

            if (data.Length < offset + jsonLength)
            {
                throw new ArgumentException($"{_logPrefix} Incomplete RAG response");
            }

            return Encoding.UTF8.GetString(data, offset, jsonLength);
        }

        /// <summary>
        /// Encode a RAG response message (for testing).
        /// </summary>
        public static byte[] EncodeRagResponse(string operationContextJson, uint requestId = 0)
        {
            if (string.IsNullOrEmpty(operationContextJson))
            {
                throw new ArgumentException(
                    $"{_logPrefix} Operation context JSON cannot be null or empty"
                );
            }

            byte[] jsonBytes = Encoding.UTF8.GetBytes(operationContextJson);

            byte[] message = new byte[HEADER_SIZE + INT_SIZE + jsonBytes.Length];
            int offset = 0;

            byte[] header = EncodeHeader(MessageType.RAG_RESPONSE, requestId);
            Buffer.BlockCopy(header, 0, message, offset, HEADER_SIZE);
            offset += HEADER_SIZE;

            Buffer.BlockCopy(BitConverter.GetBytes(jsonBytes.Length), 0, message, offset, INT_SIZE);
            offset += INT_SIZE;
            Buffer.BlockCopy(jsonBytes, 0, message, offset, jsonBytes.Length);

            return message;
        }

        #endregion

        #region Status Query Messages (Unity → Python)

        /// <summary>
        /// Encode a status query message for sending to Python CommandServer.
        /// Format: [type:1][request_id:4][robot_id_len:4][robot_id:N][detailed:1]
        /// </summary>
        /// <param name="robotId">Robot identifier (e.g., "Robot1", "AR4_Robot")</param>
        /// <param name="detailed">If true, request detailed joint information</param>
        /// <param name="requestId">Request ID for correlation (default 0)</param>
        /// <returns>Encoded message bytes</returns>
        public static byte[] EncodeStatusQuery(
            string robotId,
            bool detailed = false,
            uint requestId = 0
        )
        {
            if (string.IsNullOrEmpty(robotId))
            {
                throw new ArgumentException($"{_logPrefix} Robot ID cannot be null or empty");
            }

            byte[] robotIdBytes = Encoding.UTF8.GetBytes(robotId);

            int totalSize = HEADER_SIZE + INT_SIZE + robotIdBytes.Length + 1;
            byte[] message = new byte[totalSize];

            int offset = 0;

            byte[] header = EncodeHeader(MessageType.STATUS_QUERY, requestId);
            Buffer.BlockCopy(header, 0, message, offset, HEADER_SIZE);
            offset += HEADER_SIZE;

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

            message[offset] = (byte)(detailed ? 1 : 0);

            return message;
        }

        /// <summary>
        /// Decode a status query message (for testing).
        /// </summary>
        public static void DecodeStatusQuery(
            byte[] data,
            out uint requestId,
            out string robotId,
            out bool detailed
        )
        {
            int offset = DecodeHeader(data, 0, out MessageType msgType, out requestId);

            if (msgType != MessageType.STATUS_QUERY)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Expected STATUS_QUERY message, got {msgType}"
                );
            }

            int robotIdLen = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;
            robotId = Encoding.UTF8.GetString(data, offset, robotIdLen);
            offset += robotIdLen;

            detailed = data[offset] != 0;
        }

        #endregion

        #region Status Response Messages (Python → Unity / Unity → Python)

        /// <summary>
        /// Decode a status response message received from Python CommandServer or sent by WorldStatePublisher.
        /// Format: [type:1][request_id:4][json_len:4][robot_status_json:N]
        /// </summary>
        /// <param name="data">Raw message bytes</param>
        /// <param name="requestId">Decoded request ID for correlation</param>
        /// <returns>JSON string with robot status</returns>
        public static string DecodeStatusResponse(byte[] data, out uint requestId)
        {
            if (data == null)
            {
                throw new ArgumentException($"{_logPrefix} Data cannot be null");
            }

            if (data.Length < HEADER_SIZE + INT_SIZE)
            {
                throw new ArgumentException($"{_logPrefix} Invalid status response: too short");
            }

            int offset = DecodeHeader(data, 0, out MessageType msgType, out requestId);

            if (msgType != MessageType.STATUS_RESPONSE)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Expected STATUS_RESPONSE message, got {msgType}"
                );
            }

            int jsonLength = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;

            if (data.Length < offset + jsonLength)
            {
                throw new ArgumentException($"{_logPrefix} Incomplete status response");
            }

            return Encoding.UTF8.GetString(data, offset, jsonLength);
        }

        /// <summary>
        /// Encode a status response message for sending to Python (Unity → Python status response).
        /// Format: [type:1][request_id:4][json_len:4][robot_status_json:N]
        /// </summary>
        /// <param name="statusJson">JSON string containing robot status</param>
        /// <param name="requestId">Request ID for correlation (must match query request ID)</param>
        /// <returns>Encoded message bytes</returns>
        public static byte[] EncodeStatusResponse(string statusJson, uint requestId)
        {
            if (string.IsNullOrEmpty(statusJson))
            {
                throw new ArgumentException($"{_logPrefix} Status JSON cannot be null or empty");
            }

            byte[] jsonBytes = Encoding.UTF8.GetBytes(statusJson);

            byte[] message = new byte[HEADER_SIZE + INT_SIZE + jsonBytes.Length];
            int offset = 0;

            byte[] header = EncodeHeader(MessageType.STATUS_RESPONSE, requestId);
            Buffer.BlockCopy(header, 0, message, offset, HEADER_SIZE);
            offset += HEADER_SIZE;

            Buffer.BlockCopy(BitConverter.GetBytes(jsonBytes.Length), 0, message, offset, INT_SIZE);
            offset += INT_SIZE;

            Buffer.BlockCopy(jsonBytes, 0, message, offset, jsonBytes.Length);

            return message;
        }

        #endregion

        #region Validation Helpers

        /// <summary>
        /// Validate that image data is within the protocol's size limits
        /// </summary>
        public static bool IsValidImageSize(byte[] imageBytes)
        {
            if (imageBytes == null)
                return false;
            return imageBytes.Length > 0 && imageBytes.Length <= MAX_IMAGE_SIZE;
        }

        /// <summary>
        /// Peek at the message type without decoding the full message.
        /// </summary>
        public static MessageType PeekMessageType(byte[] data)
        {
            if (data == null)
            {
                throw new ArgumentException($"{_logPrefix} Data cannot be null");
            }

            if (data.Length < TYPE_SIZE)
            {
                throw new ArgumentException($"{_logPrefix} Data too short to peek message type");
            }

            return (MessageType)data[0];
        }

        /// <summary>
        /// Peek at the request ID without decoding the full message.
        /// </summary>
        public static uint PeekRequestId(byte[] data)
        {
            if (data == null)
            {
                throw new ArgumentException($"{_logPrefix} Data cannot be null");
            }

            if (data.Length < HEADER_SIZE)
            {
                throw new ArgumentException($"{_logPrefix} Data too short to peek request ID");
            }

            return BitConverter.ToUInt32(data, TYPE_SIZE);
        }

        #endregion
    }
}
