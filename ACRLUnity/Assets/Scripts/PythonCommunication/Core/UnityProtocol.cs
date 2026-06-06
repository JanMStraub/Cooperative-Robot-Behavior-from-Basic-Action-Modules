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
        AUTORT_COMMAND = 0x09,
        AUTORT_RESPONSE = 0x0A,
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

        private const string _logPrefix = "[UNITY_PROTOCOL]";

        // Zero-allocation alternative to BitConverter.GetBytes(int).
        private static void WriteInt32LE(byte[] buffer, ref int offset, int value)
        {
            buffer[offset] = (byte)(value);
            buffer[offset + 1] = (byte)(value >> 8);
            buffer[offset + 2] = (byte)(value >> 16);
            buffer[offset + 3] = (byte)(value >> 24);
            offset += INT_SIZE;
        }

        private static byte[] EncodeHeader(MessageType messageType, uint requestId)
        {
            byte[] header = new byte[HEADER_SIZE];
            header[0] = (byte)messageType;
            // Direct byte writes — zero allocation, avoids BitConverter.GetBytes heap alloc
            header[1] = (byte)(requestId);
            header[2] = (byte)(requestId >> 8);
            header[3] = (byte)(requestId >> 16);
            header[4] = (byte)(requestId >> 24);
            return header;
        }

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

            // Explicit little-endian decode — matches EncodeHeader's bit-shift writes
            // and avoids BitConverter's platform-endian dependency.
            requestId =
                (uint)data[offset]
                | ((uint)data[offset + 1] << 8)
                | ((uint)data[offset + 2] << 16)
                | ((uint)data[offset + 3] << 24);
            offset += INT_SIZE;

            return offset;
        }

        /// <summary>Format: [type:1][request_id:4][camera_id_len:4][camera_id:N][prompt_len:4][prompt:N][image_len:4][image_data:N]</summary>
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

            WriteInt32LE(message, ref offset, cameraIdBytes.Length);
            Buffer.BlockCopy(cameraIdBytes, 0, message, offset, cameraIdBytes.Length);
            offset += cameraIdBytes.Length;

            WriteInt32LE(message, ref offset, promptBytes.Length);
            Buffer.BlockCopy(promptBytes, 0, message, offset, promptBytes.Length);
            offset += promptBytes.Length;

            WriteInt32LE(message, ref offset, imageBytes.Length);
            Buffer.BlockCopy(imageBytes, 0, message, offset, imageBytes.Length);

            return message;
        }

        /// <summary>Format: [type:1][request_id:4][pair_id_len:4][pair_id:N][cam_L_id_len:4][cam_L_id:N][cam_R_id_len:4][cam_R_id:N][prompt_len:4][prompt:N][img_L_len:4][img_L:N][img_R_len:4][img_R:N]</summary>
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

            WriteInt32LE(message, ref offset, pairIdBytes.Length);
            Buffer.BlockCopy(pairIdBytes, 0, message, offset, pairIdBytes.Length);
            offset += pairIdBytes.Length;

            WriteInt32LE(message, ref offset, leftIdBytes.Length);
            Buffer.BlockCopy(leftIdBytes, 0, message, offset, leftIdBytes.Length);
            offset += leftIdBytes.Length;

            WriteInt32LE(message, ref offset, rightIdBytes.Length);
            Buffer.BlockCopy(rightIdBytes, 0, message, offset, rightIdBytes.Length);
            offset += rightIdBytes.Length;

            WriteInt32LE(message, ref offset, promptBytes.Length);
            Buffer.BlockCopy(promptBytes, 0, message, offset, promptBytes.Length);
            offset += promptBytes.Length;

            WriteInt32LE(message, ref offset, leftImageBytes.Length);
            Buffer.BlockCopy(leftImageBytes, 0, message, offset, leftImageBytes.Length);
            offset += leftImageBytes.Length;

            WriteInt32LE(message, ref offset, rightImageBytes.Length);
            Buffer.BlockCopy(rightImageBytes, 0, message, offset, rightImageBytes.Length);

            return message;
        }

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

        /// <summary>Format: [type:1][request_id:4][json_len:4][json_data:N]</summary>
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

            WriteInt32LE(message, ref offset, jsonBytes.Length);

            Buffer.BlockCopy(jsonBytes, 0, message, offset, jsonBytes.Length);

            return message;
        }

        /// <summary>Format: [type:1][request_id:4][query_len:4][query_text:N][top_k:4][filters_json_len:4][filters_json:N]</summary>
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

            WriteInt32LE(message, ref offset, queryBytes.Length);
            Buffer.BlockCopy(queryBytes, 0, message, offset, queryBytes.Length);
            offset += queryBytes.Length;

            WriteInt32LE(message, ref offset, topK);

            WriteInt32LE(message, ref offset, filtersBytes.Length);
            Buffer.BlockCopy(filtersBytes, 0, message, offset, filtersBytes.Length);

            return message;
        }

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

        /// <summary>Format: [type:1][request_id:4][json_len:4][operation_context_json:N]</summary>
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

            WriteInt32LE(message, ref offset, jsonBytes.Length);
            Buffer.BlockCopy(jsonBytes, 0, message, offset, jsonBytes.Length);

            return message;
        }

        /// <summary>Format: [type:1][request_id:4][robot_id_len:4][robot_id:N][detailed:1]</summary>
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

            WriteInt32LE(message, ref offset, robotIdBytes.Length);
            Buffer.BlockCopy(robotIdBytes, 0, message, offset, robotIdBytes.Length);
            offset += robotIdBytes.Length;

            message[offset] = (byte)(detailed ? 1 : 0);

            return message;
        }

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

        /// <summary>Format: [type:1][request_id:4][json_len:4][robot_status_json:N]</summary>
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

            WriteInt32LE(message, ref offset, jsonBytes.Length);

            Buffer.BlockCopy(jsonBytes, 0, message, offset, jsonBytes.Length);

            return message;
        }

        /// <summary>Format: [type:1][request_id:4][cmd_type_len:4][cmd_type:N][params_json_len:4][params_json:N]</summary>
        public static byte[] EncodeAutoRTCommand(
            string commandType,
            string paramsJson = null,
            uint requestId = 0
        )
        {
            if (string.IsNullOrEmpty(commandType))
            {
                throw new ArgumentException($"{_logPrefix} Command type cannot be null or empty");
            }

            byte[] commandTypeBytes = Encoding.UTF8.GetBytes(commandType);

            if (string.IsNullOrEmpty(paramsJson))
            {
                paramsJson = "{}";
            }

            byte[] paramsBytes = Encoding.UTF8.GetBytes(paramsJson);

            int totalSize =
                HEADER_SIZE + INT_SIZE * 2 + commandTypeBytes.Length + paramsBytes.Length;
            byte[] message = new byte[totalSize];

            int offset = 0;

            byte[] header = EncodeHeader(MessageType.AUTORT_COMMAND, requestId);
            Buffer.BlockCopy(header, 0, message, offset, HEADER_SIZE);
            offset += HEADER_SIZE;

            Buffer.BlockCopy(
                BitConverter.GetBytes(commandTypeBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(commandTypeBytes, 0, message, offset, commandTypeBytes.Length);
            offset += commandTypeBytes.Length;

            Buffer.BlockCopy(
                BitConverter.GetBytes(paramsBytes.Length),
                0,
                message,
                offset,
                INT_SIZE
            );
            offset += INT_SIZE;
            Buffer.BlockCopy(paramsBytes, 0, message, offset, paramsBytes.Length);

            return message;
        }

        public static void DecodeAutoRTCommand(
            byte[] data,
            out uint requestId,
            out string commandType,
            out string paramsJson
        )
        {
            int offset = DecodeHeader(data, 0, out MessageType msgType, out requestId);

            if (msgType != MessageType.AUTORT_COMMAND)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Expected AUTORT_COMMAND message, got {msgType}"
                );
            }

            int commandTypeLen = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;
            commandType = Encoding.UTF8.GetString(data, offset, commandTypeLen);
            offset += commandTypeLen;

            int paramsLen = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;
            paramsJson = Encoding.UTF8.GetString(data, offset, paramsLen);
        }

        /// <summary>Format: [type:1][request_id:4][json_len:4][json:N]</summary>
        public static string DecodeAutoRTResponse(byte[] data, out uint requestId)
        {
            if (data == null)
            {
                throw new ArgumentException($"{_logPrefix} Data cannot be null");
            }

            if (data.Length < HEADER_SIZE + INT_SIZE)
            {
                throw new ArgumentException($"{_logPrefix} Invalid AutoRT response: too short");
            }

            int offset = DecodeHeader(data, 0, out MessageType msgType, out requestId);

            if (msgType != MessageType.AUTORT_RESPONSE)
            {
                throw new ArgumentException(
                    $"{_logPrefix} Expected AUTORT_RESPONSE message, got {msgType}"
                );
            }

            int jsonLength = BitConverter.ToInt32(data, offset);
            offset += INT_SIZE;

            if (data.Length < offset + jsonLength)
            {
                throw new ArgumentException($"{_logPrefix} Incomplete AutoRT response");
            }

            return Encoding.UTF8.GetString(data, offset, jsonLength);
        }

        public static byte[] EncodeAutoRTResponse(string responseJson, uint requestId = 0)
        {
            if (string.IsNullOrEmpty(responseJson))
            {
                throw new ArgumentException($"{_logPrefix} Response JSON cannot be null or empty");
            }

            byte[] jsonBytes = Encoding.UTF8.GetBytes(responseJson);

            byte[] message = new byte[HEADER_SIZE + INT_SIZE + jsonBytes.Length];
            int offset = 0;

            byte[] header = EncodeHeader(MessageType.AUTORT_RESPONSE, requestId);
            Buffer.BlockCopy(header, 0, message, offset, HEADER_SIZE);
            offset += HEADER_SIZE;

            Buffer.BlockCopy(BitConverter.GetBytes(jsonBytes.Length), 0, message, offset, INT_SIZE);
            offset += INT_SIZE;
            Buffer.BlockCopy(jsonBytes, 0, message, offset, jsonBytes.Length);

            return message;
        }

        public static bool IsValidImageSize(byte[] imageBytes)
        {
            if (imageBytes == null)
                return false;
            return imageBytes.Length > 0 && imageBytes.Length <= MAX_IMAGE_SIZE;
        }

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
    }
}
