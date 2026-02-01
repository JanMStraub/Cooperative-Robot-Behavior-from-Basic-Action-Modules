using NUnit.Framework;
using System;
using PythonCommunication.Core;

namespace Tests.EditMode
{
    /// <summary>
    /// Tests for UnityProtocol wire protocol encoding/decoding.
    /// Validates Protocol V2 message format compatibility with Python.
    /// </summary>
    public class UnityProtocolTests
    {
        #region Constants Tests

        [Test]
        public void Protocol_HasCorrectVersion()
        {
            Assert.AreEqual(2, UnityProtocol.VERSION);
        }

        [Test]
        public void Protocol_HasCorrectHeaderSize()
        {
            Assert.AreEqual(5, UnityProtocol.HEADER_SIZE);
            Assert.AreEqual(1, UnityProtocol.TYPE_SIZE);
            Assert.AreEqual(4, UnityProtocol.INT_SIZE);
        }

        [Test]
        public void Protocol_HasCorrectLimits()
        {
            // No MAX_STRING_LENGTH - LLM responses can be arbitrarily large
            Assert.AreEqual(10 * 1024 * 1024, UnityProtocol.MAX_IMAGE_SIZE);
        }

        #endregion

        #region MessageType Tests

        [Test]
        public void MessageType_HasExpectedValues()
        {
            Assert.AreEqual(0x01, (byte)MessageType.IMAGE);
            Assert.AreEqual(0x02, (byte)MessageType.RESULT);
            Assert.AreEqual(0x03, (byte)MessageType.RAG_QUERY);
            Assert.AreEqual(0x04, (byte)MessageType.RAG_RESPONSE);
            Assert.AreEqual(0x05, (byte)MessageType.STATUS_QUERY);
            Assert.AreEqual(0x06, (byte)MessageType.STATUS_RESPONSE);
            Assert.AreEqual(0x07, (byte)MessageType.STEREO_IMAGE);
        }

        #endregion

        #region Image Message Tests

        [Test]
        public void EncodeImageMessage_CreatesValidMessage()
        {
            string cameraId = "Camera1";
            string prompt = "Describe this scene";
            byte[] imageBytes = new byte[] { 0x89, 0x50, 0x4E, 0x47 }; // PNG header
            uint requestId = 12345;

            byte[] message = UnityProtocol.EncodeImageMessage(cameraId, prompt, imageBytes, requestId);

            Assert.IsNotNull(message);
            Assert.Greater(message.Length, UnityProtocol.HEADER_SIZE);
        }

        [Test]
        public void EncodeDecodeImageMessage_RoundTrip()
        {
            string cameraId = "TestCamera";
            string prompt = "Test prompt";
            byte[] imageBytes = new byte[] { 1, 2, 3, 4, 5 };
            uint requestId = 42;

            byte[] encoded = UnityProtocol.EncodeImageMessage(cameraId, prompt, imageBytes, requestId);
            UnityProtocol.DecodeImageMessage(encoded, out uint decodedRequestId, out string decodedCameraId,
                out string decodedPrompt, out byte[] decodedImage);

            Assert.AreEqual(requestId, decodedRequestId);
            Assert.AreEqual(cameraId, decodedCameraId);
            Assert.AreEqual(prompt, decodedPrompt);
            Assert.AreEqual(imageBytes, decodedImage);
        }

        [Test]
        public void EncodeImageMessage_WithEmptyPrompt_Succeeds()
        {
            byte[] message = UnityProtocol.EncodeImageMessage("Cam1", "", new byte[] { 1, 2, 3 });
            Assert.IsNotNull(message);
        }

        [Test]
        public void EncodeImageMessage_WithNullPrompt_Succeeds()
        {
            byte[] message = UnityProtocol.EncodeImageMessage("Cam1", null, new byte[] { 1, 2, 3 });
            Assert.IsNotNull(message);
        }

        [Test]
        public void EncodeImageMessage_ThrowsOnEmptyCameraId()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeImageMessage("", "prompt", new byte[] { 1 }));
        }

        [Test]
        public void EncodeImageMessage_ThrowsOnNullCameraId()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeImageMessage(null, "prompt", new byte[] { 1 }));
        }

        [Test]
        public void EncodeImageMessage_ThrowsOnEmptyImage()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeImageMessage("Cam1", "prompt", new byte[0]));
        }

        [Test]
        public void EncodeImageMessage_ThrowsOnNullImage()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeImageMessage("Cam1", "prompt", null));
        }

        [Test]
        public void EncodeImageMessage_ThrowsOnOversizedImage()
        {
            byte[] oversized = new byte[UnityProtocol.MAX_IMAGE_SIZE + 1];
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeImageMessage("Cam1", "prompt", oversized));
        }

        #endregion

        #region Stereo Image Message Tests

        [Test]
        public void EncodeStereoImageMessage_CreatesValidMessage()
        {
            byte[] message = UnityProtocol.EncodeStereoImageMessage(
                "StereoPair1", "LeftCam", "RightCam", "prompt",
                new byte[] { 1, 2, 3 }, new byte[] { 4, 5, 6 }, 100);

            Assert.IsNotNull(message);
            Assert.Greater(message.Length, UnityProtocol.HEADER_SIZE);
        }

        [Test]
        public void EncodeStereoImageMessage_ThrowsOnEmptyPairId()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeStereoImageMessage("", "L", "R", "", new byte[] { 1 }, new byte[] { 1 }));
        }

        [Test]
        public void EncodeStereoImageMessage_ThrowsOnEmptyLeftImage()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeStereoImageMessage("P", "L", "R", "", new byte[0], new byte[] { 1 }));
        }

        #endregion

        #region Result Message Tests

        [Test]
        public void EncodeDecodeResultMessage_RoundTrip()
        {
            string json = "{\"success\":true,\"response\":\"Test\"}";
            uint requestId = 999;

            byte[] encoded = UnityProtocol.EncodeResultMessage(json, requestId);
            string decoded = UnityProtocol.DecodeResultMessage(encoded, out uint decodedRequestId);

            Assert.AreEqual(json, decoded);
            Assert.AreEqual(requestId, decodedRequestId);
        }

        [Test]
        public void EncodeResultMessage_ThrowsOnEmptyJson()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeResultMessage(""));
        }

        [Test]
        public void DecodeResultMessage_ThrowsOnShortData()
        {
            byte[] shortData = new byte[5];
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.DecodeResultMessage(shortData, out _));
        }

        [Test]
        public void DecodeResultMessage_ThrowsOnWrongMessageType()
        {
            byte[] encoded = UnityProtocol.EncodeRagResponse("{}", 0);
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.DecodeResultMessage(encoded, out _));
        }

        #endregion

        #region RAG Query Tests

        [Test]
        public void EncodeDecodeRagQuery_RoundTrip()
        {
            string query = "move robot to position";
            int topK = 5;
            string filters = "{\"category\":\"NAVIGATION\"}";
            uint requestId = 123;

            byte[] encoded = UnityProtocol.EncodeRagQuery(query, topK, filters, requestId);
            UnityProtocol.DecodeRagQuery(encoded, out uint decodedRequestId, out string decodedQuery,
                out int decodedTopK, out string decodedFilters);

            Assert.AreEqual(requestId, decodedRequestId);
            Assert.AreEqual(query, decodedQuery);
            Assert.AreEqual(topK, decodedTopK);
            Assert.AreEqual(filters, decodedFilters);
        }

        [Test]
        public void EncodeRagQuery_DefaultsToEmptyFilters()
        {
            byte[] encoded = UnityProtocol.EncodeRagQuery("test query", 3, null);
            UnityProtocol.DecodeRagQuery(encoded, out _, out _, out _, out string filters);
            Assert.AreEqual("{}", filters);
        }

        [Test]
        public void EncodeRagQuery_ThrowsOnEmptyQuery()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeRagQuery(""));
        }

        [Test]
        public void EncodeRagQuery_ThrowsOnInvalidTopK()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeRagQuery("query", 0));
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeRagQuery("query", 101));
        }

        #endregion

        #region RAG Response Tests

        [Test]
        public void EncodeDecodeRagResponse_RoundTrip()
        {
            string context = "{\"operations\":[{\"name\":\"move_to_coordinate\"}]}";
            uint requestId = 456;

            byte[] encoded = UnityProtocol.EncodeRagResponse(context, requestId);
            string decoded = UnityProtocol.DecodeRagResponse(encoded, out uint decodedRequestId);

            Assert.AreEqual(context, decoded);
            Assert.AreEqual(requestId, decodedRequestId);
        }

        [Test]
        public void EncodeRagResponse_ThrowsOnEmptyJson()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeRagResponse(""));
        }

        #endregion

        #region Status Query Tests

        [Test]
        public void EncodeDecodeStatusQuery_RoundTrip()
        {
            string robotId = "Robot1";
            bool detailed = true;
            uint requestId = 789;

            byte[] encoded = UnityProtocol.EncodeStatusQuery(robotId, detailed, requestId);
            UnityProtocol.DecodeStatusQuery(encoded, out uint decodedRequestId,
                out string decodedRobotId, out bool decodedDetailed);

            Assert.AreEqual(requestId, decodedRequestId);
            Assert.AreEqual(robotId, decodedRobotId);
            Assert.AreEqual(detailed, decodedDetailed);
        }

        [Test]
        public void EncodeStatusQuery_ThrowsOnEmptyRobotId()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeStatusQuery(""));
        }

        #endregion

        #region Status Response Tests

        [Test]
        public void EncodeDecodeStatusResponse_RoundTrip()
        {
            string statusJson = "{\"robotId\":\"Robot1\",\"isMoving\":true}";
            uint requestId = 321;

            byte[] encoded = UnityProtocol.EncodeStatusResponse(statusJson, requestId);
            string decoded = UnityProtocol.DecodeStatusResponse(encoded, out uint decodedRequestId);

            Assert.AreEqual(statusJson, decoded);
            Assert.AreEqual(requestId, decodedRequestId);
        }

        [Test]
        public void EncodeStatusResponse_ThrowsOnEmptyJson()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.EncodeStatusResponse("", 0));
        }

        #endregion

        #region Validation Helper Tests

        // Removed: IsValidStringLength test - no string length limit for LLM responses

        [Test]
        public void IsValidImageSize_ValidatesCorrectly()
        {
            Assert.IsTrue(UnityProtocol.IsValidImageSize(new byte[] { 1 }));
            Assert.IsTrue(UnityProtocol.IsValidImageSize(new byte[UnityProtocol.MAX_IMAGE_SIZE]));
            Assert.IsFalse(UnityProtocol.IsValidImageSize(null));
            Assert.IsFalse(UnityProtocol.IsValidImageSize(new byte[0]));
            Assert.IsFalse(UnityProtocol.IsValidImageSize(new byte[UnityProtocol.MAX_IMAGE_SIZE + 1]));
        }

        [Test]
        public void PeekMessageType_ReturnsCorrectType()
        {
            byte[] imageMsg = UnityProtocol.EncodeImageMessage("cam", "p", new byte[] { 1 });
            Assert.AreEqual(MessageType.IMAGE, UnityProtocol.PeekMessageType(imageMsg));

            byte[] resultMsg = UnityProtocol.EncodeResultMessage("{}");
            Assert.AreEqual(MessageType.RESULT, UnityProtocol.PeekMessageType(resultMsg));
        }

        [Test]
        public void PeekRequestId_ReturnsCorrectId()
        {
            uint requestId = 12345;
            byte[] message = UnityProtocol.EncodeResultMessage("{}", requestId);
            Assert.AreEqual(requestId, UnityProtocol.PeekRequestId(message));
        }

        [Test]
        public void PeekMessageType_ThrowsOnShortData()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.PeekMessageType(new byte[0]));
        }

        [Test]
        public void PeekRequestId_ThrowsOnShortData()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.PeekRequestId(new byte[3]));
        }

        #endregion

        #region Header Tests

        [Test]
        public void DecodeHeader_ThrowsOnInsufficientData()
        {
            Assert.Throws<ArgumentException>(() =>
                UnityProtocol.DecodeHeader(new byte[3], 0, out _, out _));
        }

        [Test]
        public void DecodeHeader_RespectsOffset()
        {
            byte[] data = new byte[10];
            data[2] = (byte)MessageType.RESULT;
            BitConverter.GetBytes((uint)999).CopyTo(data, 3);

            int newOffset = UnityProtocol.DecodeHeader(data, 2, out MessageType type, out uint requestId);

            Assert.AreEqual(MessageType.RESULT, type);
            Assert.AreEqual(999u, requestId);
            Assert.AreEqual(7, newOffset);
        }

        #endregion

        #region UTF-8 Encoding Tests

        [Test]
        public void ImageMessage_HandlesUnicodeCorrectly()
        {
            string cameraId = "Camera_\u00e4\u00f6\u00fc"; // German umlauts
            string prompt = "Test \u4e2d\u6587"; // Chinese characters
            byte[] image = new byte[] { 1, 2, 3 };

            byte[] encoded = UnityProtocol.EncodeImageMessage(cameraId, prompt, image);
            UnityProtocol.DecodeImageMessage(encoded, out _, out string decodedCamera,
                out string decodedPrompt, out _);

            Assert.AreEqual(cameraId, decodedCamera);
            Assert.AreEqual(prompt, decodedPrompt);
        }

        #endregion
    }
}
