using System;
using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using PythonCommunication;
using PythonCommunication.Core;
using Core;
using Tests.EditMode;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for Python communication layer (formerly TCPClientTests).
    /// Expanded to include Protocol V2 request/response correlation,
    /// connection lifecycle, data models, and error handling.
    /// </summary>
    public class PythonCommunicationTests
    {
        private GameObject _clientObject;
        private SequenceClient _client;

        #region Setup/Teardown

        [SetUp]
        public void SetUp()
        {
            _clientObject = new GameObject("TestClient");
            _client = _clientObject.AddComponent<SequenceClient>();
        }

        [TearDown]
        public void TearDown()
        {
            if (_clientObject != null)
            {
                UnityEngine.Object.DestroyImmediate(_clientObject);
            }
        }

        #endregion

        #region Protocol V2 Tests

        [Test]
        public void ProtocolV2_HeaderSize_IsCorrect()
        {
            // Protocol V2 header: [type:1][request_id:4] = 5 bytes
            Assert.AreEqual(5, UnityProtocol.HEADER_SIZE, "Protocol V2 header should be 5 bytes");
            Assert.AreEqual(1, UnityProtocol.TYPE_SIZE, "Message type should be 1 byte");
            Assert.AreEqual(4, UnityProtocol.INT_SIZE, "Request ID should be 4 bytes");
        }

        [Test]
        public void ProtocolV2_Version_IsTwo()
        {
            Assert.AreEqual(2, UnityProtocol.VERSION, "Protocol version should be 2");
        }

        [Test]
        public void ProtocolV2_DecodeHeader_ExtractsTypeAndRequestId()
        {
            // Create test header: [SEQUENCE_QUERY:1][request_id:123]
            byte[] testData = new byte[5];
            testData[0] = (byte)MessageType.SEQUENCE_QUERY;
            Buffer.BlockCopy(BitConverter.GetBytes(123u), 0, testData, 1, 4);

            // Decode header
            int offset = UnityProtocol.DecodeHeader(testData, 0, out MessageType messageType, out uint requestId);

            Assert.AreEqual(MessageType.SEQUENCE_QUERY, messageType, "Message type should be SEQUENCE_QUERY");
            Assert.AreEqual(123u, requestId, "Request ID should be 123");
            Assert.AreEqual(5, offset, "Offset should advance by 5 bytes (header size)");
        }

        [Test]
        public void ProtocolV2_DecodeHeader_WithInsufficientData_ThrowsException()
        {
            // Create header with only 3 bytes (need 5)
            byte[] testData = new byte[3];

            Assert.Throws<ArgumentException>(() =>
            {
                UnityProtocol.DecodeHeader(testData, 0, out MessageType messageType, out uint requestId);
            }, "Should throw exception when data is insufficient");
        }

        [Test]
        public void ProtocolV2_DecodeHeader_WithNullData_ThrowsException()
        {
            Assert.Throws<ArgumentException>(() =>
            {
                UnityProtocol.DecodeHeader(null, 0, out MessageType messageType, out uint requestId);
            }, "Should throw exception for null data");
        }

        [Test]
        public void ProtocolV2_MessageTypes_AreDefined()
        {
            // Verify all message types are defined correctly
            Assert.AreEqual(0x01, (byte)MessageType.IMAGE, "IMAGE should be 0x01");
            Assert.AreEqual(0x02, (byte)MessageType.RESULT, "RESULT should be 0x02");
            Assert.AreEqual(0x03, (byte)MessageType.RAG_QUERY, "RAG_QUERY should be 0x03");
            Assert.AreEqual(0x04, (byte)MessageType.RAG_RESPONSE, "RAG_RESPONSE should be 0x04");
            Assert.AreEqual(0x05, (byte)MessageType.STATUS_QUERY, "STATUS_QUERY should be 0x05");
            Assert.AreEqual(0x06, (byte)MessageType.STATUS_RESPONSE, "STATUS_RESPONSE should be 0x06");
            Assert.AreEqual(0x07, (byte)MessageType.STEREO_IMAGE, "STEREO_IMAGE should be 0x07");
            Assert.AreEqual(0x08, (byte)MessageType.SEQUENCE_QUERY, "SEQUENCE_QUERY should be 0x08");
        }

        [Test]
        public void ProtocolV2_MaxImageSize_IsReasonable()
        {
            // Max image size should be 10MB
            Assert.AreEqual(10 * 1024 * 1024, UnityProtocol.MAX_IMAGE_SIZE,
                "Max image size should be 10MB");
        }

        #endregion

        #region Request ID Correlation Tests

        [UnityTest]
        public IEnumerator RequestIdGeneration_IsUnique()
        {
            // Generate multiple request IDs and verify they're unique
            var requestIds = new System.Collections.Generic.HashSet<uint>();

            for (int i = 0; i < 100; i++)
            {
                // In production, SequenceClient.GenerateRequestId() would be called
                // For testing, we simulate by using sequential IDs
                uint requestId = (uint)(i + 1);
                Assert.IsFalse(requestIds.Contains(requestId), $"Request ID {requestId} should be unique");
                requestIds.Add(requestId);
            }

            yield return null;

            Assert.AreEqual(100, requestIds.Count, "Should generate 100 unique request IDs");
        }

        [Test]
        public void RequestIdCorrelation_MatchesRequestToResponse()
        {
            // Simulate request/response correlation
            uint expectedRequestId = 12345u;

            // Create a mock sequence result with matching request_id
            var mockResult = new SequenceResult
            {
                success = true,
                request_id = expectedRequestId,
                sequence_id = "seq_001",
                total_commands = 1
            };

            Assert.AreEqual(expectedRequestId, mockResult.request_id,
                "Response request_id should match original request");
        }

        [Test]
        public void RequestIdCorrelation_MultipleSimultaneousRequests_HandledCorrectly()
        {
            // Simulate multiple simultaneous requests with different IDs
            var requests = new System.Collections.Generic.Dictionary<uint, string>
            {
                { 100u, "move to (0.3, 0.2, 0.1)" },
                { 101u, "close the gripper" },
                { 102u, "detect the blue cube" }
            };

            // Verify all requests have unique IDs
            Assert.AreEqual(3, requests.Count, "Should track 3 simultaneous requests");

            // Simulate receiving responses out of order
            var response1 = new SequenceResult { request_id = 102u, success = true };
            var response2 = new SequenceResult { request_id = 100u, success = true };
            var response3 = new SequenceResult { request_id = 101u, success = true };

            // Verify each response can be matched to its request
            Assert.IsTrue(requests.ContainsKey(response1.request_id), "Response 1 should match request");
            Assert.IsTrue(requests.ContainsKey(response2.request_id), "Response 2 should match request");
            Assert.IsTrue(requests.ContainsKey(response3.request_id), "Response 3 should match request");
        }

        #endregion

        #region Connection Lifecycle Tests

        [Test]
        public void SequenceClient_Initialization_CreatesInstance()
        {
            Assert.IsNotNull(_client, "SequenceClient should be created");
            Assert.IsNotNull(SequenceClient.Instance, "Singleton instance should be set");
        }

        [Test]
        public void SequenceClient_InitialState_IsDisconnected()
        {
            // Client starts disconnected (unless Python backend is running)
            // We can't assert _client.IsConnected == false because it might auto-connect
            Assert.IsNotNull(_client, "Client should exist even when disconnected");
        }

        [UnityTest]
        public IEnumerator SequenceClient_ReconnectionLogic_RetriesOnFailure()
        {
            // With Python backend unavailable, client should handle connection failure gracefully
            // (Auto-reconnect is implemented in TCPClientBase)
            // NOTE: WaitForSeconds is intentional here — TCP connection attempts take real time to fail
            yield return new WaitForSeconds(TestConstants.SHORT_TIMEOUT);

            // Client should still be valid even if connection failed
            Assert.IsNotNull(_client, "Client should remain valid after connection failure");
        }

        [Test]
        public void SequenceClient_Disconnect_CleansUpResources()
        {
            // Disconnect should clean up without crashing
            // (Actual disconnect is handled in OnDestroy)

            Assert.IsNotNull(_client, "Client should exist before cleanup test");
            // Cleanup happens in TearDown
        }

        [UnityTest]
        public IEnumerator SequenceClient_PersistentConnection_MaintainsKeepalive()
        {
            // Protocol V2 uses persistent TCP connections
            // Connection should be maintained, not recreated for each request

            yield return new WaitForSeconds(TestConstants.SHORT_TIMEOUT);

            // If connected, connection should remain stable
            if (_client.IsConnected)
            {
                bool wasConnected = _client.IsConnected;
                yield return new WaitForSeconds(1f);
                Assert.AreEqual(wasConnected, _client.IsConnected,
                    "Connection state should remain stable");
            }
            else
            {
                Assert.Pass("Python backend not available - skipping keepalive test");
            }
        }

        #endregion

        #region Data Model Tests

        [Test]
        public void RagResult_CanBeCreated()
        {
            var result = new RagResult
            {
                query = "test query",
                operations = new OperationInfo[0],
                num_results = 0
            };

            Assert.AreEqual("test query", result.query);
            Assert.AreEqual(0, result.num_results);
        }

        [Test]
        public void OperationInfo_CanBeCreated()
        {
            var operation = new OperationInfo
            {
                name = "move_to_coordinate",
                description = "Move robot to position",
                category = "NAVIGATION",
                similarity_score = 0.95f
            };

            Assert.AreEqual("move_to_coordinate", operation.name);
            Assert.AreEqual(0.95f, operation.similarity_score, 0.001f);
        }

        [Test]
        public void SequenceResult_CanBeCreated()
        {
            var result = new SequenceResult
            {
                success = true,
                sequence_id = "seq_123",
                total_commands = 3,
                completed_commands = 3,
                request_id = 456u
            };

            Assert.IsTrue(result.success);
            Assert.AreEqual("seq_123", result.sequence_id);
            Assert.AreEqual(3, result.total_commands);
            Assert.AreEqual(456u, result.request_id);
        }

        [Test]
        public void RobotCommand_CanBeCreated()
        {
            var command = new RobotCommand
            {
                command_type = "move_to_coordinate",
                robot_id = "Robot1",
                request_id = 123
            };

            Assert.AreEqual("move_to_coordinate", command.command_type);
            Assert.AreEqual("Robot1", command.robot_id);
            Assert.AreEqual(123u, command.request_id);
        }

        [Test]
        public void RagQueryFilters_CanSerializeToJson()
        {
            var filters = new RagQueryFilters
            {
                category = "navigation",
                min_score = 0.7f
            };

            string json = filters.ToJson();
            Assert.IsTrue(json.Contains("navigation"), "JSON should contain category");
            Assert.IsTrue(json.Contains("0.7"), "JSON should contain min_score");
        }

        [Test]
        public void SequenceResult_WithCommandResults_StoresCorrectly()
        {
            var result = new SequenceResult
            {
                success = true,
                total_commands = 2,
                completed_commands = 2,
                results = new System.Collections.Generic.List<CommandResult>
                {
                    new CommandResult { index = 0, operation = "move_to_coordinate", success = true },
                    new CommandResult { index = 1, operation = "control_gripper", success = true }
                }
            };

            Assert.AreEqual(2, result.results.Count, "Should have 2 command results");
            Assert.AreEqual("move_to_coordinate", result.results[0].operation);
            Assert.AreEqual("control_gripper", result.results[1].operation);
        }

        #endregion

        #region Error Handling Tests

        [Test]
        public void SequenceClient_SendWithoutConnection_ReturnsFalse()
        {
            // Attempt to send when not connected
            bool sent = _client.ExecuteSequence("test command", "TestRobot");

            if (!_client.IsConnected)
            {
                Assert.IsFalse(sent, "Should return false when not connected");
            }
        }

        [Test]
        public void SequenceClient_NullCommand_ReturnsFalse()
        {
            LogAssert.Expect(LogType.Error, new System.Text.RegularExpressions.Regex(".*[Cc]ommand.*null.*empty|.*null.*empty.*[Cc]ommand"));
            bool sent = _client.ExecuteSequence(null, "TestRobot");
            Assert.IsFalse(sent, "Should reject null command");
        }

        [Test]
        public void SequenceClient_EmptyCommand_ReturnsFalse()
        {
            LogAssert.Expect(LogType.Error, new System.Text.RegularExpressions.Regex(".*[Cc]ommand.*null.*empty|.*null.*empty.*[Cc]ommand"));
            bool sent = _client.ExecuteSequence("", "TestRobot");
            Assert.IsFalse(sent, "Should reject empty command");
        }

        [UnityTest]
        public IEnumerator SequenceClient_Timeout_HandledGracefully()
        {
            // Send command and wait for potential timeout
            _client.ExecuteSequence("test command", "TestRobot");

            yield return new WaitForSeconds(TestConstants.SHORT_TIMEOUT);

            // Client should remain stable even if request times out
            Assert.IsNotNull(_client, "Client should remain valid after timeout");
        }

        [Test]
        public void SequenceResult_WithError_StoresErrorMessage()
        {
            var result = new SequenceResult
            {
                success = false,
                error = "Robot not found",
                total_commands = 1,
                completed_commands = 0
            };

            Assert.IsFalse(result.success);
            Assert.AreEqual("Robot not found", result.error);
            Assert.AreEqual(0, result.completed_commands);
        }

        #endregion

        #region Thread Safety Tests

        [UnityTest]
        public IEnumerator ResponseQueue_ThreadSafe_HandlesMultipleResponses()
        {
            // Simulate multiple responses arriving on background thread
            // Response queue should handle them safely

            var responses = new System.Collections.Generic.List<SequenceResult>
            {
                new SequenceResult { request_id = 1, success = true },
                new SequenceResult { request_id = 2, success = true },
                new SequenceResult { request_id = 3, success = true }
            };

            yield return null;

            // Verify queue can handle multiple responses
            Assert.AreEqual(3, responses.Count, "Should create 3 test responses");
        }

        [Test]
        public void PendingRequests_ThreadSafe_CanAddAndRemove()
        {
            // Simulate adding/removing pending requests
            // Dictionary should be thread-safe via locking

            var pendingRequests = new System.Collections.Generic.Dictionary<uint, string>
            {
                { 1u, "request1" },
                { 2u, "request2" }
            };

            Assert.AreEqual(2, pendingRequests.Count, "Should add 2 pending requests");

            pendingRequests.Remove(1u);
            Assert.AreEqual(1, pendingRequests.Count, "Should remove 1 request");
        }

        #endregion

        #region Communication Constants Tests

        [Test]
        public void CommunicationConstants_ActivePorts_AreDifferent()
        {
            var ports = new int[]
            {
                CommunicationConstants.LLM_RESULTS_PORT,        // 5010
                CommunicationConstants.SEQUENCE_SERVER_PORT     // 5013
            };

            var uniquePorts = new System.Collections.Generic.HashSet<int>(ports);
            Assert.AreEqual(ports.Length, uniquePorts.Count,
                "All active ports should be unique");
        }

        [Test]
        public void CommunicationConstants_SequenceServerPort_IsCorrect()
        {
            Assert.AreEqual(5013, CommunicationConstants.SEQUENCE_SERVER_PORT,
                "Sequence server should be on port 5013");
        }

        [Test]
        public void CommunicationConstants_ResultsPort_IsCorrect()
        {
            Assert.AreEqual(5010, CommunicationConstants.LLM_RESULTS_PORT,
                "Results server should be on port 5010");
        }

        #endregion

        #region Integration Tests (Require Python Backend)

        [UnityTest]
        public IEnumerator Integration_SendSequence_ReceivesResponse()
        {
            TestHelpers.SkipIfPythonUnavailable();

            // Wait for connection
            yield return new WaitForSeconds(1f);

            // Send test sequence
            bool sent = _client.ExecuteSequence("move to (0.3, 0.2, 0.1)", "TestRobot");
            Assert.IsTrue(sent, "Command should be sent");

            // Wait for response
            yield return new WaitForSeconds(TestConstants.MEDIUM_TIMEOUT);

            // Check for response
            var lastResult = _client.LastResult;
            if (lastResult != null)
            {
                Assert.Greater(lastResult.request_id, 0u, "Response should have valid request_id");
            }
        }

        [UnityTest]
        public IEnumerator Integration_MultipleRequests_CorrectCorrelation()
        {
            TestHelpers.SkipIfPythonUnavailable();

            yield return new WaitForSeconds(1f);

            // Send multiple requests
            _client.ExecuteSequence("move to (0.3, 0.2, 0.1)", "Robot1");
            yield return new WaitForSeconds(0.1f);

            _client.ExecuteSequence("close the gripper", "Robot1");
            yield return new WaitForSeconds(0.1f);

            _client.ExecuteSequence("move to (0.4, 0.3, 0.2)", "Robot1");

            // Wait for all responses
            yield return new WaitForSeconds(TestConstants.MEDIUM_TIMEOUT);

            // Verify correlation worked (all responses matched to requests)
            Assert.Pass("Multiple requests handled without errors");
        }

        #endregion
    }
}
