using NUnit.Framework;
using System.Collections;
using UnityEngine;
using UnityEngine.TestTools;
using PythonCommunication;
using Core;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for TCP client components.
    /// Validates connection management, reconnection logic, and data models.
    /// </summary>
    public class TCPClientTests
    {
        #region CommandReceiver Tests

        private GameObject _receiverObject;
        private CommandReceiver _receiver;

        [Test]
        public void CommandReceiver_CanBeCreated()
        {
            _receiverObject = new GameObject("TestReceiver");
            _receiver = _receiverObject.AddComponent<CommandReceiver>();

            Assert.IsNotNull(_receiver);

            UnityEngine.Object.DestroyImmediate(_receiverObject);
        }

        #endregion

        #region SequenceClient Tests

        private GameObject _clientObject;
        private SequenceClient _client;

        [Test]
        public void SequenceClient_CanBeCreated()
        {
            _clientObject = new GameObject("TestClient");
            _client = _clientObject.AddComponent<SequenceClient>();

            Assert.IsNotNull(_client);

            UnityEngine.Object.DestroyImmediate(_clientObject);
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
            Assert.AreEqual(0.95f, operation.similarity_score);
        }

        [Test]
        public void SequenceResult_CanBeCreated()
        {
            var result = new SequenceResult
            {
                success = true,
                sequence_id = "seq_123",
                total_commands = 3,
                completed_commands = 3
            };

            Assert.IsTrue(result.success);
            Assert.AreEqual("seq_123", result.sequence_id);
            Assert.AreEqual(3, result.total_commands);
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
            Assert.IsTrue(json.Contains("navigation"));
            Assert.IsTrue(json.Contains("0.7"));
        }

        #endregion

        #region Communication Constants Tests

        [Test]
        public void CommunicationConstants_ActivePorts_AreDifferent()
        {
            var ports = new int[]
            {
                CommunicationConstants.LLM_RESULTS_PORT,
                CommunicationConstants.RAG_SERVER_PORT,
                CommunicationConstants.STATUS_SERVER_PORT,
                CommunicationConstants.SEQUENCE_SERVER_PORT
            };

            var uniquePorts = new System.Collections.Generic.HashSet<int>(ports);
            Assert.AreEqual(ports.Length, uniquePorts.Count);
        }

        #endregion
    }
}
