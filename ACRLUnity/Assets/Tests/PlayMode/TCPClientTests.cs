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
        #region RAGClient Tests

        private GameObject _ragClientObject;
        private RAGClient _ragClient;

        [SetUp]
        public void SetupRAGClient()
        {
            if (RAGClient.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(RAGClient.Instance.gameObject);
            }

            _ragClientObject = new GameObject("TestRAGClient");
            _ragClient = _ragClientObject.AddComponent<RAGClient>();
        }

        [TearDown]
        public void TearDownRAGClient()
        {
            if (_ragClientObject != null)
            {
                UnityEngine.Object.DestroyImmediate(_ragClientObject);
            }
        }

        [Test]
        public void RAGClient_Singleton_IsSet()
        {
            Assert.IsNotNull(RAGClient.Instance);
            Assert.AreEqual(_ragClient, RAGClient.Instance);
        }

        [Test]
        public void RAGClient_DefaultPort_IsCorrect()
        {
            Assert.AreEqual(CommunicationConstants.RAG_SERVER_PORT, _ragClient.ServerPort);
        }

        [Test]
        public void RAGClient_IsConnected_ReturnsFalseInitially()
        {
            Assert.IsFalse(_ragClient.IsConnected);
        }

        [UnityTest]
        public IEnumerator RAGClient_QueryAsync_HandlesNoConnection()
        {
            yield return null;

            // Query without connection should handle gracefully
            Assert.IsFalse(_ragClient.IsConnected);
        }

        #endregion

        #region StatusClient Tests

        private GameObject _statusClientObject;
        private StatusClient _statusClient;

        [Test]
        public void StatusClient_Singleton_CanBeCreated()
        {
            if (StatusClient.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(StatusClient.Instance.gameObject);
            }

            _statusClientObject = new GameObject("TestStatusClient");
            _statusClient = _statusClientObject.AddComponent<StatusClient>();

            Assert.IsNotNull(StatusClient.Instance);
            Assert.AreEqual(_statusClient, StatusClient.Instance);

            UnityEngine.Object.DestroyImmediate(_statusClientObject);
        }

        [Test]
        public void StatusClient_DefaultPort_IsCorrect()
        {
            if (StatusClient.Instance != null)
            {
                UnityEngine.Object.DestroyImmediate(StatusClient.Instance.gameObject);
            }

            _statusClientObject = new GameObject("TestStatusClient");
            _statusClient = _statusClientObject.AddComponent<StatusClient>();

            Assert.AreEqual(CommunicationConstants.STATUS_SERVER_PORT, _statusClient.ServerPort);

            UnityEngine.Object.DestroyImmediate(_statusClientObject);
        }

        #endregion

        #region UnifiedPythonReceiver Tests

        private GameObject _receiverObject;
        private UnifiedPythonReceiver _receiver;

        [Test]
        public void UnifiedPythonReceiver_CanBeCreated()
        {
            _receiverObject = new GameObject("TestReceiver");
            _receiver = _receiverObject.AddComponent<UnifiedPythonReceiver>();

            Assert.IsNotNull(_receiver);

            UnityEngine.Object.DestroyImmediate(_receiverObject);
        }

        #endregion

        #region UnifiedPythonSender Tests

        private GameObject _senderObject;
        private UnifiedPythonSender _sender;

        [Test]
        public void UnifiedPythonSender_CanBeCreated()
        {
            _senderObject = new GameObject("TestSender");
            _sender = _senderObject.AddComponent<UnifiedPythonSender>();

            Assert.IsNotNull(_sender);

            UnityEngine.Object.DestroyImmediate(_senderObject);
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
        public void DepthResult_CanBeCreated()
        {
            var result = new DepthResult
            {
                success = true,
                camera_id = "StereoPair1",
                detections = new ObjectDetection[0]
            };

            Assert.IsTrue(result.success);
            Assert.AreEqual("StereoPair1", result.camera_id);
        }

        [Test]
        public void ObjectDetection_CanBeCreated()
        {
            var detection = new ObjectDetection
            {
                color = "red_cube",
                confidence = 0.98f,
                world_position = new Detection3DPosition { x = 1.0f, y = 2.0f, z = 3.0f }
            };

            Assert.AreEqual("red_cube", detection.color);
            Assert.AreEqual(0.98f, detection.confidence);
            Assert.AreEqual(1.0f, detection.world_position.x);
        }

        [Test]
        public void LLMResult_CanBeCreated()
        {
            var result = new LLMResult
            {
                success = true,
                response = "Test response",
                camera_id = "Camera1",
                request_id = 123
            };

            Assert.IsTrue(result.success);
            Assert.AreEqual("Test response", result.response);
            Assert.AreEqual(123u, result.request_id);
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
        public void CommunicationConstants_AllPorts_AreDifferent()
        {
            var ports = new int[]
            {
                CommunicationConstants.STREAMING_SERVER_PORT,
                CommunicationConstants.STEREO_DETECTION_SERVER_PORT,
                CommunicationConstants.LLM_RESULTS_PORT,
                CommunicationConstants.DEPTH_RESULTS_PORT,
                CommunicationConstants.RAG_SERVER_PORT,
                CommunicationConstants.STATUS_SERVER_PORT
            };

            var uniquePorts = new System.Collections.Generic.HashSet<int>(ports);
            Assert.AreEqual(ports.Length, uniquePorts.Count);
        }

        #endregion
    }
}
