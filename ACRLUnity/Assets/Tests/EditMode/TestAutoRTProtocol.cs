using NUnit.Framework;
using PythonCommunication.Core;
using PythonCommunication.DataModels;
using System;
using System.Collections.Generic;
using UnityEngine;

namespace Tests.EditMode
{
    /// <summary>
    /// Unit tests for AutoRT protocol encoding/decoding.
    /// Tests UnityProtocol AutoRT message types and data model serialization.
    /// </summary>
    public class TestAutoRTProtocol
    {
        #region Protocol Encoding/Decoding Tests

        [Test]
        public void TestAutoRTCommand_Generate_EncodeDecode()
        {
            // Arrange
            string commandType = "generate";
            string paramsJson = "{\"num_tasks\":5,\"robot_ids\":[\"Robot1\",\"Robot2\"],\"strategy\":\"balanced\"}";
            uint requestId = 12345;

            // Act - Encode
            byte[] encoded = UnityProtocol.EncodeAutoRTCommand(commandType, paramsJson, requestId);

            // Assert - Verify header
            Assert.Greater(encoded.Length, UnityProtocol.HEADER_SIZE, "Encoded message should be larger than header");
            Assert.AreEqual((byte)MessageType.AUTORT_COMMAND, encoded[0], "Message type should be AUTORT_COMMAND");

            // Act - Decode
            UnityProtocol.DecodeAutoRTCommand(encoded, out uint decodedRequestId, out string decodedCommand, out string decodedParams);

            // Assert - Verify decoded values
            Assert.AreEqual(requestId, decodedRequestId, "Request ID should match");
            Assert.AreEqual(commandType, decodedCommand, "Command type should match");
            Assert.AreEqual(paramsJson, decodedParams, "Params JSON should match");
        }

        [Test]
        public void TestAutoRTCommand_StartLoop_EncodeDecode()
        {
            // Arrange
            string commandType = "start_loop";
            string paramsJson = "{\"loop_delay\":5.0,\"robot_ids\":[\"Robot1\"],\"strategy\":\"explore\"}";
            uint requestId = 99999;

            // Act
            byte[] encoded = UnityProtocol.EncodeAutoRTCommand(commandType, paramsJson, requestId);
            UnityProtocol.DecodeAutoRTCommand(encoded, out uint decodedRequestId, out string decodedCommand, out string decodedParams);

            // Assert
            Assert.AreEqual(requestId, decodedRequestId);
            Assert.AreEqual(commandType, decodedCommand);
            Assert.AreEqual(paramsJson, decodedParams);
        }

        [Test]
        public void TestAutoRTCommand_ExecuteTask_EncodeDecode()
        {
            // Arrange
            string commandType = "execute_task";
            string paramsJson = "{\"task_id\":\"task_abc123\"}";
            uint requestId = 54321;

            // Act
            byte[] encoded = UnityProtocol.EncodeAutoRTCommand(commandType, paramsJson, requestId);
            UnityProtocol.DecodeAutoRTCommand(encoded, out uint decodedRequestId, out string decodedCommand, out string decodedParams);

            // Assert
            Assert.AreEqual(requestId, decodedRequestId);
            Assert.AreEqual("execute_task", decodedCommand);
            Assert.True(decodedParams.Contains("task_abc123"), "Params should contain task ID");
        }

        [Test]
        public void TestAutoRTCommand_EmptyParams()
        {
            // Arrange
            string commandType = "stop_loop";
            string paramsJson = "{}";
            uint requestId = 11111;

            // Act
            byte[] encoded = UnityProtocol.EncodeAutoRTCommand(commandType, paramsJson, requestId);
            UnityProtocol.DecodeAutoRTCommand(encoded, out uint decodedRequestId, out string decodedCommand, out string decodedParams);

            // Assert
            Assert.AreEqual(requestId, decodedRequestId);
            Assert.AreEqual("stop_loop", decodedCommand);
            Assert.AreEqual("{}", decodedParams);
        }

        [Test]
        public void TestAutoRTCommand_NullParams()
        {
            // Arrange
            string commandType = "get_status";
            uint requestId = 22222;

            // Act - Null should be converted to "{}"
            byte[] encoded = UnityProtocol.EncodeAutoRTCommand(commandType, null, requestId);
            UnityProtocol.DecodeAutoRTCommand(encoded, out uint decodedRequestId, out string decodedCommand, out string decodedParams);

            // Assert
            Assert.AreEqual(requestId, decodedRequestId);
            Assert.AreEqual("get_status", decodedCommand);
            Assert.AreEqual("{}", decodedParams);
        }

        [Test]
        public void TestAutoRTCommand_InvalidCommandType()
        {
            // Arrange
            string emptyCommand = "";

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
            {
                UnityProtocol.EncodeAutoRTCommand(emptyCommand, "{}", 0);
            }, "Empty command type should throw exception");
        }

        [Test]
        public void TestAutoRTResponse_Success_EncodeDecode()
        {
            // Arrange
            string responseJson = @"{
                ""success"": true,
                ""tasks"": [
                    {
                        ""task_id"": ""task_123"",
                        ""description"": ""Move to position"",
                        ""operations"": [],
                        ""required_robots"": [""Robot1""],
                        ""estimated_complexity"": 3,
                        ""reasoning"": ""Test task""
                    }
                ],
                ""loop_running"": false,
                ""error"": null
            }";
            uint requestId = 33333;

            // Act - Encode
            byte[] encoded = UnityProtocol.EncodeAutoRTResponse(responseJson, requestId);

            // Assert - Verify header
            Assert.Greater(encoded.Length, UnityProtocol.HEADER_SIZE, "Encoded message should be larger than header");
            Assert.AreEqual((byte)MessageType.AUTORT_RESPONSE, encoded[0], "Message type should be AUTORT_RESPONSE");

            // Act - Decode
            string decoded = UnityProtocol.DecodeAutoRTResponse(encoded, out uint decodedRequestId);

            // Assert
            Assert.AreEqual(requestId, decodedRequestId, "Request ID should match");
            Assert.IsNotEmpty(decoded, "Decoded JSON should not be empty");
            Assert.True(decoded.Contains("task_123"), "Should contain task ID");
        }

        [Test]
        public void TestAutoRTResponse_Error_EncodeDecode()
        {
            // Arrange
            string responseJson = @"{
                ""success"": false,
                ""tasks"": [],
                ""loop_running"": false,
                ""error"": ""Task generation failed""
            }";
            uint requestId = 44444;

            // Act
            byte[] encoded = UnityProtocol.EncodeAutoRTResponse(responseJson, requestId);
            string decoded = UnityProtocol.DecodeAutoRTResponse(encoded, out uint decodedRequestId);

            // Assert
            Assert.AreEqual(requestId, decodedRequestId);
            Assert.True(decoded.Contains("Task generation failed"), "Should contain error message");
        }

        [Test]
        public void TestAutoRTResponse_EmptyTasks()
        {
            // Arrange
            string responseJson = @"{
                ""success"": true,
                ""tasks"": [],
                ""loop_running"": true,
                ""error"": null
            }";
            uint requestId = 55555;

            // Act
            byte[] encoded = UnityProtocol.EncodeAutoRTResponse(responseJson, requestId);
            string decoded = UnityProtocol.DecodeAutoRTResponse(encoded, out uint decodedRequestId);

            // Assert
            Assert.AreEqual(requestId, decodedRequestId);
            Assert.True(decoded.Contains("\"tasks\":"), "Should contain tasks field");
        }

        [Test]
        public void TestAutoRTResponse_InvalidJson()
        {
            // Arrange
            string emptyJson = "";

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
            {
                UnityProtocol.EncodeAutoRTResponse(emptyJson, 0);
            }, "Empty JSON should throw exception");
        }

        #endregion

        #region Data Model Tests

        [Test]
        public void TestProposedTask_Initialization()
        {
            // Arrange & Act
            ProposedTask task = new ProposedTask
            {
                task_id = "task_001",
                description = "Test task",
                operations = new List<TaskOperation>(),
                required_robots = new List<string> { "Robot1" },
                estimated_complexity = 2,
                reasoning = "Simple test"
            };

            // Assert
            Assert.AreEqual("task_001", task.task_id);
            Assert.AreEqual("Test task", task.description);
            Assert.NotNull(task.operations);
            Assert.AreEqual(1, task.required_robots.Count);
            Assert.AreEqual(2, task.estimated_complexity);
        }

        [Test]
        public void TestProposedTask_GetSummary()
        {
            // Arrange
            ProposedTask task = new ProposedTask
            {
                description = "Move object",
                estimated_complexity = 3
            };

            // Act
            string summary = task.GetSummary();

            // Assert
            Assert.True(summary.Contains("Complexity 3"), "Summary should contain complexity");
            Assert.True(summary.Contains("Move object"), "Summary should contain description");
        }

        [Test]
        public void TestProposedTask_RobotCount()
        {
            // Arrange
            ProposedTask task = new ProposedTask
            {
                required_robots = new List<string> { "Robot1", "Robot2" }
            };

            // Act
            int count = task.RobotCount;

            // Assert
            Assert.AreEqual(2, count);
        }

        [Test]
        public void TestProposedTask_OperationCount()
        {
            // Arrange
            ProposedTask task = new ProposedTask
            {
                operations = new List<TaskOperation>
                {
                    new TaskOperation { type = "move" },
                    new TaskOperation { type = "grasp" },
                    new TaskOperation { type = "release" }
                }
            };

            // Act
            int count = task.OperationCount;

            // Assert
            Assert.AreEqual(3, count);
        }

        [Test]
        public void TestTaskOperation_Initialization()
        {
            // Arrange & Act
            TaskOperation op = new TaskOperation
            {
                type = "move_to_coordinate",
                robot_id = "Robot1",
                parameters = new Dictionary<string, object>
                {
                    { "x", 0.5f },
                    { "y", 0.3f },
                    { "z", 0.2f }
                }
            };

            // Assert
            Assert.AreEqual("move_to_coordinate", op.type);
            Assert.AreEqual("Robot1", op.robot_id);
            Assert.AreEqual(3, op.parameters.Count);
        }

        [Test]
        public void TestAutoRTResponse_Initialization()
        {
            // Arrange & Act
            AutoRTResponse response = new AutoRTResponse
            {
                success = true,
                tasks = new List<ProposedTask>
                {
                    new ProposedTask { task_id = "task_1" },
                    new ProposedTask { task_id = "task_2" }
                },
                loop_running = false,
                error = null,
                request_id = 12345
            };

            // Assert
            Assert.True(response.success);
            Assert.AreEqual(2, response.tasks.Count);
            Assert.False(response.loop_running);
            Assert.AreEqual(12345, response.request_id);
        }

        [Test]
        public void TestAutoRTResponse_HasError_True()
        {
            // Arrange
            AutoRTResponse response = new AutoRTResponse
            {
                success = false,
                error = "Generation failed"
            };

            // Act & Assert
            Assert.True(response.HasError, "Should have error when success is false");
        }

        [Test]
        public void TestAutoRTResponse_HasError_False()
        {
            // Arrange
            AutoRTResponse response = new AutoRTResponse
            {
                success = true,
                error = null
            };

            // Act & Assert
            Assert.False(response.HasError, "Should not have error when success is true");
        }

        [Test]
        public void TestAutoRTResponse_ErrorMessage()
        {
            // Arrange
            AutoRTResponse errorResponse = new AutoRTResponse
            {
                success = false,
                error = "Test error"
            };

            AutoRTResponse successResponse = new AutoRTResponse
            {
                success = true,
                error = null
            };

            // Act & Assert
            Assert.AreEqual("Test error", errorResponse.ErrorMessage);
            Assert.AreEqual("No error", successResponse.ErrorMessage);
        }

        [Test]
        public void TestTaskSelectionStrategy_EnumValues()
        {
            // Assert - Verify all enum values exist
            Assert.AreEqual(0, (int)TaskSelectionStrategy.Balanced);
            Assert.AreEqual(1, (int)TaskSelectionStrategy.Explore);
            Assert.AreEqual(2, (int)TaskSelectionStrategy.Exploit);
            Assert.AreEqual(3, (int)TaskSelectionStrategy.Random);
        }

        #endregion

        #region JSON Serialization Tests

        [Test]
        public void TestProposedTask_JsonSerialization()
        {
            // Arrange
            ProposedTask task = new ProposedTask
            {
                task_id = "task_123",
                description = "Test task",
                operations = new List<TaskOperation>
                {
                    new TaskOperation
                    {
                        type = "move",
                        robot_id = "Robot1"
                    }
                },
                required_robots = new List<string> { "Robot1" },
                estimated_complexity = 2,
                reasoning = "Test"
            };

            // Act - Serialize
            string json = JsonUtility.ToJson(task);

            // Assert - Verify JSON contains expected fields
            Assert.True(json.Contains("task_123"), "JSON should contain task_id");
            Assert.True(json.Contains("Test task"), "JSON should contain description");

            // Act - Deserialize
            ProposedTask deserialized = JsonUtility.FromJson<ProposedTask>(json);

            // Assert - Verify deserialized values
            Assert.AreEqual(task.task_id, deserialized.task_id);
            Assert.AreEqual(task.description, deserialized.description);
            Assert.AreEqual(task.estimated_complexity, deserialized.estimated_complexity);
        }

        [Test]
        public void TestAutoRTResponse_JsonSerialization()
        {
            // Arrange
            AutoRTResponse response = new AutoRTResponse
            {
                success = true,
                loop_running = false,
                error = null
            };

            // Act - Serialize
            string json = JsonUtility.ToJson(response);

            // Assert
            Assert.True(json.Contains("\"success\":true"), "JSON should contain success field");

            // Act - Deserialize
            AutoRTResponse deserialized = JsonUtility.FromJson<AutoRTResponse>(json);

            // Assert
            Assert.AreEqual(response.success, deserialized.success);
            Assert.AreEqual(response.loop_running, deserialized.loop_running);
        }

        #endregion

        #region Edge Cases

        [Test]
        public void TestAutoRTCommand_LargeParams()
        {
            // Arrange - Create large params JSON
            List<string> manyRobots = new List<string>();
            for (int i = 0; i < 100; i++)
            {
                manyRobots.Add($"Robot{i}");
            }
            string largeParams = $"{{\"robot_ids\":[{string.Join(",", manyRobots.ConvertAll(r => $"\"{r}\""))}]}}";
            uint requestId = 77777;

            // Act
            byte[] encoded = UnityProtocol.EncodeAutoRTCommand("generate", largeParams, requestId);
            UnityProtocol.DecodeAutoRTCommand(encoded, out uint decodedRequestId, out string decodedCommand, out string decodedParams);

            // Assert
            Assert.AreEqual(requestId, decodedRequestId);
            Assert.True(decodedParams.Length > 100, "Should handle large params");
        }

        [Test]
        public void TestProposedTask_NullLists()
        {
            // Arrange
            ProposedTask task = new ProposedTask
            {
                task_id = "test",
                operations = null,
                required_robots = null
            };

            // Act & Assert - Should not throw
            Assert.AreEqual(0, task.RobotCount, "Should handle null robot list");
            Assert.AreEqual(0, task.OperationCount, "Should handle null operation list");
        }

        [Test]
        public void TestAutoRTResponse_MultipleTasks()
        {
            // Arrange - Create response with many tasks
            AutoRTResponse response = new AutoRTResponse
            {
                success = true,
                loop_running = false
            };

            for (int i = 0; i < 20; i++)
            {
                response.tasks.Add(new ProposedTask
                {
                    task_id = $"task_{i}",
                    description = $"Task {i}",
                    estimated_complexity = i % 5
                });
            }

            // Act - Serialize and deserialize
            string json = JsonUtility.ToJson(response);
            AutoRTResponse deserialized = JsonUtility.FromJson<AutoRTResponse>(json);

            // Assert
            Assert.AreEqual(20, response.tasks.Count);
        }

        #endregion

        #region Request ID Tests

        [Test]
        public void TestRequestId_MaxValue()
        {
            // Arrange
            uint maxRequestId = uint.MaxValue;
            string commandType = "generate";

            // Act
            byte[] encoded = UnityProtocol.EncodeAutoRTCommand(commandType, "{}", maxRequestId);
            UnityProtocol.DecodeAutoRTCommand(encoded, out uint decodedRequestId, out _, out _);

            // Assert
            Assert.AreEqual(maxRequestId, decodedRequestId, "Should handle max uint value");
        }

        [Test]
        public void TestRequestId_Zero()
        {
            // Arrange
            uint zeroRequestId = 0;
            string commandType = "stop_loop";

            // Act
            byte[] encoded = UnityProtocol.EncodeAutoRTCommand(commandType, "{}", zeroRequestId);
            UnityProtocol.DecodeAutoRTCommand(encoded, out uint decodedRequestId, out _, out _);

            // Assert
            Assert.AreEqual(zeroRequestId, decodedRequestId, "Should handle zero request ID");
        }

        #endregion
    }
}
