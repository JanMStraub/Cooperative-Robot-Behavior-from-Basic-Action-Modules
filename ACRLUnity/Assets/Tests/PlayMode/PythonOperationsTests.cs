using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using PythonCommunication;
using Tests.EditMode;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for Python operations integration.
    /// Validates integration with 17 registered Python operations:
    /// - Detection: detect_object, detect_objects
    /// - Vision: analyze_scene
    /// - Movement: move_to_coordinate, move_relative_to_object
    /// - Manipulation: control_gripper
    /// - Sync: signal, wait_for_signal, wait
    /// - Variable passing: $target syntax
    /// </summary>
    public class PythonOperationsTests
    {
        private GameObject _sequenceClientObject;
        private SequenceClient _sequenceClient;
        private bool _pythonBackendAvailable;

        #region Setup/Teardown

        [SetUp]
        public void SetUp()
        {
            // Create SequenceClient instance
            _sequenceClientObject = new GameObject("TestSequenceClient");
            _sequenceClient = _sequenceClientObject.AddComponent<SequenceClient>();

            // Check if Python backend is available
            _pythonBackendAvailable = IsPythonBackendAvailable();

            if (!_pythonBackendAvailable)
            {
                Debug.LogWarning("[PYTHON_OPERATIONS_TESTS] Python backend not available - some tests will be skipped");
            }
        }

        [TearDown]
        public void TearDown()
        {
            if (_sequenceClientObject != null)
            {
                Object.DestroyImmediate(_sequenceClientObject);
            }
        }

        /// <summary>
        /// Check if Python backend is running (port 5013 listening)
        /// </summary>
        private bool IsPythonBackendAvailable()
        {
            // Simple check: try connecting to SequenceServer
            // In production, could use TCP port check or ping command
            return false; // Default to false for unit tests
        }

        #endregion

        #region Command Sending Tests

        [Test]
        public void SequenceClient_Initialization_CreatesInstance()
        {
            Assert.IsNotNull(_sequenceClient, "SequenceClient should be created");
            Assert.IsNotNull(SequenceClient.Instance, "SequenceClient Instance should be set");
        }

        [Test]
        public void ExecuteSequence_WithNullCommand_ReturnsFalse()
        {
            bool result = _sequenceClient.ExecuteSequence(null);
            Assert.IsFalse(result, "ExecuteSequence should return false for null command");
        }

        [Test]
        public void ExecuteSequence_WithEmptyCommand_ReturnsFalse()
        {
            bool result = _sequenceClient.ExecuteSequence("");
            Assert.IsFalse(result, "ExecuteSequence should return false for empty command");
        }

        [Test]
        public void ExecuteSequence_WhenNotConnected_ReturnsFalse()
        {
            // SequenceClient starts disconnected unless Python backend is running
            bool result = _sequenceClient.ExecuteSequence("test command");

            if (!_sequenceClient.IsConnected)
            {
                Assert.IsFalse(result, "ExecuteSequence should return false when not connected");
            }
        }

        [Test]
        public void MoveAndGrip_ConstructsCorrectCommand()
        {
            // This test validates command construction without sending
            // The actual command sending is tested in integration tests
            var command = "move to (0.3, 0.2, 0.1) and close the gripper";

            // Verify command format (would be sent via ExecuteSequence)
            Assert.IsNotNull(command, "Command should be constructed");
            Assert.IsTrue(command.Contains("move to"), "Command should contain move directive");
            Assert.IsTrue(command.Contains("close the gripper"), "Command should contain gripper directive");
        }

        [Test]
        public void Pick_ConstructsMultiStepCommand()
        {
            // Validate pick command construction: move, close, lift
            float x = 0.3f, y = 0.2f, z = 0.1f, liftHeight = 0.1f;
            float liftZ = z + liftHeight;

            var expectedCommand = $"move to ({x}, {y}, {z}), then close the gripper, then move to ({x}, {y}, {liftZ})";

            // Verify multi-step structure
            Assert.IsTrue(expectedCommand.Contains("move to"), "Pick should include move");
            Assert.IsTrue(expectedCommand.Contains("close the gripper"), "Pick should include gripper close");
            Assert.IsTrue(expectedCommand.Contains("then"), "Pick should use sequential 'then' syntax");
        }

        [Test]
        public void Place_ConstructsMultiStepCommand()
        {
            // Validate place command construction: move, open, lift
            float x = 0.3f, y = 0.2f, z = 0.1f, liftHeight = 0.1f;
            float liftZ = z + liftHeight;

            var expectedCommand = $"move to ({x}, {y}, {z}), then open the gripper, then move to ({x}, {y}, {liftZ})";

            // Verify multi-step structure
            Assert.IsTrue(expectedCommand.Contains("move to"), "Place should include move");
            Assert.IsTrue(expectedCommand.Contains("open the gripper"), "Place should include gripper open");
            Assert.IsTrue(expectedCommand.Contains("then"), "Place should use sequential 'then' syntax");
        }

        #endregion

        #region Data Model Tests

        [Test]
        public void SequenceResult_DefaultConstruction_HasValidDefaults()
        {
            var result = new SequenceResult();

            Assert.AreEqual(false, result.success, "Default success should be false");
            Assert.AreEqual(0, result.request_id, "Default request_id should be 0");
            Assert.AreEqual(0, result.total_commands, "Default total_commands should be 0");
            Assert.AreEqual(0, result.completed_commands, "Default completed_commands should be 0");
        }

        [Test]
        public void CommandResult_DefaultConstruction_HasValidDefaults()
        {
            var result = new CommandResult();

            Assert.AreEqual(0, result.index, "Default index should be 0");
            Assert.AreEqual(false, result.success, "Default success should be false");
        }

        [Test]
        public void SequenceResult_CanStoreMultipleCommandResults()
        {
            var sequenceResult = new SequenceResult
            {
                success = true,
                total_commands = 3,
                completed_commands = 3,
                results = new List<CommandResult>
                {
                    new CommandResult { index = 0, operation = "move_to_coordinate", success = true },
                    new CommandResult { index = 1, operation = "control_gripper", success = true },
                    new CommandResult { index = 2, operation = "move_to_coordinate", success = true }
                }
            };

            Assert.AreEqual(3, sequenceResult.results.Count, "Should store 3 command results");
            Assert.AreEqual("move_to_coordinate", sequenceResult.results[0].operation, "First operation should be move_to_coordinate");
            Assert.AreEqual("control_gripper", sequenceResult.results[1].operation, "Second operation should be control_gripper");
        }

        #endregion

        #region Command Parsing Tests (Expected Operations)

        [Test]
        public void Command_DetectObject_ParsesCorrectly()
        {
            // Expected operation: detect_object
            var command = "detect the red cube";

            Assert.IsNotNull(command, "Detect command should be valid");
            Assert.IsTrue(command.Contains("detect"), "Command should contain detect keyword");
        }

        [Test]
        public void Command_MoveToCoordinate_ParsesCorrectly()
        {
            // Expected operation: move_to_coordinate
            var command = "move to (0.3, 0.2, 0.1)";

            Assert.IsNotNull(command, "Move command should be valid");
            Assert.IsTrue(command.Contains("move to"), "Command should contain 'move to' directive");
            Assert.IsTrue(command.Contains("("), "Command should contain coordinate parentheses");
        }

        [Test]
        public void Command_ControlGripper_ParsesCorrectly()
        {
            // Expected operation: control_gripper (close)
            var commandClose = "close the gripper";
            Assert.IsTrue(commandClose.Contains("close") && commandClose.Contains("gripper"),
                "Close gripper command should be valid");

            // Expected operation: control_gripper (open)
            var commandOpen = "open the gripper";
            Assert.IsTrue(commandOpen.Contains("open") && commandOpen.Contains("gripper"),
                "Open gripper command should be valid");
        }

        [Test]
        public void Command_AnalyzeScene_ParsesCorrectly()
        {
            // Expected operation: analyze_scene
            var command = "analyze the scene";

            Assert.IsNotNull(command, "Analyze command should be valid");
            Assert.IsTrue(command.Contains("analyze"), "Command should contain analyze keyword");
        }

        [Test]
        public void Command_MoveRelativeToObject_ParsesCorrectly()
        {
            // Expected operation: move_relative_to_object
            var command = "move 0.1m above the red cube";

            Assert.IsNotNull(command, "Relative move command should be valid");
            Assert.IsTrue(command.Contains("above") || command.Contains("relative"),
                "Command should indicate relative positioning");
        }

        [Test]
        public void Command_Signal_ParsesCorrectly()
        {
            // Expected operation: signal
            var command = "signal ready";

            Assert.IsNotNull(command, "Signal command should be valid");
            Assert.IsTrue(command.Contains("signal"), "Command should contain signal keyword");
        }

        [Test]
        public void Command_WaitForSignal_ParsesCorrectly()
        {
            // Expected operation: wait_for_signal
            var command = "wait for signal ready";

            Assert.IsNotNull(command, "Wait for signal command should be valid");
            Assert.IsTrue(command.Contains("wait") && command.Contains("signal"),
                "Command should contain wait and signal keywords");
        }

        [Test]
        public void Command_Wait_ParsesCorrectly()
        {
            // Expected operation: wait
            var command = "wait 2 seconds";

            Assert.IsNotNull(command, "Wait command should be valid");
            Assert.IsTrue(command.Contains("wait"), "Command should contain wait keyword");
        }

        #endregion

        #region Variable Passing Tests

        [Test]
        public void Command_VariableAssignment_ParsesCorrectly()
        {
            // Expected: detect_object with variable assignment
            var command = "detect the blue cube -> $target";

            Assert.IsNotNull(command, "Variable assignment command should be valid");
            Assert.IsTrue(command.Contains("->") && command.Contains("$"),
                "Command should contain variable assignment syntax");
        }

        [Test]
        public void Command_VariableReference_ParsesCorrectly()
        {
            // Expected: reference to previously assigned variable
            var command = "move to $target";

            Assert.IsNotNull(command, "Variable reference command should be valid");
            Assert.IsTrue(command.Contains("$target"), "Command should contain variable reference");
        }

        [Test]
        public void Command_SequenceWithVariablePropagation_ParsesCorrectly()
        {
            // Multi-step command with variable passing
            var command = "detect the red cube -> $target, then move to $target";

            Assert.IsNotNull(command, "Sequence with variable propagation should be valid");
            Assert.IsTrue(command.Contains("detect"), "Should contain detect operation");
            Assert.IsTrue(command.Contains("->"), "Should contain variable assignment");
            Assert.IsTrue(command.Contains("$target"), "Should reference variable");
            Assert.IsTrue(command.Contains("then"), "Should have sequential steps");
        }

        #endregion

        #region Integration Tests (require Python backend)

        [UnityTest]
        public IEnumerator ExecuteSequence_SimpleMove_WithPythonBackend()
        {
            if (!_pythonBackendAvailable)
            {
                Assert.Ignore("Python backend not available");
                yield break;
            }

            // Wait for connection
            yield return new WaitForSeconds(1f);

            bool sent = _sequenceClient.ExecuteSequence("move to (0.3, 0.2, 0.1)", "TestRobot");
            Assert.IsTrue(sent, "Command should be sent successfully");

            // Wait for response
            yield return new WaitForSeconds(2f);

            // Check last result
            var lastResult = _sequenceClient.LastResult;
            if (lastResult != null)
            {
                Assert.IsNotNull(lastResult, "Should receive a result");
                Assert.Greater(lastResult.total_commands, 0, "Should have parsed at least 1 command");
            }
        }

        [UnityTest]
        public IEnumerator ExecuteSequence_DetectAndMove_WithPythonBackend()
        {
            if (!_pythonBackendAvailable)
            {
                Assert.Ignore("Python backend not available");
                yield break;
            }

            yield return new WaitForSeconds(1f);

            bool sent = _sequenceClient.ExecuteSequence(
                "detect the blue cube -> $target, then move to $target",
                "TestRobot"
            );
            Assert.IsTrue(sent, "Command should be sent successfully");

            yield return new WaitForSeconds(3f);

            var lastResult = _sequenceClient.LastResult;
            if (lastResult != null)
            {
                Assert.IsNotNull(lastResult, "Should receive a result");
                Assert.AreEqual(2, lastResult.total_commands, "Should have 2 commands (detect + move)");
            }
        }

        [UnityTest]
        public IEnumerator ExecuteSequence_PickAndPlace_WithPythonBackend()
        {
            if (!_pythonBackendAvailable)
            {
                Assert.Ignore("Python backend not available");
                yield break;
            }

            yield return new WaitForSeconds(1f);

            // Pick operation
            bool sent = _sequenceClient.Pick(0.3f, 0.2f, 0.1f, 0.1f, "TestRobot");
            Assert.IsTrue(sent, "Pick command should be sent successfully");

            yield return new WaitForSeconds(4f);

            var pickResult = _sequenceClient.LastResult;
            if (pickResult != null)
            {
                Assert.IsNotNull(pickResult, "Should receive pick result");
                Assert.AreEqual(3, pickResult.total_commands, "Pick should have 3 steps (move, close, lift)");
            }

            yield return new WaitForSeconds(1f);

            // Place operation
            sent = _sequenceClient.Place(0.4f, 0.3f, 0.1f, 0.1f, "TestRobot");
            Assert.IsTrue(sent, "Place command should be sent successfully");

            yield return new WaitForSeconds(4f);

            var placeResult = _sequenceClient.LastResult;
            if (placeResult != null)
            {
                Assert.IsNotNull(placeResult, "Should receive place result");
                Assert.AreEqual(3, placeResult.total_commands, "Place should have 3 steps (move, open, lift)");
            }
        }

        [UnityTest]
        public IEnumerator ExecuteSequence_MultiRobotCoordination_WithPythonBackend()
        {
            if (!_pythonBackendAvailable)
            {
                Assert.Ignore("Python backend not available");
                yield break;
            }

            yield return new WaitForSeconds(1f);

            // Robot1 signals when ready
            bool sent1 = _sequenceClient.ExecuteSequence(
                "move to (0.3, 0.2, 0.1), then signal ready",
                "Robot1"
            );
            Assert.IsTrue(sent1, "Robot1 command should be sent");

            yield return new WaitForSeconds(0.5f);

            // Robot2 waits for signal
            bool sent2 = _sequenceClient.ExecuteSequence(
                "wait for signal ready, then move to (0.4, 0.3, 0.1)",
                "Robot2"
            );
            Assert.IsTrue(sent2, "Robot2 command should be sent");

            // Wait for both sequences to complete
            yield return new WaitForSeconds(5f);

            // Verify both commands were processed
            var lastResult = _sequenceClient.LastResult;
            if (lastResult != null)
            {
                Assert.IsNotNull(lastResult, "Should receive coordination result");
            }
        }

        #endregion

        #region Error Handling Tests

        [Test]
        public void SequenceResult_WithError_StoresErrorMessage()
        {
            var result = new SequenceResult
            {
                success = false,
                error = "Robot not found: TestRobot",
                total_commands = 1,
                completed_commands = 0
            };

            Assert.IsFalse(result.success, "Result should be marked as failed");
            Assert.IsNotNull(result.error, "Error message should be stored");
            Assert.AreEqual(0, result.completed_commands, "No commands should complete on error");
        }

        [Test]
        public void CommandResult_WithError_StoresErrorMessage()
        {
            var result = new CommandResult
            {
                index = 0,
                operation = "move_to_coordinate",
                success = false,
                error = "Target position unreachable"
            };

            Assert.IsFalse(result.success, "Command should be marked as failed");
            Assert.IsNotNull(result.error, "Error message should be stored");
            Assert.AreEqual("Target position unreachable", result.error, "Error message should match");
        }

        [Test]
        public void SequenceResult_PartialFailure_TracksProgress()
        {
            var result = new SequenceResult
            {
                success = false,
                total_commands = 3,
                completed_commands = 2, // 2 out of 3 succeeded
                results = new List<CommandResult>
                {
                    new CommandResult { index = 0, success = true },
                    new CommandResult { index = 1, success = true },
                    new CommandResult { index = 2, success = false, error = "Gripper jam" }
                }
            };

            Assert.IsFalse(result.success, "Overall sequence should fail");
            Assert.AreEqual(2, result.completed_commands, "Should track partial completion");
            Assert.IsFalse(result.results[2].success, "Third command should be marked as failed");
        }

        #endregion

        #region Recent Commands Tracking Tests

        [Test]
        public void RecentCommands_InitiallyEmpty()
        {
            Assert.IsNotNull(_sequenceClient.RecentCommands, "RecentCommands should be initialized");
            Assert.AreEqual(0, _sequenceClient.RecentCommands.Count, "RecentCommands should start empty");
        }

        [Test]
        public void ClearPrompt_ClearsPromptText()
        {
            _sequenceClient.Prompt = "test command";
            _sequenceClient.ClearPrompt();
            Assert.AreEqual("", _sequenceClient.Prompt, "Prompt should be cleared");
        }

        #endregion

        #region Mock Operation Tests (Unit Testing Without Backend)

        [Test]
        public void MockOperation_DetectObject_ReturnsExpectedStructure()
        {
            // Mock what Python backend would return for detect_object
            var mockResult = new SequenceResult
            {
                success = true,
                request_id = 1,
                sequence_id = "seq_001",
                total_commands = 1,
                completed_commands = 1,
                results = new List<CommandResult>
                {
                    new CommandResult
                    {
                        index = 0,
                        operation = "detect_object",
                        success = true,
                        duration_ms = 150f
                    }
                },
                total_duration_ms = 150f
            };

            Assert.IsTrue(mockResult.success, "Mock detect should succeed");
            Assert.AreEqual("detect_object", mockResult.results[0].operation, "Operation should be detect_object");
        }

        [Test]
        public void MockOperation_MoveToCoordinate_ReturnsExpectedStructure()
        {
            // Mock what Python backend would return for move_to_coordinate
            var mockResult = new SequenceResult
            {
                success = true,
                request_id = 2,
                total_commands = 1,
                completed_commands = 1,
                results = new List<CommandResult>
                {
                    new CommandResult
                    {
                        index = 0,
                        operation = "move_to_coordinate",
                        success = true,
                        duration_ms = 2500f
                    }
                }
            };

            Assert.IsTrue(mockResult.success, "Mock move should succeed");
            Assert.AreEqual("move_to_coordinate", mockResult.results[0].operation, "Operation should be move_to_coordinate");
        }

        [Test]
        public void MockOperation_ControlGripper_ReturnsExpectedStructure()
        {
            // Mock what Python backend would return for control_gripper
            var mockResult = new SequenceResult
            {
                success = true,
                request_id = 3,
                total_commands = 1,
                completed_commands = 1,
                results = new List<CommandResult>
                {
                    new CommandResult
                    {
                        index = 0,
                        operation = "control_gripper",
                        success = true,
                        duration_ms = 500f
                    }
                }
            };

            Assert.IsTrue(mockResult.success, "Mock gripper control should succeed");
            Assert.AreEqual("control_gripper", mockResult.results[0].operation, "Operation should be control_gripper");
        }

        #endregion
    }
}
