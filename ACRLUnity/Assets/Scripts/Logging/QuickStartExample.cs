using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Logging
{
    /// <summary>
    /// Quick start example demonstrating  logging system
    /// Much simpler than the previous 400-line example
    /// </summary>
    public class QuickStartExample : MonoBehaviour
    {
        [Header("Example Setup")]
        public RobotController leftRobot;
        public RobotController rightRobot;
        public GameObject targetObject;
        public Transform destination;

        [Header("Settings")]
        public bool autoRun = false;
        public float delayBeforeStart = 2f;

        private RobotLogger _logger;

        private void Start()
        {
            _logger = RobotLogger.Instance;

            if (_logger == null)
            {
                Debug.LogError("RobotLogger not found. Add it to the scene first.");
                return;
            }

            if (autoRun)
            {
                StartCoroutine(RunExampleWithDelay());
            }
        }

        private IEnumerator RunExampleWithDelay()
        {
            yield return new WaitForSeconds(delayBeforeStart);
            RunPickAndPlaceExample();
        }

        [ContextMenu("Run Pick and Place Example")]
        public void RunPickAndPlaceExample()
        {
            StartCoroutine(PickAndPlaceSequence());
        }

        private IEnumerator PickAndPlaceSequence()
        {
            Debug.Log("=== Starting  Pick and Place Example ===");

            // 1. Start a coordination task
            string taskId = _logger.LogCoordination(
                "collaborative_pick_and_place",
                new[] { leftRobot.robotId, rightRobot.robotId },
                $"Pick {targetObject.name} and place at {destination.name}",
                new[] { targetObject.name }
            );

            // 2. Move robots to target
            string moveLeft = _logger.StartAction(
                "approach_target",
                ActionType.Movement,
                new[] { leftRobot.robotId },
                leftRobot.endEffectorBase.position,
                targetObject.transform.position,
                new[] { targetObject.name }
            );

            leftRobot.SetTarget(targetObject);
            yield return new WaitUntil(() => leftRobot.GetDistanceToTarget() < 0.1f);

            _logger.CompleteAction(moveLeft, true, 0.9f);

            // 3. Pick object
            string pickAction = _logger.StartAction(
                "pick_object",
                ActionType.Manipulation,
                new[] { leftRobot.robotId },
                objectIds: new[] { targetObject.name },
                description: "Grasping target object"
            );

            // Simulate gripper closing
            var gripper = leftRobot.GetComponentInChildren<GripperController>();
            if (gripper != null)
            {
                gripper.CloseGrippers();
                yield return new WaitForSeconds(1f);
            }

            targetObject.transform.SetParent(leftRobot.endEffectorBase);
            _logger.CompleteAction(pickAction, true, 0.85f);

            // 4. Move to destination
            string moveToDestination = _logger.StartAction(
                "move_to_destination",
                ActionType.Movement,
                new[] { leftRobot.robotId },
                leftRobot.endEffectorBase.position,
                destination.position,
                new[] { targetObject.name }
            );

            leftRobot.SetTarget(destination.gameObject);
            yield return new WaitUntil(() => leftRobot.GetDistanceToTarget() < 0.1f);

            _logger.CompleteAction(moveToDestination, true, 0.9f);

            // 5. Place object
            string placeAction = _logger.StartAction(
                "place_object",
                ActionType.Manipulation,
                new[] { leftRobot.robotId },
                objectIds: new[] { targetObject.name },
                description: "Releasing object at destination"
            );

            targetObject.transform.SetParent(null);
            targetObject.transform.position = destination.position;

            if (gripper != null)
            {
                gripper.OpenGrippers();
                yield return new WaitForSeconds(1f);
            }

            _logger.CompleteAction(placeAction, true, 0.9f);

            // 6. Complete coordination task
            var metrics = new Dictionary<string, float>
            {
                ["success"] = 1f,
                ["total_time"] = Time.time,
                ["efficiency"] = 0.85f,
            };

            _logger.CompleteAction(taskId, true, 0.9f, null, metrics);

            // 7. Capture final environment
            _logger.CaptureEnvironment("task_complete");

            Debug.Log("=== Example Complete! ===");
            Debug.Log($"Logs saved to: {Application.persistentDataPath}/RobotLogs");
        }

        [ContextMenu("Export Logs")]
        public void ExportLogs()
        {
            LLMExporter.QuickExport("jsonl");
            Debug.Log("Logs exported!");
        }

        [ContextMenu("Export Conversational Format")]
        public void ExportConversational()
        {
            LLMExporter.QuickExport("conversational");
            Debug.Log("Conversational format exported!");
        }

        [ContextMenu("Show Statistics")]
        public void ShowStatistics()
        {
            string logDir = System.IO.Path.Combine(Application.persistentDataPath, "RobotLogs");
            var files = System.IO.Directory.GetFiles(logDir, "robot_actions_*.jsonl");

            if (files.Length > 0)
            {
                var stats = LLMExporter.GenerateStatistics(files[0]);
                Debug.Log("Statistics generated - check console for details");
            }
        }
    }
}
