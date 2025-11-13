using System.Collections.Generic;
using System.Linq;
using PythonCommunication;
using Robotics;
using UnityEngine;
using Utilities;

/// <summary>
/// Coordinates robot navigation to detected objects using stereo depth estimation
/// </summary>
public class ObjectNavigationController : MonoBehaviour
{
    [Header("Robot Configuration")]
    [SerializeField]
    private string leftRobotId = "LeftRobot";

    [SerializeField]
    private string rightRobotId = "RightRobot";

    [Header("Detection Settings")]
    [SerializeField]
    private string targetObjectColor = "red"; // Filter by object color (e.g., "red", "blue", "green")

    // Helper variable
    private const string _logPrefix = "[OBJECT_NAVIGATION_CONTROLLER]";

    /// <summary>
    /// Initialize and subscribe to depth results
    /// </summary>
    void Start()
    {
        if (UnifiedPythonReceiver.Instance != null)
        {
            UnifiedPythonReceiver.Instance.OnDepthResultReceived += OnDepthResultReceived;
            Debug.Log(
                $"{_logPrefix} subscribed: Left={leftRobotId}, Right={rightRobotId}, targeting {targetObjectColor} cubes"
            );
        }
        else
        {
            Debug.LogError(
                $"{_logPrefix} UnifiedPythonReceiver.Instance is null! Ensure UnifiedPythonReceiver exists in scene."
            );
        }
    }

    /// <summary>
    /// Handle received depth results and set robot targets based on X position
    /// Left robot gets leftmost target, right robot gets rightmost target
    /// </summary>
    void OnDepthResultReceived(DepthResult result)
    {
        if (!result.success || result.detections == null || result.detections.Length == 0)
        {
            Debug.LogWarning($"{_logPrefix} No objects detected in stereo view");
            return;
        }

        // Filter detections by target color
        var matchingDetections = result
            .detections.Where(d => d.color == targetObjectColor && d.world_position != null)
            .ToList();

        if (matchingDetections.Count == 0)
        {
            Debug.LogWarning(
                $"{_logPrefix} No {targetObjectColor} cubes with 3D positions found. Available: {string.Join(", ", result.detections.Select(d => d.color))}"
            );
            return;
        }

        // Sort detections by X position (left to right)
        var sortedDetections = matchingDetections.OrderBy(d => d.world_position.x).ToList();

        // Assign leftmost target to left robot
        ObjectDetection leftTarget = sortedDetections.First();
        Vector3 leftTargetPos = new Vector3(
            leftTarget.world_position.x,
            leftTarget.world_position.y,
            leftTarget.world_position.z
        );

        List<GameObject> leftTargetObjects = ObjectFinder.Instance.FindGraspableObjects(
            leftTargetPos
        );

        // Assign rightmost target to right robot
        ObjectDetection rightTarget = sortedDetections.Last();
        Vector3 rightTargetPos = new Vector3(
            rightTarget.world_position.x,
            rightTarget.world_position.y,
            rightTarget.world_position.z
        );

        List<GameObject> rightTargetObjects = ObjectFinder.Instance.FindGraspableObjects(
            rightTargetPos
        );

        // Set robot targets via RobotManager
        if (RobotManager.Instance != null)
        {
            // Check if robots are registered
            if (!RobotManager.Instance.RobotInstances.ContainsKey(leftRobotId))
            {
                Debug.LogError(
                    $"{_logPrefix} Robot '{leftRobotId}' is not registered with RobotManager! Available robots: {string.Join(", ", RobotManager.Instance.RobotInstances.Keys)}"
                );
            }
            if (!RobotManager.Instance.RobotInstances.ContainsKey(rightRobotId))
            {
                Debug.LogError(
                    $"{_logPrefix} Robot '{rightRobotId}' is not registered with RobotManager! Available robots: {string.Join(", ", RobotManager.Instance.RobotInstances.Keys)}"
                );
            }

            RobotManager.Instance.SetRobotTarget(leftRobotId, leftTargetObjects[0]);
            RobotManager.Instance.SetRobotTarget(rightRobotId, rightTargetObjects[0]);

            Debug.Log(
                $"{_logPrefix} Targets assigned:\n"
                    + $"  {leftRobotId} -> {targetObjectColor} at {leftTargetPos} (X={leftTarget.world_position.x:F3}, depth={leftTarget.depth_m:F3}m)\n"
                    + $"  {rightRobotId} -> {targetObjectColor} at {rightTargetPos} (X={rightTarget.world_position.x:F3}, depth={rightTarget.depth_m:F3}m)"
            );

            // Verify targets were set by checking robot instances
            if (RobotManager.Instance.RobotInstances.TryGetValue(leftRobotId, out var leftRobot))
            {
                Debug.Log(
                    $"{_logPrefix} {leftRobotId} controller target set: {leftRobot.controller.GetCurrentTarget().HasValue}"
                );
            }
            if (RobotManager.Instance.RobotInstances.TryGetValue(rightRobotId, out var rightRobot))
            {
                Debug.Log(
                    $"{_logPrefix} {rightRobotId} controller target set: {rightRobot.controller.GetCurrentTarget().HasValue}"
                );
            }
        }
        else
        {
            Debug.LogError(
                $"{_logPrefix} RobotManager.Instance is null! Cannot set robot targets."
            );
        }
    }

    /// <summary>
    /// Unsubscribe from events on destroy
    /// </summary>
    void OnDestroy()
    {
        if (UnifiedPythonReceiver.Instance != null)
        {
            UnifiedPythonReceiver.Instance.OnDepthResultReceived -= OnDepthResultReceived;
        }
    }
}
