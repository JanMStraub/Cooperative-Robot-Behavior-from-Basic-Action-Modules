/*
 * GripperController.cs
 *
 * Author: Fabian Kontor
 * Source: https://github.com/zebleck/AR4/blob/mlagents/Scripts/GripperController.cs
 * Modified by: Jan M. Straub
 *
 * Description:
 * Provides smooth control over AR4 gripper using ArticulationBody components.
 */

using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

#if UNITY_EDITOR
[CustomEditor(typeof(GripperController))]
public class GripperControllerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        var controller = (GripperController)target;

        if (controller.leftGripper == null || controller.rightGripper == null)
        {
            EditorGUILayout.HelpBox(
                "Assign both gripper references to enable manual control.",
                MessageType.Error
            );
            return;
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Gripper Control", EditorStyles.boldLabel);

        if (GUILayout.Button("Open Grippers"))
            controller.OpenGrippers();

        if (GUILayout.Button("Close Grippers"))
            controller.CloseGrippers();

        float lower = controller.leftGripper.xDrive.lowerLimit;
        float upper = controller.leftGripper.xDrive.upperLimit;

        float newPosition = GUILayout.HorizontalSlider(controller.targetPosition, lower, upper);

        if (!Mathf.Approximately(newPosition, controller.targetPosition))
        {
            controller.targetPosition = newPosition;
            EditorUtility.SetDirty(controller);
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Debug Info", EditorStyles.boldLabel);
        EditorGUILayout.LabelField(
            "Left Target",
            controller.leftGripper.xDrive.target.ToString("F2")
        );
        EditorGUILayout.LabelField(
            "Right Target",
            controller.rightGripper.xDrive.target.ToString("F2")
        );
    }
}
#endif

[RequireComponent(typeof(Transform))]
public class GripperController : MonoBehaviour
{
    [Header("Gripper References")]
    public ArticulationBody leftGripper;
    public ArticulationBody rightGripper;

    [Header("Control Parameters")]
    public float maxForce = 100f;
    public float speed = 10f;

    [Range(0f, 1f)]
    public float targetPosition = 0f;

    public float CurrentPosition => leftGripper?.jointPosition[0] ?? 0f;

    private void SetupDrive(ArticulationBody gripper)
    {
        var drive = gripper.xDrive;
        drive.forceLimit = maxForce;
        drive.stiffness = 1000f;
        drive.damping = 100f;
        gripper.xDrive = drive;
    }

    private void ApplyTargetToGrippers(float target)
    {
        ApplyDriveTarget(leftGripper, target);
        ApplyDriveTarget(rightGripper, target);
    }

    private void ApplyDriveTarget(ArticulationBody gripper, float target)
    {
        var drive = gripper.xDrive;
        drive.target = target;
        gripper.xDrive = drive;
    }

    public void SetGripperPosition(float normalizedPosition)
    {
        targetPosition = Mathf.Clamp01(normalizedPosition);
    }

    public void OpenGrippers()
    {
        targetPosition = leftGripper.xDrive.upperLimit;
    }

    public void CloseGrippers()
    {
        targetPosition = leftGripper.xDrive.lowerLimit;
    }

    public void ResetGrippers()
    {
        targetPosition = 0f;
        ResetGripper(leftGripper);
        ResetGripper(rightGripper);
    }

    private void ResetGripper(ArticulationBody gripper)
    {
        ApplyDriveTarget(gripper, 0f);
        gripper.jointPosition = new ArticulationReducedSpace(0f);
        gripper.jointVelocity = new ArticulationReducedSpace(0f);
        gripper.jointForce = new ArticulationReducedSpace(0f);
    }

    private void Awake()
    {
        if (leftGripper == null || rightGripper == null)
        {
            Debug.LogError("Gripper references not assigned!");
            return;
        }

        SetupDrive(leftGripper);
        SetupDrive(rightGripper);
    }

    private void Update()
    {
        float newTarget = Mathf.MoveTowards(
            leftGripper.xDrive.target,
            targetPosition,
            speed * Time.deltaTime
        );

        ApplyTargetToGrippers(newTarget);
    }
}
