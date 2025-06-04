using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class GripperController : MonoBehaviour
{
    [Header("Gripper References")]
    public ArticulationBody leftGripper;
    public ArticulationBody rightGripper;

    [Header("Control Parameters")]
    public float maxForce = 100f;
    public float speed = 10f;
    
    public float targetPosition = 0f;
    public float currentPosition {
        get {
            return leftGripper.jointPosition[0];
        }
    }
    
    private void Awake()
    {
        InitializeGrippers();
    }
    
    private void InitializeGrippers()
    {
        // Validate gripper references
        if (leftGripper == null || rightGripper == null)
        {
            Debug.LogError("Gripper references not assigned in GripperController!");
            return;
        }
        
        // Initialize the left gripper drive
        var leftDrive = leftGripper.xDrive;
        leftDrive.forceLimit = maxForce;
        leftDrive.stiffness = 10000;
        leftDrive.damping = 100;
        leftGripper.xDrive = leftDrive;
        
        // Initialize the right gripper drive
        var rightDrive = rightGripper.xDrive;
        rightDrive.forceLimit = maxForce;
        rightDrive.stiffness = 10000;
        rightDrive.damping = 100;
        rightGripper.xDrive = rightDrive;
    }
    
    private void Update()
    {
        var leftDrive = leftGripper.xDrive;
        var rightDrive = rightGripper.xDrive;

        // Gradually move towards target position
        float smoothPosition = Mathf.MoveTowards(leftDrive.target, targetPosition, speed * Time.deltaTime);

        leftDrive.target = smoothPosition;
        rightDrive.target = smoothPosition;

        leftGripper.xDrive = leftDrive;
        rightGripper.xDrive = rightDrive;

        
    }
    
    public void SetGripperPosition(float normalizedPosition)
    {
        targetPosition = Mathf.Clamp01(normalizedPosition);
    }
    
    public void OpenGrippers()
    {
        var drive = leftGripper.xDrive;
        targetPosition = drive.upperLimit;
    }

    public void ResetGrippers()
    {
        targetPosition = 0f;

        var leftDrive = leftGripper.xDrive;
        leftDrive.target = 0f;
        leftGripper.xDrive = leftDrive;
        leftGripper.jointPosition = new ArticulationReducedSpace(0f);
        leftGripper.jointForce = new ArticulationReducedSpace(0f);
        leftGripper.jointVelocity = new ArticulationReducedSpace(0f);

        var rightDrive = rightGripper.xDrive;
        rightDrive.target = 0f;
        rightGripper.xDrive = rightDrive;
        rightGripper.jointPosition = new ArticulationReducedSpace(0f);
        rightGripper.jointForce = new ArticulationReducedSpace(0f);
        rightGripper.jointVelocity = new ArticulationReducedSpace(0f);
    }
    
    public void CloseGrippers()
    {
        var drive = leftGripper.xDrive;
        targetPosition = drive.lowerLimit; 
    }
    
#if UNITY_EDITOR
    [CustomEditor(typeof(GripperController))]
    public class GripperControllerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            
            GripperController controller = (GripperController)target;
            
            // Make sure the gripper references are initialized
            if (controller.leftGripper == null || controller.rightGripper == null)
            {
                EditorGUILayout.HelpBox("One or both gripper ArticulationBody components not assigned!", MessageType.Error);
                return;
            }
            
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Gripper Control", EditorStyles.boldLabel);
            
            if (GUILayout.Button("Open Grippers"))
            {
                controller.OpenGrippers();
            }
            
            if (GUILayout.Button("Close Grippers"))
            {
                controller.CloseGrippers();
            }

            float newPosition = GUILayout.HorizontalSlider(
                controller.targetPosition, 
                controller.leftGripper.xDrive.lowerLimit, 
                controller.leftGripper.xDrive.upperLimit
            );

            if (newPosition != controller.targetPosition)
            {
                controller.targetPosition = newPosition;
                EditorUtility.SetDirty(target);
            }
            
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Current Settings", EditorStyles.boldLabel);
            
            var leftDrive = controller.leftGripper.xDrive;
            var rightDrive = controller.rightGripper.xDrive;
            
            EditorGUILayout.LabelField("Left Gripper Target Angle", leftDrive.target.ToString("F2"));
            EditorGUILayout.LabelField("Right Gripper Target Angle", rightDrive.target.ToString("F2"));
        }
    }
#endif
}