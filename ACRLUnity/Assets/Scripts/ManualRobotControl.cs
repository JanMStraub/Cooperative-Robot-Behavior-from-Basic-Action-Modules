using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(RobotController))]
public class ManualRobotController : MonoBehaviour
{
    private int? _selectedJointIndex = null;
    private RobotController _robotController;
    private RobotManager _robotManager;
    private const float AdjustmentStep = 1f;

    /// <summary>
    /// Maps number key presses to joint indices.
    /// </summary>
    private void HandleJointSelection()
    {
        if (Keyboard.current.digit1Key.wasPressedThisFrame)
            _selectedJointIndex = 0;
        else if (Keyboard.current.digit2Key.wasPressedThisFrame)
            _selectedJointIndex = 1;
        else if (Keyboard.current.digit3Key.wasPressedThisFrame)
            _selectedJointIndex = 2;
        else if (Keyboard.current.digit4Key.wasPressedThisFrame)
            _selectedJointIndex = 3;
        else if (Keyboard.current.digit5Key.wasPressedThisFrame)
            _selectedJointIndex = 4;
        else if (Keyboard.current.digit6Key.wasPressedThisFrame)
            _selectedJointIndex = 5;

        if (_selectedJointIndex.HasValue)
        {
            Debug.Log($"Selected Joint {_selectedJointIndex.Value + 1}");
        }
    }

    /// <summary>
    /// Handles input and updates articulation joints accordingly.
    /// </summary>
    private void MoveJoints()
    {
        HandleJointSelection();

        if (_selectedJointIndex.HasValue)
        {
            ArticulationBody currentJoint = _robotController.robotJoints[_selectedJointIndex.Value];

            float adjustment = 0f;

            if (Keyboard.current.leftArrowKey.isPressed)
                adjustment = -AdjustmentStep;
            else if (Keyboard.current.rightArrowKey.isPressed)
                adjustment = AdjustmentStep;

            if (adjustment != 0f)
            {
                ArticulationDrive drive = currentJoint.xDrive;

                float current = drive.target;
                float target = Mathf.Clamp(adjustment, drive.lowerLimit, drive.upperLimit);
                float step = _robotManager.robotAdjustmentSpeed * Time.deltaTime;

                if (!Mathf.Approximately(current, target))
                {
                    float newTarget = Mathf.MoveTowards(current, target, step);
                    drive.target = newTarget;
                    currentJoint.xDrive = drive;
                    Debug.Log($"Joint {_selectedJointIndex.Value + 1} target set to {newTarget}°");
                }
            }
        }
        else if (Keyboard.current.anyKey.wasPressedThisFrame)
        {
            Debug.LogWarning("No joint selected. Press a number key (1-6) to select a joint.");
        }
    }

    private void Start()
    {
        _robotController = GetComponent<RobotController>();
        _robotManager = RobotManager.Instance;
    }

    private void FixedUpdate()
    {
        MoveJoints();
    }
}