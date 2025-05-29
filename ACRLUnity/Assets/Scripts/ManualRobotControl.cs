using UnityEngine;
using UnityEngine.InputSystem;

public class ManualRobotController : MonoBehaviour
{
    private int? _selectedJointIndex = null; // Currently selected joint index
    private RobotManager _robotManagerInstance;

    /// <summary>
    /// Handles the movement of robot joints based on keyboard input.
    /// Maps number keys (1-6) to select joints, and left/right arrow keys to
    /// adjust the selected joint's target.
    /// </summary>
    private void MoveJoints()
    {
        // Initialize current joint
        ArticulationBody currentJoint = null;

        // Map number keys to joints
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

        // Select the joint based on the key pressed

        if (_selectedJointIndex.HasValue)
        {
            currentJoint = this.GetComponent<RobotController>().robotJoints[
                _selectedJointIndex.Value
            ];
        }

        if (currentJoint != null)
        {
            var drive = currentJoint.xDrive;
            // Adjust the drive target for the selected joint
            float adjustment = 0f;

            if (Keyboard.current.leftArrowKey.isPressed)
            {
                adjustment = -1f; // Decrease target
            }
            else if (Keyboard.current.rightArrowKey.isPressed)
            {
                adjustment = 1f; // Increase target
            }

            if (adjustment != 0f)
            {
                // Compute the new target and clamp within limits
                float newTarget = Mathf.Clamp(
                    drive.target
                        + adjustment * this.GetComponent<RobotController>().GetMaxStepSpeed(),
                    drive.lowerLimit,
                    drive.upperLimit
                );

                if (newTarget != drive.target) // Only update if within limits
                {
                    drive.target = newTarget;
                    currentJoint.xDrive = drive;
                }
                else
                {
                    Debug.LogWarning("The limit of this joint is reached.");
                }
            }
        }
        else if (Keyboard.current.anyKey.wasPressedThisFrame)
        {
            Debug.LogWarning("No joint selected. Press a number key to select a joint.");
        }
    }

    private void Start()
    {
        _robotManagerInstance = RobotManager.Instance;
    }

    void FixedUpdate()
    {
        MoveJoints();
    }
}
