using EasyButtons;
using UnityEngine;

public class GripperController : MonoBehaviour
{
    public ArticulationBody[] robotGripper;

    [Button]
    private void Open() => Debug.Log("Button Clicked!");

    [Button]
    private void Close() => Debug.Log("Button Clicked!");

    
}
