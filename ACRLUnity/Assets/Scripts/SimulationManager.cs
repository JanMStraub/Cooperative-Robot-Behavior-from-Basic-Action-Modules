using UnityEngine;

public class SimulationManager : MonoBehaviour
{
    private PythonCaller _pythonCaller;

    [Header("Control Parameters")]
    public bool stopRobot = false;
    public static SimulationManager Instance { get; private set; } // Singleton instance

    // Singleton pattern initialization
    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    private void Start()
    {
        _pythonCaller = PythonCaller.Instance;
    }

    private void LateUpdate()
    {
        if (_pythonCaller.IsActive()) { }
    }
}
