using UnityEngine;

public class SimulationManager : MonoBehaviour
{
    private PythonCaller _pythonCaller;
    private bool _sceenshotsSaved = false;

    public bool stopRobot = false;
    public string screenshotExportFolder;
    public static SimulationManager Instance { get; private set; } // Singleton instance

    /// <summary>
    /// Sets the flag indicating whether screenshots have been saved.
    /// </summary>
    /// <param name="setting"> The value to set the screenshotsSaved flag to.
    /// </param>
    public void SetScreenshotsSaved(bool setting)
    {
        _sceenshotsSaved = setting;
    }

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
