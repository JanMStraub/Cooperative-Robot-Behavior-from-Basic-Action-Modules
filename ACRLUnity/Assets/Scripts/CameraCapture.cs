using System.IO;
using UnityEngine;

public class CameraCapture : MonoBehaviour
{
    private int _counter = 0;
    private string _exportFolder;
    private SimulationManager _simulationManagerInstance;
    public int imageWidth;
    public int imageHeight;
    public bool takeScreenshot;
    public Camera mainCamera;

    /// <summary>
    /// Saves the images from the left and right camera.
    /// </summary>
    public void CaptureAndSave()
    {
        CaptureCamera(mainCamera, $"{_exportFolder}/{_counter}.png");
        _counter += 1;
    }

    /// <summary>
    /// Captures the scene from the specified camera and saves it as an image
    /// to the given path.
    /// </summary>
    /// <param name="cam"> The Camera object from which to capture the scene.
    /// </param>
    /// <param name="savePath"> The relative path (within the project) to save
    /// the captured image.</param>
    public void CaptureCamera(Camera cam, string savePath)
    {
        // Create a temporary RenderTexture
        RenderTexture renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        cam.targetTexture = renderTexture;
        cam.Render();

        // Set up a Texture2D to copy the RenderTexture
        RenderTexture.active = renderTexture;
        Texture2D texture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        texture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture.Apply();

        // Save to disk
        string path = Path.Combine(Application.dataPath, savePath);
        byte[] bytes = texture.EncodeToPNG(); // Change to EncodeToJPG() for JPG
        File.WriteAllBytes(path, bytes);

        // Cleanup
        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(renderTexture);
        Destroy(texture);

        Debug.Log($"Saved image to: {path}");

        _simulationManagerInstance.SetScreenshotsSaved(true);
    }

    private void Start()
    {
        _simulationManagerInstance = SimulationManager.Instance;
        _exportFolder = _simulationManagerInstance.screenshotExportFolder;

        CaptureAndSave();
    }

    private void Update()
    {
        // Check if the "E" key was pressed
        if (Input.GetKeyDown(KeyCode.E) || takeScreenshot)
        {
            CaptureAndSave();
        }
    }
}
