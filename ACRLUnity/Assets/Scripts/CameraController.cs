using System.Collections;
using System.IO;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

#if UNITY_EDITOR
[CustomEditor(typeof(CameraController))]
public class CameraControllerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        var controller = (CameraController)target;

        if (GUILayout.Button("Take screenshot"))
            controller.CaptureAndSave();
    }
}
#endif

public class CameraController : MonoBehaviour
{
    private int _counter = 0;
    private string _robotArmName;
    private string _rootName;
    private Camera _mainCamera;

    public int imageWidth;
    public int imageHeight;

    /// <summary>
    /// Saves the images from the left and right camera.
    /// </summary>
    public void CaptureAndSave()
    {
        StartCoroutine(
            CaptureCamera(_mainCamera, $"Data/Screenshots/{_rootName}/{_robotArmName}/{_counter}.jpg")
        );
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
    public IEnumerator CaptureCamera(Camera cam, string savePath)
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
        string pathDir = Path.GetDirectoryName(path);
        if (!Directory.Exists(pathDir))
        {
            Directory.CreateDirectory(pathDir);
        }
        byte[] bytes = texture.EncodeToJPG(); // Change to EncodeToPNG() for PNG
        File.WriteAllBytes(path, bytes);

        // Cleanup
        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(renderTexture);
        Destroy(texture);

        Debug.Log($"Saved image to: {path}");

        yield return null;
    }

    private string FindArmRoot(Transform current)
    {
        while (current.parent != null)
        {
            current = current.parent;
            if (current.name is "AR4Left" or "AR4Right")
            {
                return current.name;
            }
        }
        return null;
    }

    private void Start()
    {
        _mainCamera = GetComponent<Camera>();
        _robotArmName = FindArmRoot(_mainCamera.transform);
        _rootName = _mainCamera.transform.root.name;
    }
}
