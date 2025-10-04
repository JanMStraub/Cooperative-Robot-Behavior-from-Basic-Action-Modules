using System;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using Logging;
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

        if (GUILayout.Button("Take Screenshot"))
            controller.CaptureAndSave();
    }
}
#endif

/// <summary>
/// Controls camera screenshot capture for robot cameras.
/// Captures images from the robot's camera and saves them to disk with proper organization.
/// </summary>
public class CameraController : MonoBehaviour
{
    [Header("Image Settings")]
    [SerializeField]
    [Tooltip("Width of captured images in pixels")]
    private int _imageWidth = 1000;

    [SerializeField]
    [Tooltip("Height of captured images in pixels")]
    private int _imageHeight = 1000;

    [SerializeField]
    [Range(1, 100)]
    [Tooltip("JPEG quality (1-100, only applies if using JPEG format)")]
    private int _jpegQuality = 85;

    [Header("File Settings")]
    [SerializeField]
    [Tooltip("Image format to use for screenshots")]
    private ImageFormat _imageFormat = ImageFormat.JPG;

    [SerializeField]
    [Tooltip("Use Application.persistentDataPath for runtime builds")]
    private bool _usePersistentDataPath = true;

    [SerializeField]
    [Tooltip("Include timestamp in filename")]
    private bool _includeTimestamp = false;

    [Header("Robot Detection")]
    [SerializeField]
    [Tooltip("List of robot name patterns to search for")]
    private string[] _robotNamePatterns = { "AR4Left", "AR4Right" };

    [SerializeField]
    [Tooltip("Fallback name if no robot detected (leave empty to use camera GameObject name)")]
    private string _fallbackRobotName = "";

    // Core components
    private Camera _mainCamera;
    private MainLogger _logger;

    // State tracking
    private int _counter = 0;
    private string _robotArmName;
    private string _rootName;
    private bool _isCapturing = false;

    // Properties
    public int ImageWidth => _imageWidth;
    public int ImageHeight => _imageHeight;
    public bool IsCapturing => _isCapturing;
    public int CaptureCount => _counter;

    /// <summary>
    /// Image format options for screenshots.
    /// </summary>
    public enum ImageFormat
    {
        JPG,
        PNG,
    }

    /// <summary>
    /// Unity Start callback - initializes camera and validates configuration.
    /// </summary>
    private void Start()
    {
        try
        {
            // Get camera component
            _mainCamera = GetComponent<Camera>();
            if (_mainCamera == null)
            {
                Debug.LogError($"CameraController: No Camera component found on {gameObject.name}");
                enabled = false;
                return;
            }

            // Get logging components
            _logger = MainLogger.Instance;

            // Validate image dimensions
            if (_imageWidth <= 0 || _imageHeight <= 0)
            {
                Debug.LogWarning(
                    $"CameraController: Invalid image dimensions ({_imageWidth}x{_imageHeight}). "
                        + "Setting to default 1920x1080"
                );
                _imageWidth = 1920;
                _imageHeight = 1080;
            }

            // Find robot arm name
            _robotArmName = FindArmRoot(_mainCamera.transform);
            if (string.IsNullOrEmpty(_robotArmName))
            {
                // Use camera name as fallback if no custom fallback is set
                string fallbackName = string.IsNullOrEmpty(_fallbackRobotName)
                    ? _mainCamera.gameObject.name
                    : _fallbackRobotName;

                // Log hierarchy for debugging
                string hierarchy = GetHierarchyPath(_mainCamera.transform);
                Debug.Log(
                    $"CameraController: Could not find robot arm in hierarchy.\n"
                        + $"Camera hierarchy: {hierarchy}\n"
                        + $"Looking for patterns: [{string.Join(", ", _robotNamePatterns)}]\n"
                        + $"Using fallback: {fallbackName}"
                );
                _robotArmName = fallbackName;
            }

            // Get root name
            _rootName = _mainCamera.transform.root.name;

            // Log initialization
            _logger?.LogSimulationEvent(
                "camera_controller_initialized",
                $"Camera: {gameObject.name}, Robot: {_robotArmName}, Resolution: {_imageWidth}x{_imageHeight}"
            );

            Debug.Log(
                $"CameraController initialized: Robot={_robotArmName}, "
                    + $"Root={_rootName}, Resolution={_imageWidth}x{_imageHeight}, Format={_imageFormat}"
            );
        }
        catch (Exception ex)
        {
            Debug.LogError($"CameraController: Failed to initialize: {ex.Message}");
            enabled = false;
        }
    }

    /// <summary>
    /// Captures a screenshot from the camera and saves it to disk.
    /// </summary>
    public void CaptureAndSave()
    {
        if (_mainCamera == null)
        {
            Debug.LogError("CameraController: Cannot capture - camera is null");
            return;
        }

        if (_isCapturing)
        {
            Debug.LogWarning("CameraController: Already capturing, ignoring request");
            return;
        }

        StartCoroutine(CaptureAndSaveCoroutine());
    }

    /// <summary>
    /// Coroutine that handles the screenshot capture process.
    /// </summary>
    private IEnumerator CaptureAndSaveCoroutine()
    {
        _isCapturing = true;
        string filename = GenerateFilename();
        string relativePath = $"Screenshots/{_rootName}/{_robotArmName}/{filename}";
        string errorMessage = null;

        // Capture the image
        yield return CaptureCamera(_mainCamera, relativePath, error => errorMessage = error);

        // Check for errors and log accordingly
        if (errorMessage == null)
        {
            // Increment counter on success
            _counter++;

            // Log successful capture
            if (_logger != null)
            {
                _logger.LogAction(
                    "screenshot_captured",
                    _robotArmName,
                    filename,
                    _mainCamera.transform.position,
                    null,
                    0f,
                    true
                );
            }
        }
        else
        {
            // Log failure
            Debug.LogError($"CameraController: Capture failed: {errorMessage}");

            if (_logger != null)
            {
                _logger.LogAction(
                    "screenshot_failed",
                    _robotArmName,
                    null,
                    null,
                    null,
                    0f,
                    false,
                    errorMessage
                );
            }
        }

        _isCapturing = false;
    }

    /// <summary>
    /// Generates a filename for the screenshot based on current settings.
    /// </summary>
    /// <returns>The generated filename</returns>
    private string GenerateFilename()
    {
        string extension = _imageFormat == ImageFormat.PNG ? "png" : "jpg";
        string timestamp = _includeTimestamp ? $"_{DateTime.Now:yyyyMMdd_HHmmss}" : "";

        return $"{_counter}{timestamp}.{extension}";
    }

    /// <summary>
    /// Captures the scene from the specified camera and saves it as an image to the given path.
    /// </summary>
    /// <param name="cam">The Camera object from which to capture the scene</param>
    /// <param name="relativePath">The relative path to save the captured image</param>
    /// <param name="onError">Callback to invoke if an error occurs</param>
    private IEnumerator CaptureCamera(Camera cam, string relativePath, Action<string> onError)
    {
        RenderTexture renderTexture = null;
        Texture2D texture = null;
        string errorMsg = null;

        // Create a temporary RenderTexture
        renderTexture = new RenderTexture(_imageWidth, _imageHeight, 24);
        cam.targetTexture = renderTexture;
        cam.Render();

        // Wait for end of frame to ensure rendering is complete
        yield return new WaitForEndOfFrame();

        // Set up a Texture2D to copy the RenderTexture
        RenderTexture.active = renderTexture;
        texture = new Texture2D(_imageWidth, _imageHeight, TextureFormat.RGB24, false);
        texture.ReadPixels(new Rect(0, 0, _imageWidth, _imageHeight), 0, 0);
        texture.Apply();

        // Reset render texture
        cam.targetTexture = null;
        RenderTexture.active = null;

        // Encode image
        byte[] bytes = null;
        try
        {
            bytes =
                _imageFormat == ImageFormat.PNG
                    ? texture.EncodeToPNG()
                    : texture.EncodeToJPG(_jpegQuality);
        }
        catch (Exception ex)
        {
            errorMsg = $"Image encoding failed: {ex.Message}";
            onError?.Invoke(errorMsg);
        }

        // Save to disk asynchronously if encoding succeeded
        if (bytes != null)
        {
            string fullPath = GetFullPath(relativePath);
            yield return SaveImageAsync(fullPath, bytes, onError);

            if (errorMsg == null)
            {
                Debug.Log($"CameraController: Saved image to: {fullPath}");
            }
        }

        // Cleanup
        if (renderTexture != null)
        {
            Destroy(renderTexture);
        }
        if (texture != null)
        {
            Destroy(texture);
        }
    }

    /// <summary>
    /// Gets the full file path for saving screenshots.
    /// </summary>
    /// <param name="relativePath">The relative path</param>
    /// <returns>The full absolute path</returns>
    private string GetFullPath(string relativePath)
    {
        string basePath = _usePersistentDataPath
            ? Application.persistentDataPath
            : Application.dataPath;

        return Path.Combine(basePath, relativePath);
    }

    /// <summary>
    /// Asynchronously saves image data to disk.
    /// </summary>
    /// <param name="fullPath">The full path to save the file</param>
    /// <param name="bytes">The image bytes to write</param>
    /// <param name="onError">Callback to invoke if an error occurs</param>
    private IEnumerator SaveImageAsync(string fullPath, byte[] bytes, Action<string> onError)
    {
        bool saveComplete = false;
        string saveError = null;

        // Start async save operation
        Task.Run(() =>
        {
            try
            {
                // Ensure directory exists
                string directory = Path.GetDirectoryName(fullPath);
                if (!Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                // Write file
                File.WriteAllBytes(fullPath, bytes);
            }
            catch (Exception ex)
            {
                saveError = $"Failed to save image: {ex.Message}";
            }
            finally
            {
                saveComplete = true;
            }
        });

        // Wait for save to complete
        while (!saveComplete)
        {
            yield return null;
        }

        // Check for errors
        if (saveError != null)
        {
            onError?.Invoke(saveError);
        }
    }

    /// <summary>
    /// Gets the full hierarchy path for a transform (for debugging).
    /// </summary>
    /// <param name="transform">The transform to get the path for</param>
    /// <returns>The full hierarchy path</returns>
    private string GetHierarchyPath(Transform transform)
    {
        if (transform == null)
            return "null";

        string path = transform.name;
        Transform current = transform.parent;

        while (current != null)
        {
            path = current.name + "/" + path;
            current = current.parent;
        }

        return path;
    }

    /// <summary>
    /// Searches up the transform hierarchy to find a robot arm name matching configured patterns.
    /// </summary>
    /// <param name="current">The starting transform</param>
    /// <returns>The robot arm name if found, null otherwise</returns>
    private string FindArmRoot(Transform current)
    {
        if (current == null)
            return null;

        // Check current transform and all parents
        while (current != null)
        {
            // Check against all configured patterns
            foreach (string pattern in _robotNamePatterns)
            {
                if (string.IsNullOrEmpty(pattern))
                    continue;

                // Exact match
                if (current.name == pattern)
                {
                    return current.name;
                }

                // Contains match (e.g., "AR4Left" contains pattern)
                if (current.name.Contains(pattern))
                {
                    return current.name;
                }
            }

            // Move up to parent
            current = current.parent;
        }

        return null;
    }

    /// <summary>
    /// Resets the screenshot counter.
    /// </summary>
    public void ResetCounter()
    {
        _counter = 0;
        Debug.Log("CameraController: Counter reset to 0");
    }

    /// <summary>
    /// Sets the image format for future screenshots.
    /// </summary>
    /// <param name="format">The image format to use</param>
    public void SetImageFormat(ImageFormat format)
    {
        _imageFormat = format;
        Debug.Log($"CameraController: Image format set to {format}");
    }

    /// <summary>
    /// Sets the JPEG quality for future screenshots.
    /// </summary>
    /// <param name="quality">Quality value from 1-100</param>
    public void SetJpegQuality(int quality)
    {
        _jpegQuality = Mathf.Clamp(quality, 1, 100);
        Debug.Log($"CameraController: JPEG quality set to {_jpegQuality}");
    }

    /// <summary>
    /// Unity OnDestroy callback - logs final statistics.
    /// </summary>
    private void OnDestroy()
    {
        if (_logger != null && _counter > 0)
        {
            _logger.LogSimulationEvent(
                "camera_controller_destroyed",
                $"Camera {gameObject.name} captured {_counter} screenshots"
            );
        }
    }
}
