using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Logging;
using LLMCommunication;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

#if UNITY_EDITOR
[CustomEditor(typeof(CameraController))]
public class CameraControllerEditor : Editor
{
    private bool _lastConnectionState;

    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        var controller = (CameraController)target;

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Capture Controls", EditorStyles.boldLabel);

        // Capture buttons
        EditorGUILayout.BeginHorizontal();
        if (GUILayout.Button("Take Screenshot", GUILayout.Height(30)))
            controller.CaptureAndSave();

        if (GUILayout.Button("Send to LLM", GUILayout.Height(30)))
            controller.CaptureAndSend();
        EditorGUILayout.EndHorizontal();

        if (!Application.isPlaying)
            return;

        bool isConnected = ImageSender.Instance != null && ImageSender.Instance.IsConnected;
        string statusText = isConnected ? "Connected" : "Disconnected";
        Color statusColor = isConnected ? new Color(0.6f, 1f, 0.6f) : new Color(1f, 0.6f, 0.6f);

        var originalColor = GUI.color;
        GUI.color = statusColor;
        EditorGUILayout.TextField("Server Status", statusText, EditorStyles.boldLabel);
        GUI.color = originalColor;

        // LLM processing indicator
        if (controller.IsWaitingForLLM)
        {
            float elapsed = controller.LLMElapsedTime;
            GUI.color = new Color(1f, 0.9f, 0.4f); // Yellow color for processing
            EditorGUILayout.TextField("LLM Status", $"Processing... ({elapsed:F1}s)", EditorStyles.boldLabel);
            GUI.color = originalColor;

            // Repaint continuously while processing to update the timer
            Repaint();
        }
        else
        {
            EditorGUILayout.TextField("LLM Status", "Idle");
        }

        // Only repaint when connection status changes
        if (isConnected != _lastConnectionState)
        {
            _lastConnectionState = isConnected;
            EditorUtility.SetDirty(target);
            Repaint();
        }
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
    [Range(0.25f, 2.0f)]
    [Tooltip("Resolution scale multiplier (0.25x to 2.0x)")]
    private float _resolutionScale = 1.0f;

    [SerializeField]
    [Tooltip("Capture in grayscale (reduces memory and bandwidth by 66%)")]
    private bool _grayscaleMode = false;

    [SerializeField]
    [Tooltip("Material with a grayscale shader. Required if Grayscale Mode is enabled.")]
    private Material _grayscaleMaterial;

    [SerializeField]
    [Tooltip("Compression quality preset")]
    private CompressionPreset _compressionPreset = CompressionPreset.Balanced;

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
    private string _fallbackRobotName = "AR4";

    [Header("LLM Settings")]
    [SerializeField]
    [Tooltip("A prompt to associate with the image when sending to the LLM")]
    private string _llmPrompt = "Analyze the image";

    // Core components
    private Camera _mainCamera;
    private MainLogger _logger;

    // State tracking
    private int _captureCounter = 0;
    private string _robotArmName;
    private string _rootName;
    private bool _isCapturing = false;
    private RenderTexture _cachedRenderTexture;

    // LLM processing state
    private bool _isWaitingForLLM = false;
    private float _llmSentTime = 0f;
    private string _lastSentCameraId = "";
    private Texture2D _cachedTexture;

    // Events
    public event Action<byte[]> OnCaptureComplete;
    public event Action<string> OnCaptureFailed;

    // Properties
    public bool IsCapturing => _isCapturing;
    public int CaptureCount => _captureCounter;
    public bool IsWaitingForLLM => _isWaitingForLLM;
    public float LLMElapsedTime => _isWaitingForLLM ? Time.time - _llmSentTime : 0f;

    /// <summary>
    /// Compression quality presets.
    /// </summary>
    public enum CompressionPreset
    {
        Fast, // Low quality, fast encoding (JPEG 40)
        Balanced, // Medium quality, balanced speed (JPEG 70)
        Quality, // High quality, slower encoding (JPEG 95)
    }

    /// <summary>
    /// Capture request types.
    /// </summary>
    public enum CaptureType
    {
        SaveToFile,
        SendToServer,
    }

    private void OnValidate()
    {
        if (_grayscaleMode && _grayscaleMaterial == null)
        {
            Debug.LogWarning(
                "[CameraController] Grayscale Mode is enabled, but no Grayscale Material is assigned. Disabling mode."
            );
            _grayscaleMode = false;
        }
    }

    /// <summary>
    /// Unity Start callback - initializes camera and validates configuration.
    /// </summary>
    private void Start()
    {
        // Get camera component
        _mainCamera = GetComponent<Camera>();
        if (_mainCamera == null)
        {
            Debug.LogError($"[CAMERA_CONTROLLER] No Camera component found on {gameObject.name}");
            enabled = false;
            return;
        }

        // Get logging components
        _logger = MainLogger.Instance;

        // Validate image dimensions
        if (_imageWidth <= 0 || _imageHeight <= 0)
        {
            Debug.LogWarning(
                $"[CAMERA_CONTROLLER] Invalid image dimensions ({_imageWidth}x{_imageHeight}). "
                    + "Setting to default 1000x1000"
            );
            _imageWidth = 1000;
            _imageHeight = 1000;
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
                $"[CAMERA_CONTROLLER] Could not find robot arm in hierarchy.\n"
                    + $"Camera hierarchy: {hierarchy}\n"
                    + $"Looking for patterns: [{string.Join(", ", _robotNamePatterns)}]\n"
                    + $"Using fallback: {fallbackName}"
            );
            _robotArmName = fallbackName;
        }

        // Get root name
        _rootName = _mainCamera.transform.root.name;

        // Subscribe to LLM results
        if (LLMResultsReceiver.Instance != null)
        {
            LLMResultsReceiver.Instance.OnResultReceived += HandleLLMResult;
        }

        // Log initialization
        Debug.Log(
            $"[CAMERA_CONTROLLER] Initialized: Camera={gameObject.name}, Robot={_robotArmName}, Resolution={_imageWidth}x{_imageHeight}"
        );
    }

    /// <summary>
    /// Handles LLM result received event.
    /// </summary>
    private void HandleLLMResult(LLMResult result)
    {
        if (result == null)
            return;

        // Check if this result is for our camera
        if (result.camera_id == _lastSentCameraId && _isWaitingForLLM)
        {
            _isWaitingForLLM = false;
        }
    }

    /// <summary>
    /// Captures a screenshot and saves it to disk if not already capturing.
    /// </summary>
    public void CaptureAndSave() => InitiateCapture(CaptureType.SaveToFile);

    /// <summary>
    /// Captures a screenshot and sends it to the LLM server if not already capturing.
    /// </summary>
    public void CaptureAndSend() => InitiateCapture(CaptureType.SendToServer);

    /// <summary>
    /// Initiates the capture process if the controller isn't busy.
    /// </summary>
    private void InitiateCapture(CaptureType type)
    {
        if (_isCapturing)
        {
            Debug.LogWarning("[CameraController] A capture is already in progress. Please wait.");
            return;
        }
        if (!enabled)
        {
            Debug.LogWarning("[CameraController] Cannot capture, component is disabled.");
            return;
        }
        StartCoroutine(ProcessCaptureCoroutine(type));
    }

    /// <summary>
    /// Processes a single capture request, handling the entire pipeline.
    /// </summary>
    private IEnumerator ProcessCaptureCoroutine(CaptureType type)
    {
        _isCapturing = true;
        float startTime = Time.realtimeSinceStartup;
        string errorMessage = null;
        byte[] imageBytes = null;

        try
        {
            // Capture image
            imageBytes = CaptureImageBytes();
            Debug.Log($"[CameraController] Captured {imageBytes?.Length ?? 0} bytes");
            if (imageBytes == null)
                throw new Exception("Image capture returned null bytes.");

            // Process based on type
            switch (type)
            {
                case CaptureType.SaveToFile:
                    string filename = GenerateFilename();
                    string fullPath = GetFullPath(
                        $"Screenshots/{_rootName}/{_robotArmName}/{filename}"
                    );
                    SaveImageToFile(fullPath, imageBytes);
                    Debug.Log($"[CameraController] Image saved to: {fullPath}");
                    break;

                case CaptureType.SendToServer:
                    if (ImageSender.Instance == null || !ImageSender.Instance.IsConnected)
                    {
                        throw new Exception("ImageSender not available or not connected.");
                    }
                    ImageSender.Instance.SendImageData(imageBytes, _robotArmName, _llmPrompt);

                    // Track LLM processing state
                    _isWaitingForLLM = true;
                    _llmSentTime = Time.time;
                    _lastSentCameraId = _robotArmName;

                    Debug.Log(
                        $"[CameraController] Image sent successfully ({imageBytes.Length / 1024f:F1} KB). Waiting for LLM response..."
                    );
                    break;
            }
        }
        catch (Exception ex)
        {
            errorMessage = ex.Message;
            Debug.LogError($"[CameraController] Capture failed: {errorMessage}");
        }

        // Finalize and log
        float captureTime = Time.realtimeSinceStartup - startTime;
        if (errorMessage == null)
        {
            _captureCounter++;
            OnCaptureComplete?.Invoke(imageBytes);

            // Log successful capture to MainLogger for LLM training
            if (_logger != null)
            {
                string actionId = _logger.StartAction(
                    "CameraCapture",
                    ActionType.Observation,
                    new[] { _robotArmName },
                    description: $"Camera captured image ({type})"
                );

                _logger.CompleteAction(
                    actionId,
                    success: true,
                    qualityScore: 1.0f,
                    metrics: new Dictionary<string, float>
                    {
                        { "captureTimeMs", captureTime * 1000f },
                        { "fileSizeKB", imageBytes.Length / 1024f },
                        { "width", _imageWidth * _resolutionScale },
                        { "height", _imageHeight * _resolutionScale },
                    }
                );
            }
        }
        else
        {
            OnCaptureFailed?.Invoke(errorMessage);

            // Log failed capture to MainLogger
            if (_logger != null)
            {
                string actionId = _logger.StartAction(
                    "CameraCapture",
                    ActionType.Observation,
                    new[] { _robotArmName },
                    description: $"Camera capture attempt ({type})"
                );

                _logger.CompleteAction(actionId, success: false, errorMessage: errorMessage);
            }
        }

        _isCapturing = false;
        yield return null;
    }

    /// <summary>
    /// Captures the camera view as a byte array using GPU-accelerated processing.
    /// Must be called after WaitForEndOfFrame.
    /// </summary>
    private byte[] CaptureImageBytes()
    {
        int scaledWidth = Mathf.RoundToInt(_imageWidth * _resolutionScale);
        int scaledHeight = Mathf.RoundToInt(_imageHeight * _resolutionScale);

        // Create or reuse render texture
        if (
            _cachedRenderTexture == null
            || _cachedRenderTexture.width != scaledWidth
            || _cachedRenderTexture.height != scaledHeight
        )
        {
            if (_cachedRenderTexture != null)
            {
                Destroy(_cachedRenderTexture);
                _cachedRenderTexture = null;
            }
            _cachedRenderTexture = new RenderTexture(scaledWidth, scaledHeight, 24);
        }

        // Render camera to texture
        _mainCamera.targetTexture = _cachedRenderTexture;
        _mainCamera.Render();

        RenderTexture source = _cachedRenderTexture;

        // Apply grayscale if enabled
        if (_grayscaleMode && _grayscaleMaterial != null)
        {
            RenderTexture grayRT = RenderTexture.GetTemporary(scaledWidth, scaledHeight);
            Graphics.Blit(source, grayRT, _grayscaleMaterial);
            source = grayRT;
        }

        // Create or reuse texture2D
        if (
            _cachedTexture == null
            || _cachedTexture.width != scaledWidth
            || _cachedTexture.height != scaledHeight
        )
        {
            if (_cachedTexture != null)
            {
                Destroy(_cachedTexture);
                _cachedTexture = null;
            }
            _cachedTexture = new Texture2D(scaledWidth, scaledHeight, TextureFormat.RGB24, false);
        }

        // Read pixels from GPU to CPU
        RenderTexture.active = source;
        _cachedTexture.ReadPixels(new Rect(0, 0, scaledWidth, scaledHeight), 0, 0);
        _cachedTexture.Apply();

        // Cleanup
        _mainCamera.targetTexture = null;
        RenderTexture.active = null;
        if (source != _cachedRenderTexture)
            RenderTexture.ReleaseTemporary(source);

        // Encode to JPEG
        int quality = GetEncodingSettings();
        byte[] bytes = _cachedTexture.EncodeToJPG(quality);

        return bytes;
    }

    /// <summary>
    /// Saves image bytes to disk synchronously.
    /// </summary>
    private void SaveImageToFile(string fullPath, byte[] bytes)
    {
        string directory = Path.GetDirectoryName(fullPath);
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
        File.WriteAllBytes(fullPath, bytes);
    }

    /// <summary>
    /// Generates a filename for the screenshot based on current settings.
    /// </summary>
    private string GenerateFilename()
    {
        string extension = "jpg";
        string timestamp = _includeTimestamp ? $"_{DateTime.Now:yyyyMMdd_HHmmss}" : "";

        return $"{_captureCounter}{timestamp}.{extension}";
    }

    /// <summary>
    /// Gets the full file path for saving screenshots.
    /// </summary>
    private string GetFullPath(string relativePath)
    {
        string basePath = _usePersistentDataPath
            ? Application.persistentDataPath
            : Application.dataPath;

        return Path.Combine(basePath, relativePath);
    }

    /// <summary>
    /// Gets the full hierarchy path for a transform (for debugging).
    /// </summary>
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
    /// Gets encoding quality based on compression preset.
    /// </summary>
    private int GetEncodingSettings()
    {
        return _compressionPreset switch
        {
            CompressionPreset.Fast => 40,
            CompressionPreset.Balanced => 70,
            CompressionPreset.Quality => 95,
            _ => 85,
        };
    }

    /// <summary>
    /// Unity OnDestroy callback - logs final statistics and cleans up resources.
    /// </summary>
    private void OnDestroy()
    {
        // Unsubscribe from LLM results
        if (LLMResultsReceiver.Instance != null)
        {
            LLMResultsReceiver.Instance.OnResultReceived -= HandleLLMResult;
        }

        if (_cachedRenderTexture != null)
        {
            Destroy(_cachedRenderTexture);
            _cachedRenderTexture = null;
        }
        if (_cachedTexture != null)
        {
            Destroy(_cachedTexture);
            _cachedTexture = null;
        }

        if (_captureCounter > 0)
        {
            Debug.Log(
                $"[CAMERA_CONTROLLER] Camera {gameObject.name} destroyed after capturing {_captureCounter} screenshots"
            );
        }
    }
}
