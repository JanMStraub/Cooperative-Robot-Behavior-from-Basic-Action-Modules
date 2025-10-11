using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

/// <summary>
/// Sends camera images to a Python server via TCP for LLM vision processing.
/// Supports both continuous streaming and on-demand image sending.
/// </summary>
public class ImageSender : MonoBehaviour
{
    public static ImageSender Instance { get; private set; }

    [Header("Connection Settings")]
    [SerializeField]
    [Tooltip("IP address of the Python server")]
    private string _serverIP = "127.0.0.1";

    [SerializeField]
    [Tooltip("Port for TCP connection")]
    private int _serverPort = 5005;

    [SerializeField]
    [Tooltip("Automatically reconnect if connection is lost")]
    private bool _autoReconnect = true;

    [SerializeField]
    [Tooltip("Time between reconnection attempts in seconds")]
    private float _reconnectInterval = 2f;

    [Header("Streaming Settings (Optional)")]
    [SerializeField]
    [Tooltip("Enable continuous streaming mode")]
    private bool _enableStreaming = false;

    [SerializeField]
    [Tooltip("Time between streamed images in seconds")]
    private float _sendInterval = 0.2f;

    [SerializeField]
    [Tooltip("Camera to stream from (only used in streaming mode)")]
    private Camera _streamCamera;

    [SerializeField]
    [Tooltip("Camera identifier for streaming")]
    private string _streamCameraId = "Main";

    // Connection state
    private TcpClient _client;
    private NetworkStream _stream;
    private bool _isConnected = false;
    private float _reconnectTimer = 0f;
    private float _streamTimer = 0f;

    // Properties
    public bool IsConnected => _isConnected && _client != null && _client.Connected;
    public string ServerIP => _serverIP;
    public int ServerPort => _serverPort;

    /// <summary>
    /// Unity Awake callback - initializes singleton.
    /// </summary>
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

    /// <summary>
    /// Unity Start callback - establishes initial connection.
    /// </summary>
    private void Start()
    {
        ConnectToServer();
    }

    /// <summary>
    /// Unity Update callback - handles reconnection and streaming.
    /// </summary>
    private void Update()
    {
        // Handle reconnection
        if (_autoReconnect && !IsConnected)
        {
            _reconnectTimer += Time.deltaTime;
            if (_reconnectTimer >= _reconnectInterval)
            {
                ConnectToServer();
                _reconnectTimer = 0f;
            }
        }

        // Handle continuous streaming if enabled
        if (_enableStreaming && IsConnected && _streamCamera != null)
        {
            _streamTimer += Time.deltaTime;
            if (_streamTimer >= _sendInterval)
            {
                CaptureAndSendCamera(_streamCamera, _streamCameraId);
                _streamTimer = 0f;
            }
        }
    }

    /// <summary>
    /// Establishes TCP connection to the Python server.
    /// </summary>
    private void ConnectToServer()
    {
        try
        {
            _client = new TcpClient(_serverIP, _serverPort);
            _stream = _client.GetStream();
            _isConnected = true;
            Debug.Log($"[IMAGE_SENDER] Connected to {_serverIP}:{_serverPort}");
        }
        catch (Exception e)
        {
            _isConnected = false;
            Debug.LogWarning($"[IMAGE_SENDER] Connection failed: {e.Message}");
        }
    }

    /// <summary>
    /// Sends pre-encoded image data to the Python server.
    /// </summary>
    /// <param name="imageBytes">Encoded image data (PNG/JPG)</param>
    /// <param name="cameraId">Identifier for the camera</param>
    /// <param name="prompt">Optional prompt for LLM vision processing</param>
    /// <returns>True if sent successfully, false otherwise</returns>
    public bool SendImageData(byte[] imageBytes, string cameraId, string prompt = "")
    {
        if (!IsConnected)
        {
            Debug.LogWarning("[IMAGE_SENDER] Cannot send image - not connected to server");
            return false;
        }

        if (imageBytes == null || imageBytes.Length == 0)
        {
            Debug.LogError("[IMAGE_SENDER] Cannot send empty image data");
            return false;
        }

        if (string.IsNullOrEmpty(cameraId))
        {
            Debug.LogWarning("[IMAGE_SENDER] Camera ID is empty, using 'Unknown'");
            cameraId = "Unknown";
        }

        // Ensure prompt is not null
        if (prompt == null)
        {
            prompt = "";
        }

        try
        {
            // Send camera ID header (UTF-8 string with length prefix)
            byte[] idBytes = Encoding.UTF8.GetBytes(cameraId);
            byte[] idLength = BitConverter.GetBytes(idBytes.Length);
            _stream.Write(idLength, 0, idLength.Length);
            _stream.Write(idBytes, 0, idBytes.Length);

            // Send prompt header (UTF-8 string with length prefix)
            byte[] promptBytes = Encoding.UTF8.GetBytes(prompt);
            byte[] promptLength = BitConverter.GetBytes(promptBytes.Length);
            _stream.Write(promptLength, 0, promptLength.Length);
            _stream.Write(promptBytes, 0, promptBytes.Length);

            // Send image size (4 bytes)
            byte[] sizeInfo = BitConverter.GetBytes(imageBytes.Length);
            _stream.Write(sizeInfo, 0, sizeInfo.Length);

            // Send image data
            _stream.Write(imageBytes, 0, imageBytes.Length);
            _stream.Flush();

            string promptInfo = string.IsNullOrEmpty(prompt) ? "" : $" with prompt: '{prompt}'";
            Debug.Log($"[IMAGE_SENDER] Sent {imageBytes.Length} bytes for camera '{cameraId}'{promptInfo}");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"[IMAGE_SENDER] Error sending image: {e.Message}");
            _isConnected = false;
            return false;
        }
    }

    /// <summary>
    /// Captures image from camera and sends it to the Python server.
    /// </summary>
    /// <param name="cam">Camera to capture from</param>
    /// <param name="cameraId">String identifier for the camera</param>
    /// <param name="prompt">Optional prompt for LLM vision processing</param>
    /// <returns>True if captured and sent successfully, false otherwise</returns>
    public bool CaptureAndSendCamera(Camera cam, string cameraId, string prompt = "")
    {
        if (cam == null)
        {
            Debug.LogError("[IMAGE_SENDER] Camera is null");
            return false;
        }

        if (!IsConnected)
        {
            Debug.LogWarning("[IMAGE_SENDER] Cannot send image - not connected to server");
            return false;
        }

        RenderTexture rt = null;
        Texture2D texture = null;

        try
        {
            // Create temporary render texture
            rt = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 24);
            cam.targetTexture = rt;
            cam.Render();

            // Read pixels into Texture2D
            RenderTexture.active = rt;
            texture = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            texture.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            texture.Apply();

            // Encode and send
            byte[] imageData = texture.EncodeToPNG();
            bool success = SendImageData(imageData, cameraId, prompt);

            return success;
        }
        catch (Exception e)
        {
            Debug.LogError($"[IMAGE_SENDER] Error capturing/sending image: {e.Message}");
            return false;
        }
        finally
        {
            // Cleanup
            if (cam != null)
            {
                cam.targetTexture = null;
            }
            RenderTexture.active = null;

            if (rt != null)
            {
                Destroy(rt);
            }
            if (texture != null)
            {
                Destroy(texture);
            }
        }
    }

    /// <summary>
    /// Manually disconnects from the server.
    /// </summary>
    public void Disconnect()
    {
        try
        {
            _stream?.Close();
            _client?.Close();
            _isConnected = false;
            Debug.Log("[IMAGE_SENDER] Disconnected from server");
        }
        catch (Exception e)
        {
            Debug.LogError($"[IMAGE_SENDER] Error during disconnect: {e.Message}");
        }
    }

    /// <summary>
    /// Manually reconnects to the server.
    /// </summary>
    public void Reconnect()
    {
        Disconnect();
        ConnectToServer();
    }

    /// <summary>
    /// Unity OnDestroy callback - cleanup connection.
    /// </summary>
    private void OnDestroy()
    {
        Disconnect();
    }

    /// <summary>
    /// Unity OnApplicationQuit callback - cleanup connection.
    /// </summary>
    private void OnApplicationQuit()
    {
        Disconnect();
    }
}
