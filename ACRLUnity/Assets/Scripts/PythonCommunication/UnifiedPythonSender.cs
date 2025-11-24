using System;
using PythonCommunication.Core;
using UnityEngine;
using Core;

namespace PythonCommunication
{
    /// <summary>
    /// Unified sender for all Python communication.
    /// Routes commands through SequenceServer (port 5013) for the new architecture.
    /// Legacy image sending methods are kept for backwards compatibility but route through SequenceClient.
    /// </summary>
    public class UnifiedPythonSender : MonoBehaviour
    {
        public static UnifiedPythonSender Instance { get; private set; }

        [Header("Default Settings")]
        [Tooltip("Default robot ID for commands")]
        [SerializeField]
        private string _defaultRobotId = "Robot1";

        [Tooltip("Default camera ID for detection")]
        [SerializeField]
        private string _defaultCameraId = "TableStereoCamera";

        private const string _logPrefix = "[UNIFIED_PYTHON_SENDER]";

        #region Singleton

        /// <summary>
        /// Initialize singleton instance
        /// </summary>
        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                Debug.Log($"{_logPrefix} Initialized - routes through SequenceClient (port 5013)");
            }
            else
            {
                Destroy(gameObject);
            }
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                Instance = null;
            }
        }

        #endregion

        #region Public API

        /// <summary>
        /// Check if connected to SequenceServer via SequenceClient
        /// </summary>
        public bool IsConnected => SequenceClient.Instance != null && SequenceClient.Instance.IsConnected;

        /// <summary>
        /// Check if single camera connection is active (legacy - now uses SequenceClient)
        /// </summary>
        public bool IsSingleCameraConnected => IsConnected;

        /// <summary>
        /// Check if stereo connection is active (legacy - now uses SequenceClient)
        /// </summary>
        public bool IsStereoConnected => IsConnected;

        /// <summary>
        /// Send a detect command for single camera analysis.
        /// Routes through SequenceClient to SequenceServer.
        /// </summary>
        /// <param name="cameraId">Camera identifier</param>
        /// <param name="prompt">Analysis prompt (e.g., "detect red cube")</param>
        /// <returns>True if command was sent successfully</returns>
        public bool SendAnalyzeCommand(string cameraId, string prompt = "analyze scene")
        {
            if (SequenceClient.Instance == null)
            {
                Debug.LogError($"{_logPrefix} SequenceClient not available");
                return false;
            }

            string command = $"analyze_scene {prompt}";
            bool success = SequenceClient.Instance.ExecuteSequence(command, _defaultRobotId);

            if (success)
            {
                Debug.Log($"{_logPrefix} Sent analyze command for camera '{cameraId}': {prompt}");
            }

            return success;
        }

        /// <summary>
        /// Send a detect command for stereo depth detection.
        /// Routes through SequenceClient to SequenceServer.
        /// </summary>
        /// <param name="cameraId">Stereo camera pair identifier</param>
        /// <param name="objectType">Object type to detect (e.g., "cube", "red cube")</param>
        /// <returns>True if command was sent successfully</returns>
        public bool SendDetectCommand(string cameraId = null, string objectType = "")
        {
            if (SequenceClient.Instance == null)
            {
                Debug.LogError($"{_logPrefix} SequenceClient not available");
                return false;
            }

            string camera = cameraId ?? _defaultCameraId;
            string command = string.IsNullOrEmpty(objectType)
                ? "detect_object"
                : $"detect_object {objectType}";

            bool success = SequenceClient.Instance.ExecuteSequence(command, _defaultRobotId);

            if (success)
            {
                Debug.Log($"{_logPrefix} Sent detect command for camera '{camera}': {objectType}");
            }

            return success;
        }

        /// <summary>
        /// Legacy method - sends detect command via SequenceClient.
        /// Kept for backwards compatibility with code that sends images directly.
        /// </summary>
        [Obsolete("Use SendDetectCommand instead. Direct image sending is no longer supported.")]
        public bool SendImage(byte[] imageBytes, string cameraId, string prompt = "")
        {
            Debug.LogWarning($"{_logPrefix} SendImage is deprecated. Use SendDetectCommand or SendAnalyzeCommand instead.");
            return SendAnalyzeCommand(cameraId, prompt);
        }

        /// <summary>
        /// Legacy method - sends detect command via SequenceClient.
        /// Kept for backwards compatibility with code that captures and sends images.
        /// </summary>
        [Obsolete("Use SendDetectCommand instead. Direct image sending is no longer supported.")]
        public bool CaptureAndSend(Camera cam, string cameraId, string prompt = "")
        {
            Debug.LogWarning($"{_logPrefix} CaptureAndSend is deprecated. Use SendDetectCommand or SendAnalyzeCommand instead.");
            return SendAnalyzeCommand(cameraId, prompt);
        }

        /// <summary>
        /// Legacy method - sends detect command via SequenceClient.
        /// Kept for backwards compatibility with code that sends stereo pairs.
        /// </summary>
        [Obsolete("Use SendDetectCommand instead. Direct image sending is no longer supported.")]
        public bool SendStereoPair(
            byte[] leftImageBytes,
            byte[] rightImageBytes,
            string cameraPairId,
            string cameraLeftId,
            string cameraRightId,
            string prompt = ""
        )
        {
            Debug.LogWarning($"{_logPrefix} SendStereoPair is deprecated. Use SendDetectCommand instead.");
            return SendDetectCommand(cameraPairId);
        }

        /// <summary>
        /// Legacy method - sends detect command via SequenceClient.
        /// Kept for backwards compatibility with code that captures stereo pairs.
        /// </summary>
        [Obsolete("Use SendDetectCommand instead. Direct image sending is no longer supported.")]
        public bool CaptureAndSendStereoPair(
            Camera leftCam,
            Camera rightCam,
            string prompt = "",
            string cameraPairId = null
        )
        {
            Debug.LogWarning($"{_logPrefix} CaptureAndSendStereoPair is deprecated. Use SendDetectCommand instead.");
            return SendDetectCommand(cameraPairId);
        }

        #endregion
    }
}
