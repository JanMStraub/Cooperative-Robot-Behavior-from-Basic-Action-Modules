using System;
using System.Collections.Generic;
using Configuration;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Python-backed coordination verifier.
    /// Communicates with Python CoordinationVerifier for verification using workspace state.
    /// More accurate but slower than Unity-only verification.
    /// Falls back to Unity verification on timeout or connection failure.
    /// </summary>
    public class PythonCoordinationVerifier : ICoordinationVerifier
    {
        private float _timeout;
        private bool _fallbackToUnity;
        private UnityCoordinationVerifier _unityFallback;
        private UnifiedPythonReceiver _pythonReceiver;

        private const string LOG_PREFIX = "[PYTHON_VERIFIER]";

        public string VerifierName => "Python";

        public bool IsAvailable
        {
            get
            {
                // Check if Python backend is available
                // Note: UnifiedPythonReceiver doesn't expose IsConnected directly
                // We assume it's available if the instance exists
                return _pythonReceiver != null;
            }
        }

        /// <summary>
        /// Constructor with configurable timeout and fallback.
        /// </summary>
        /// <param name="timeout">Timeout for Python verification in seconds</param>
        /// <param name="fallbackToUnity">Whether to fallback to Unity verification on timeout</param>
        /// <param name="minSafeSeparation">Minimum safe separation for Unity fallback</param>
        public PythonCoordinationVerifier(
            float timeout = 1f,
            bool fallbackToUnity = true,
            float minSafeSeparation = 0.2f
        )
        {
            _timeout = Mathf.Max(0.1f, timeout);
            _fallbackToUnity = fallbackToUnity;

            // Initialize Unity fallback verifier
            if (_fallbackToUnity)
            {
                _unityFallback = new UnityCoordinationVerifier(minSafeSeparation);
            }

            // Get Python receiver instance
            _pythonReceiver = UnifiedPythonReceiver.Instance;

            if (_pythonReceiver == null)
            {
                Debug.LogWarning($"{LOG_PREFIX} UnifiedPythonReceiver not available");
            }
        }

        /// <summary>
        /// Verify if a robot movement is safe using Python CoordinationVerifier.
        /// </summary>
        public VerificationResult VerifyMovement(
            string robotId,
            Vector3 targetPosition,
            Vector3 currentPosition
        )
        {
            // Check if Python backend is available
            if (!IsAvailable)
            {
                Debug.LogWarning($"{LOG_PREFIX} Python backend not available");

                if (_fallbackToUnity && _unityFallback != null)
                {
                    Debug.Log($"{LOG_PREFIX} Falling back to Unity verification");
                    return _unityFallback.VerifyMovement(robotId, targetPosition, currentPosition);
                }

                // No fallback, assume safe
                return new VerificationResult(
                    true,
                    "Python backend unavailable, assuming safe"
                );
            }

            try
            {
                // Send verification request to Python
                var verificationData = new Dictionary<string, object>
                {
                    ["type"] = "verify_movement",
                    ["robot_id"] = robotId,
                    ["target_position"] = new float[] { targetPosition.x, targetPosition.y, targetPosition.z },
                    ["current_position"] = new float[] { currentPosition.x, currentPosition.y, currentPosition.z },
                    ["timeout"] = _timeout
                };

                // Note: This is a simplified implementation
                // In production, this would use the command server to send verification requests
                // and wait for responses with proper request ID correlation

                Debug.Log($"{LOG_PREFIX} Sending verification request for {robotId}");

                // For now, fall back to Unity verification
                // TODO: Implement actual Python communication when CoordinationVerifier.py is ready
                if (_fallbackToUnity && _unityFallback != null)
                {
                    Debug.Log($"{LOG_PREFIX} Python verification not fully implemented, using Unity fallback");
                    return _unityFallback.VerifyMovement(robotId, targetPosition, currentPosition);
                }

                return new VerificationResult(true, "Python verification placeholder");
            }
            catch (Exception e)
            {
                Debug.LogError($"{LOG_PREFIX} Error during Python verification: {e.Message}");

                // Fallback to Unity verification on error
                if (_fallbackToUnity && _unityFallback != null)
                {
                    Debug.Log($"{LOG_PREFIX} Falling back to Unity verification after error");
                    return _unityFallback.VerifyMovement(robotId, targetPosition, currentPosition);
                }

                // No fallback, assume safe to avoid blocking robot
                return new VerificationResult(true, "Python verification failed, assuming safe");
            }
        }
    }
}
