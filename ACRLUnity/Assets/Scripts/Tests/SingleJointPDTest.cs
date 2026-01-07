using UnityEngine;
using System.Collections.Generic;

namespace Tests
{
    /// <summary>
    /// Phase 0 validation test: Proves that PD control eliminates oscillation
    /// compared to P-only control on a single joint.
    ///
    /// Usage:
    /// 1. Attach to a GameObject with an ArticulationBody joint
    /// 2. Toggle 'useVelocityDamping' to compare P-only vs PD control
    /// 3. Observe position graph and settling behavior
    /// 4. Press 'R' to change target and re-test
    /// </summary>
    public class SingleJointPDTest : MonoBehaviour
    {
        [Header("Joint Configuration")]
        public ArticulationBody joint;

        [Header("Control Mode")]
        [Tooltip("Enable to use PD control (position + velocity), disable for P-only control")]
        public bool useVelocityDamping = true;

        [Header("PD Gains")]
        [Tooltip("Position gain (K_p) - how strongly to correct position error")]
        [Range(0f, 50f)]
        public float positionGain = 10f;

        [Tooltip("Velocity gain (K_d) - damping term to prevent overshoot")]
        [Range(0f, 10f)]
        public float velocityGain = 2f;

        [Header("Test Configuration")]
        [Tooltip("Target joint angle in degrees")]
        public float targetAngleDegrees = 90f;

        [Tooltip("Enable to graph position over time")]
        public bool enableGraphing = true;

        [Tooltip("Duration to track convergence (seconds)")]
        public float graphDuration = 5f;

        // Performance tracking
        private List<float> _positionHistory = new List<float>();
        private List<float> _velocityHistory = new List<float>();
        private float _testStartTime;
        private bool _hasSettled = false;
        private float _settlingTime = 0f;

        // Convergence detection
        private const float CONVERGENCE_THRESHOLD = 0.01f; // 0.01 rad ≈ 0.57 degrees
        private const float VELOCITY_THRESHOLD = 0.05f; // rad/sec
        private const float CONVERGENCE_DURATION = 0.5f; // Must stay converged for 0.5s
        private float _convergenceTimer = 0f;

        void Start()
        {
            if (joint == null)
            {
                joint = GetComponent<ArticulationBody>();
                if (joint == null)
                {
                    Debug.LogError("[SingleJointPDTest] No ArticulationBody found! Please assign a joint.");
                    enabled = false;
                    return;
                }
            }

            ResetTest();
            Debug.Log($"[SingleJointPDTest] Starting test with {(useVelocityDamping ? "PD" : "P-only")} control");
            Debug.Log($"[SingleJointPDTest] Kp={positionGain}, Kd={velocityGain}");
        }

        void Update()
        {
            // Allow runtime target change for testing
            if (Input.GetKeyDown(KeyCode.R))
            {
                targetAngleDegrees = Random.Range(-90f, 90f);
                ResetTest();
                Debug.Log($"[SingleJointPDTest] New target: {targetAngleDegrees:F1}°");
            }

            // Toggle control mode
            if (Input.GetKeyDown(KeyCode.T))
            {
                useVelocityDamping = !useVelocityDamping;
                ResetTest();
                Debug.Log($"[SingleJointPDTest] Switched to {(useVelocityDamping ? "PD" : "P-only")} control");
            }
        }

        void FixedUpdate()
        {
            if (joint == null) return;

            float currentAngle = joint.jointPosition[0]; // radians
            float currentVelocity = joint.jointVelocity[0]; // rad/sec
            float targetAngle = targetAngleDegrees * Mathf.Deg2Rad;

            // PD control law
            float posError = targetAngle - currentAngle;
            float velError = 0f - currentVelocity; // Target velocity = 0 (stationary target)

            // Compute correction
            float correction;
            if (useVelocityDamping)
            {
                // PD control: correction = K_p * posError + K_d * velError
                correction = positionGain * posError + velocityGain * velError;
            }
            else
            {
                // P-only control: correction = K_p * posError
                correction = positionGain * posError;
            }

            // Apply correction to drive target
            var drive = joint.xDrive;
            drive.target = (currentAngle + correction) * Mathf.Rad2Deg;
            joint.xDrive = drive;

            // Track performance
            TrackPerformance(currentAngle, currentVelocity, targetAngle);
        }

        void TrackPerformance(float currentAngle, float currentVelocity, float targetAngle)
        {
            float elapsed = Time.time - _testStartTime;

            // Stop tracking after graph duration
            if (elapsed > graphDuration)
                return;

            // Record history for graphing
            if (enableGraphing && _positionHistory.Count < 1000)
            {
                _positionHistory.Add(currentAngle);
                _velocityHistory.Add(currentVelocity);
            }

            // Detect convergence
            float posError = Mathf.Abs(targetAngle - currentAngle);
            float velMagnitude = Mathf.Abs(currentVelocity);

            if (posError < CONVERGENCE_THRESHOLD && velMagnitude < VELOCITY_THRESHOLD)
            {
                _convergenceTimer += Time.fixedDeltaTime;

                if (!_hasSettled && _convergenceTimer >= CONVERGENCE_DURATION)
                {
                    _hasSettled = true;
                    _settlingTime = elapsed;
                    Debug.Log($"[SingleJointPDTest] ✅ CONVERGED at t={_settlingTime:F2}s " +
                              $"(error={posError * Mathf.Rad2Deg:F3}°, vel={velMagnitude:F3} rad/s)");

                    // Calculate oscillation metric
                    float oscillation = CalculateOscillation();
                    Debug.Log($"[SingleJointPDTest] Oscillation metric: {oscillation:F4} " +
                              $"({(oscillation < 0.01f ? "EXCELLENT" : oscillation < 0.05f ? "GOOD" : "POOR")})");
                }
            }
            else
            {
                _convergenceTimer = 0f; // Reset if we leave convergence zone
            }
        }

        float CalculateOscillation()
        {
            if (_positionHistory.Count < 10)
                return float.MaxValue;

            // Calculate position variance over last 50 samples (after settling)
            int startIdx = Mathf.Max(0, _positionHistory.Count - 50);
            float sum = 0f;
            float mean = 0f;

            for (int i = startIdx; i < _positionHistory.Count; i++)
            {
                mean += _positionHistory[i];
            }
            mean /= (_positionHistory.Count - startIdx);

            for (int i = startIdx; i < _positionHistory.Count; i++)
            {
                float diff = _positionHistory[i] - mean;
                sum += diff * diff;
            }

            return Mathf.Sqrt(sum / (_positionHistory.Count - startIdx));
        }

        void ResetTest()
        {
            _testStartTime = Time.time;
            _positionHistory.Clear();
            _velocityHistory.Clear();
            _hasSettled = false;
            _settlingTime = 0f;
            _convergenceTimer = 0f;
        }

        void OnDrawGizmos()
        {
            if (!enableGraphing || _positionHistory.Count < 2)
                return;

            // Draw position graph in 3D space above the joint
            Vector3 graphOrigin = transform.position + Vector3.up * 2f;
            float graphWidth = 5f;
            float graphHeight = 2f;
            float targetAngle = targetAngleDegrees * Mathf.Deg2Rad;

            // Draw axes
            Gizmos.color = Color.white;
            Gizmos.DrawLine(graphOrigin, graphOrigin + Vector3.right * graphWidth);
            Gizmos.DrawLine(graphOrigin, graphOrigin + Vector3.up * graphHeight);

            // Draw target line
            Gizmos.color = Color.green;
            float targetY = (targetAngle + 1.57f) / 3.14f * graphHeight; // Normalize to graph
            Gizmos.DrawLine(graphOrigin + Vector3.up * targetY,
                           graphOrigin + Vector3.up * targetY + Vector3.right * graphWidth);

            // Draw position history
            Gizmos.color = useVelocityDamping ? Color.cyan : Color.red;
            for (int i = 0; i < _positionHistory.Count - 1; i++)
            {
                float x1 = (float)i / _positionHistory.Count * graphWidth;
                float y1 = (_positionHistory[i] + 1.57f) / 3.14f * graphHeight;
                float x2 = (float)(i + 1) / _positionHistory.Count * graphWidth;
                float y2 = (_positionHistory[i + 1] + 1.57f) / 3.14f * graphHeight;

                Gizmos.DrawLine(graphOrigin + new Vector3(x1, y1, 0),
                               graphOrigin + new Vector3(x2, y2, 0));
            }

            // Draw convergence indicator
            if (_hasSettled)
            {
                Gizmos.color = Color.green;
                float settleX = (_settlingTime / graphDuration) * graphWidth;
                Gizmos.DrawWireSphere(graphOrigin + new Vector3(settleX, graphHeight * 0.5f, 0), 0.1f);
            }
        }

        void OnGUI()
        {
            if (!enableGraphing) return;

            GUIStyle style = new GUIStyle(GUI.skin.box);
            style.fontSize = 14;
            style.alignment = TextAnchor.UpperLeft;

            string info = $"<b>Single Joint PD Test</b>\n" +
                         $"Control Mode: {(useVelocityDamping ? "PD (Position + Velocity)" : "P-only (Position)")}\n" +
                         $"Kp={positionGain:F1}, Kd={velocityGain:F1}\n" +
                         $"Target: {targetAngleDegrees:F1}°\n" +
                         $"Current: {joint.jointPosition[0] * Mathf.Rad2Deg:F1}°\n" +
                         $"Velocity: {joint.jointVelocity[0]:F2} rad/s\n" +
                         $"Settled: {(_hasSettled ? $"✅ Yes ({_settlingTime:F2}s)" : "❌ No")}\n\n" +
                         $"<color=yellow>Press 'T' to toggle PD/P-only</color>\n" +
                         $"<color=yellow>Press 'R' for random target</color>";

            GUI.Box(new Rect(10, 10, 300, 200), info, style);
        }
    }
}
