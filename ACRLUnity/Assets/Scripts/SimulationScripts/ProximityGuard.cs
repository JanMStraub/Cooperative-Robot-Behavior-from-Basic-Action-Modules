using Core;
using Robotics;
using UnityEngine;

namespace Simulation
{
    /// <summary>
    /// Checks inter-robot proximity every FixedUpdate.
    /// When EE-to-EE distance (or any monitored link pair) drops below EE_STOP_THRESHOLD,
    /// both robots are frozen via RobotController.IsFrozenByProximity.
    /// Robots unfreeze automatically once separation exceeds EE_RESUME_THRESHOLD.
    /// Python receives the frozen state via RobotStateData.proximity_frozen on the next
    /// WorldStatePublisher cycle and can trigger a replan.
    /// </summary>
    public class ProximityGuard : MonoBehaviour
    {
        public static ProximityGuard Instance { get; private set; }

        [Header("Configuration")]
        [SerializeField]
        private bool _enabled = true;

        [SerializeField]
        private bool _enableLinkChecks = true;

        [Tooltip(
            "Override EE stop threshold in meters. 0 = use ProximityConstants.EE_STOP_THRESHOLD."
        )]
        [SerializeField]
        private float _eeStopOverride = 0f;

        [Header("Debug")]
        [SerializeField]
        private bool _verboseLogging = false;

        private RobotManager _robotManager;

        // Pre-allocated controller cache — rebuilt only when robot count changes, not per-frame.
        private RobotController[] _cache = new RobotController[0];
        private int _cacheCount = 0;
        private int _lastCount = 0;

        private float _eeStop;
        private float _eeResume;
        private float _linkStop;

        private const string LOG = "[PROXIMITY_GUARD]";

        private void Awake()
        {
            if (Instance == null)
                Instance = this;
            else
            {
                Destroy(this);
                return;
            }
        }

        private void Start()
        {
            _robotManager = RobotManager.Instance;
            if (_robotManager == null)
            {
                Debug.LogError($"{LOG} RobotManager not found — disabling ProximityGuard.");
                enabled = false;
                return;
            }

            _eeStop = _eeStopOverride > 0f ? _eeStopOverride : ProximityConstants.EE_STOP_THRESHOLD;
            _eeResume =
                _eeStop
                + (ProximityConstants.EE_RESUME_THRESHOLD - ProximityConstants.EE_STOP_THRESHOLD);
            _linkStop = ProximityConstants.LINK_STOP_THRESHOLD;

            Debug.Log(
                $"{LOG} Initialized. EE stop={_eeStop:F3}m, resume={_eeResume:F3}m, linkChecks={_enableLinkChecks}"
            );
        }

        private void FixedUpdate()
        {
            if (!_enabled || _robotManager == null)
                return;

            RefreshCache();

            if (_cacheCount < 2)
                return;

            for (int i = 0; i < _cacheCount - 1; i++)
            for (int j = i + 1; j < _cacheCount; j++)
                CheckPair(_cache[i], _cache[j]);
        }

        /// <summary>
        /// Checks one robot pair for proximity violation and updates freeze state with hysteresis.
        /// </summary>
        private void CheckPair(RobotController a, RobotController b)
        {
            float eeDist = Vector3.Distance(
                a.GetCurrentEndEffectorPosition(),
                b.GetCurrentEndEffectorPosition()
            );

            bool violation = eeDist < _eeStop || (_enableLinkChecks && CheckLinks(a, b));
            bool resolved = eeDist >= _eeResume && (!_enableLinkChecks || !CheckLinks(a, b));

            SetFreeze(a, violation, resolved);
            SetFreeze(b, violation, resolved);
        }

        /// <summary>
        /// Returns true if any monitored link pair is closer than LINK_STOP_THRESHOLD.
        /// </summary>
        private bool CheckLinks(RobotController a, RobotController b)
        {
            var ja = a.robotJoints;
            var jb = b.robotJoints;
            if (ja == null || jb == null)
                return false;

            foreach (int idx in ProximityConstants.MONITORED_LINK_INDICES)
            {
                if (idx >= ja.Length || idx >= jb.Length)
                    continue;
                if (ja[idx] == null || jb[idx] == null)
                    continue;

                float d = Vector3.Distance(ja[idx].transform.position, jb[idx].transform.position);

                if (d < _linkStop)
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Applies or clears the proximity freeze on a single robot using hysteresis logic.
        /// </summary>
        private void SetFreeze(RobotController r, bool violation, bool resolved)
        {
            if (violation && !r.IsFrozenByProximity)
            {
                r.IsFrozenByProximity = true;
                if (_verboseLogging)
                    Debug.LogWarning($"{LOG} Froze {r.robotId} — proximity violation");
            }
            else if (resolved && r.IsFrozenByProximity)
            {
                r.IsFrozenByProximity = false;
                if (_verboseLogging)
                    Debug.Log($"{LOG} Unfroze {r.robotId} — separation restored");
            }
            // In hysteresis band: maintain current state unchanged.
        }

        /// <summary>
        /// Temporarily enable or disable proximity checking (e.g. during a handoff approach).
        /// When disabled all currently-frozen robots are also unfrozen so they can continue moving.
        /// </summary>
        public static void SetEnabled(bool value)
        {
            if (Instance == null)
                return;
            Instance._enabled = value;
            if (!value)
            {
                // Unfreeze all robots so they can continue approaching.
                foreach (var kvp in Instance._robotManager.RobotInstances)
                    if (kvp.Value.controller != null)
                        kvp.Value.controller.IsFrozenByProximity = false;
            }
        }

        /// <summary>
        /// Rebuilds the flat controller cache when robot count changes. Zero-allocation in steady state.
        /// </summary>
        private void RefreshCache()
        {
            int n = _robotManager.RobotInstances.Count;
            if (n == _lastCount)
                return;

            if (_cache.Length < n)
                _cache = new RobotController[n];

            _cacheCount = 0;
            foreach (var kvp in _robotManager.RobotInstances)
                if (kvp.Value.controller != null)
                    _cache[_cacheCount++] = kvp.Value.controller;

            _lastCount = n;
        }
    }
}
