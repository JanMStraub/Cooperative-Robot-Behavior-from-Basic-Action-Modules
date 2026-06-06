using System.Collections;
using Core;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

namespace Simulation
{
    /// <summary>
    /// Singleton that initializes and manages the ROSConnection instance.
    /// Configures ROS IP, port, and provides connection health monitoring
    /// with automatic reconnection.
    /// </summary>
    public class ROSConnectionInitializer : MonoBehaviour
    {
        public static ROSConnectionInitializer Instance { get; private set; }

        [Header("ROS Connection Settings")]
        [Tooltip("ROS bridge host IP address")]
        [SerializeField]
        private string _rosHost = CommunicationConstants.SERVER_HOST;

        [Tooltip("ROS bridge port (ros_tcp_endpoint)")]
        [SerializeField]
        private int _rosPort = CommunicationConstants.ROS_TCP_ENDPOINT_PORT;

        [Header("Connection Management")]
        [Tooltip("Attempt to connect on start")]
        [SerializeField]
        private bool _connectOnStart = true;

        [Tooltip("Enable automatic reconnection on connection loss")]
        [SerializeField]
        private bool _autoReconnect = true;

        [Tooltip("Health check interval in seconds")]
        [SerializeField]
        [Range(1f, 30f)]
        private float _healthCheckInterval = 5f;

        [Header("Runtime Info")]
        [SerializeField]
        private int _reconnectAttempts;

        private ROSConnection _rosConnection;
        private Coroutine _healthCheckCoroutine;
        private const string _logPrefix = "[ROS_CONNECTION_INITIALIZER]";

        // HasConnectionThread check prevents publishers from queueing before Connect() is called (HasConnectionError starts false).
        public bool IsConnected =>
            _rosConnection != null
            && _rosConnection.HasConnectionThread
            && !_rosConnection.HasConnectionError;

        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);

                // Configure in Awake so settings are ready before other scripts call GetOrCreateInstance() in Start().
                _rosConnection = ROSConnection.GetOrCreateInstance();
                _rosConnection.RosIPAddress = _rosHost;
                _rosConnection.RosPort = _rosPort;
                _rosConnection.ConnectOnStart = false;

                Debug.Log(
                    $"{_logPrefix} ROS connection configured: {_rosHost}:{_rosPort} (connect delayed)"
                );
            }
            else
            {
                Destroy(gameObject);
                return;
            }
        }

        private void Start()
        {
            if (_connectOnStart)
            {
                StartCoroutine(DelayedConnect());
            }

            if (_autoReconnect)
            {
                _healthCheckCoroutine = StartCoroutine(HealthCheckLoop());
            }
        }

        // Short delay lets ros_tcp_endpoint fully start; avoids noisy stack traces on the first failed attempt.
        private IEnumerator DelayedConnect()
        {
            yield return new WaitForSeconds(2f);

            _rosConnection.Connect(_rosHost, _rosPort);
            _reconnectAttempts = 0;
            Debug.Log($"{_logPrefix} ROS connection initiated: {_rosHost}:{_rosPort}");
        }

        public void InitializeConnection()
        {
            _rosConnection = ROSConnection.GetOrCreateInstance();
            _rosConnection.RosIPAddress = _rosHost;
            _rosConnection.RosPort = _rosPort;

            _rosConnection.Connect(_rosHost, _rosPort);
            _reconnectAttempts = 0;

            Debug.Log($"{_logPrefix} ROS connection configured: {_rosHost}:{_rosPort}");

            if (_autoReconnect && _healthCheckCoroutine == null)
            {
                _healthCheckCoroutine = StartCoroutine(HealthCheckLoop());
            }
        }

        private IEnumerator HealthCheckLoop()
        {
            var wait = new WaitForSeconds(_healthCheckInterval);

            while (true)
            {
                yield return wait;

                if (
                    _autoReconnect
                    && _rosConnection != null
                    && _rosConnection.HasConnectionThread
                    && _rosConnection.HasConnectionError
                )
                {
                    _reconnectAttempts++;
                    Debug.LogWarning(
                        $"{_logPrefix} ROS connection error detected. Reconnect attempt #{_reconnectAttempts}"
                    );
                    InitializeConnection();
                }
            }
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                if (_healthCheckCoroutine != null)
                    StopCoroutine(_healthCheckCoroutine);
                Instance = null;
            }
        }
    }
}
