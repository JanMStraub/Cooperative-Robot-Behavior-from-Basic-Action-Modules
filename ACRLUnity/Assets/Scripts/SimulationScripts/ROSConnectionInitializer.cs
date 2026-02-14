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

        /// <summary>
        /// Whether the ROS connection is currently active.
        /// Requires a connection thread to be running AND no connection errors
        /// (HasConnectionError starts false, so without the thread check,
        /// publishers would queue messages before Connect() is even called).
        /// </summary>
        public bool IsConnected =>
            _rosConnection != null
            && _rosConnection.HasConnectionThread
            && !_rosConnection.HasConnectionError;

        /// <summary>
        /// The configured ROS host IP.
        /// </summary>
        public string ROSHost => _rosHost;

        /// <summary>
        /// The configured ROS port.
        /// </summary>
        public int ROSPort => _rosPort;

        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);

                // Configure ROSConnection early in Awake so settings are ready
                // before other scripts call GetOrCreateInstance() in Start().
                // Disable auto-connect so we can delay the connection until
                // Docker ROS services are fully started.
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

        /// <summary>
        /// Delay connection to give ros_tcp_endpoint time to fully start.
        /// ROSConnection has built-in retry logic, but the first failed attempt
        /// produces noisy stack traces — a short delay avoids most of them.
        /// </summary>
        private IEnumerator DelayedConnect()
        {
            yield return new WaitForSeconds(2f);

            _rosConnection.Connect(_rosHost, _rosPort);
            _reconnectAttempts = 0;
            Debug.Log($"{_logPrefix} ROS connection initiated: {_rosHost}:{_rosPort}");
        }

        /// <summary>
        /// Initialize the ROS connection with configured settings.
        /// </summary>
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

        /// <summary>
        /// Periodic health check that monitors connection status.
        /// </summary>
        private IEnumerator HealthCheckLoop()
        {
            var wait = new WaitForSeconds(_healthCheckInterval);

            while (true)
            {
                yield return wait;

                if (_rosConnection == null)
                {
                    if (_autoReconnect)
                    {
                        _reconnectAttempts++;
                        Debug.LogWarning(
                            $"{_logPrefix} ROS connection lost. Reconnect attempt #{_reconnectAttempts}"
                        );
                        InitializeConnection();
                    }
                }
            }
        }

        /// <summary>
        /// Reconfigure and reconnect with new settings.
        /// </summary>
        public void Reconnect(string host, int port)
        {
            _rosHost = host;
            _rosPort = port;

            if (_healthCheckCoroutine != null)
            {
                StopCoroutine(_healthCheckCoroutine);
                _healthCheckCoroutine = null;
            }

            InitializeConnection();
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
