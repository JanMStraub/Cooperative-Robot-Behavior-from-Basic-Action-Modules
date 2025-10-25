using System.Collections.Generic;
using LLMCommunication;
using UnityEngine;

/// <summary>
/// Visualizes detected cubes in the Unity scene using Gizmos and debug spheres.
/// Attach this component to any GameObject in the scene to see detection results.
/// </summary>
public class CubeDetectionVisualizer : MonoBehaviour
{
    [Header("Visualization Settings")]
    [SerializeField]
    [Tooltip("Show Gizmos in scene view")]
    private bool _showGizmos = true;

    [SerializeField]
    [Tooltip("Create debug sphere GameObjects at detected positions")]
    private bool _createDebugSpheres = true;

    [SerializeField]
    [Tooltip("How long to keep debug spheres before destroying them (seconds)")]
    private float _debugSphereLifetime = 5.0f;

    [SerializeField]
    [Tooltip("Size of the debug spheres")]
    private float _sphereSize = 0.02f;

    [Header("Colors")]
    [SerializeField]
    private Color _redCubeColor = new Color(1f, 0f, 0f, 0.7f);

    [SerializeField]
    private Color _blueCubeColor = new Color(0f, 0.5f, 1f, 0.7f);

    [SerializeField]
    private Color _unknownCubeColor = new Color(0f, 1f, 0f, 0.7f);

    // Store latest detections for Gizmo drawing
    private List<DetectedCubeWithWorld> _latestDetections = new List<DetectedCubeWithWorld>();
    private readonly object _detectionsLock = new object();

    // Track created debug spheres
    private List<GameObject> _debugSpheres = new List<GameObject>();

    private void Start()
    {
        // Subscribe to detection events
        if (DetectionResultsReceiver.Instance != null)
        {
            DetectionResultsReceiver.Instance.OnDetectionWithWorldReceived +=
                HandleDetectionResult;
            Debug.Log("[CubeDetectionVisualizer] Subscribed to detection events");
        }
        else
        {
            Debug.LogWarning(
                "[CubeDetectionVisualizer] DetectionResultsReceiver.Instance is null. Make sure it exists in the scene."
            );
        }
    }

    private void OnDestroy()
    {
        // Unsubscribe from events
        if (DetectionResultsReceiver.Instance != null)
        {
            DetectionResultsReceiver.Instance.OnDetectionWithWorldReceived -=
                HandleDetectionResult;
        }

        // Clean up debug spheres
        ClearDebugSpheres();
    }

    /// <summary>
    /// Handles detection results from the receiver
    /// </summary>
    private void HandleDetectionResult(DetectionResultWithWorld result)
    {
        if (result == null || result.CubesWithWorldCoords == null)
            return;

        Debug.Log(
            $"[CubeDetectionVisualizer] Received {result.CubesWithWorldCoords.Length} detection(s)"
        );

        // Update stored detections for Gizmo drawing
        lock (_detectionsLock)
        {
            _latestDetections.Clear();
            _latestDetections.AddRange(result.CubesWithWorldCoords);
        }

        // Create debug spheres if enabled
        if (_createDebugSpheres)
        {
            CreateDebugSpheres(result.CubesWithWorldCoords);
        }
    }

    /// <summary>
    /// Creates temporary debug sphere GameObjects at detected positions
    /// </summary>
    private void CreateDebugSpheres(DetectedCubeWithWorld[] cubes)
    {
        // Clean up old spheres first
        ClearDebugSpheres();

        foreach (var cube in cubes)
        {
            if (!cube.HasWorldPosition)
                continue;

            // Create sphere
            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = $"DEBUG_DetectedCube_{cube.OriginalDetection.color}_{cube.OriginalDetection.id}";
            sphere.transform.position = cube.WorldPosition;
            sphere.transform.localScale = Vector3.one * _sphereSize;

            // Set color based on cube color
            Color color = GetColorForCube(cube.OriginalDetection.color);
            Renderer renderer = sphere.GetComponent<Renderer>();
            if (renderer != null)
            {
                Material mat = new Material(Shader.Find("Standard"));
                mat.color = color;
                mat.SetFloat("_Metallic", 0.5f);
                mat.SetFloat("_Glossiness", 0.8f);
                renderer.material = mat;
            }

            // Remove collider to avoid interfering with raycasts
            Collider collider = sphere.GetComponent<Collider>();
            if (collider != null)
            {
                Destroy(collider);
            }

            // Store reference
            _debugSpheres.Add(sphere);

            // Destroy after lifetime
            Destroy(sphere, _debugSphereLifetime);

            Debug.Log(
                $"[CubeDetectionVisualizer] Created debug sphere for {cube.OriginalDetection.color} cube at {cube.WorldPosition}"
            );
        }
    }

    /// <summary>
    /// Clears all debug spheres
    /// </summary>
    private void ClearDebugSpheres()
    {
        foreach (GameObject sphere in _debugSpheres)
        {
            if (sphere != null)
            {
                Destroy(sphere);
            }
        }
        _debugSpheres.Clear();
    }

    /// <summary>
    /// Gets the visualization color for a cube based on its detected color
    /// </summary>
    private Color GetColorForCube(string cubeColor)
    {
        return cubeColor?.ToLower() switch
        {
            "red" => _redCubeColor,
            "blue" => _blueCubeColor,
            _ => _unknownCubeColor,
        };
    }

    /// <summary>
    /// Draws Gizmos in the scene view
    /// </summary>
    private void OnDrawGizmos()
    {
        if (!_showGizmos)
            return;

        lock (_detectionsLock)
        {
            foreach (var cube in _latestDetections)
            {
                if (!cube.HasWorldPosition)
                    continue;

                // Set Gizmo color
                Gizmos.color = GetColorForCube(cube.OriginalDetection.color);

                // Draw sphere at detection position
                Gizmos.DrawSphere(cube.WorldPosition, _sphereSize);

                // Draw wire sphere for better visibility
                Gizmos.DrawWireSphere(cube.WorldPosition, _sphereSize * 1.2f);

                // Draw line from camera to detection if available
                if (cube.HitObject != null)
                {
                    // Draw a small cube to show the hit point
                    Gizmos.DrawCube(cube.WorldPosition, Vector3.one * _sphereSize * 0.5f);
                }
            }
        }
    }

    /// <summary>
    /// Public method to manually trigger visualization update
    /// </summary>
    public void UpdateVisualization(DetectionResultWithWorld result)
    {
        HandleDetectionResult(result);
    }

    /// <summary>
    /// Clear all visualizations
    /// </summary>
    public void ClearVisualizations()
    {
        lock (_detectionsLock)
        {
            _latestDetections.Clear();
        }
        ClearDebugSpheres();
    }

#if UNITY_EDITOR
    [UnityEditor.MenuItem("GameObject/ACRL/Cube Detection Visualizer", false, 10)]
    static void CreateVisualizer(UnityEditor.MenuCommand menuCommand)
    {
        // Create a new GameObject with the visualizer component
        GameObject go = new GameObject("CubeDetectionVisualizer");
        go.AddComponent<CubeDetectionVisualizer>();

        // Register creation undo
        UnityEditor.Undo.RegisterCreatedObjectUndo(go, "Create Cube Detection Visualizer");
        UnityEditor.Selection.activeObject = go;
    }
#endif
}
