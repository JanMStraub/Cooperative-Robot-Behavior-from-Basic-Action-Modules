using System.Collections;
using System.Collections.Generic;
using System.IO;
using Robotics;
using UnityEngine;

/// <summary>
/// Captures screenshots of cubes from various angles by rotating camera around them
/// Automatically generates YOLO format labels
/// </summary>
public class SaveScreenshots : MonoBehaviour
{
    public int numScreenshots = 10;
    public string outputDir = "YOLODataset/images";
    public string labelsDir = "YOLODataset/labels";
    public GameObject[] objectsToLabel; // Assign all objects to label (cubes, robot parts, etc.)
    public RobotController[] robots; // Assign robot controllers for joint randomization

    [Header("Dataset Split Settings")]
    [Range(0f, 1f)]
    public float trainSplit = 0.8f; // 80% for training

    [Range(0f, 1f)]
    public float testSplit = 0.1f; // 10% for testing

    [Range(0f, 1f)]
    public float validSplit = 0.1f; // 10% for validation

    [Header("Camera Orbit Settings")]
    public float orbitRadius = 0.8f; // Distance from center point
    public float minElevation = 15f; // Minimum camera elevation angle
    public float maxElevation = 45f; // Maximum camera elevation angle
    public Vector3 orbitCenter = new Vector3(0f, 0.3f, 0f); // Center point to orbit around

    [Header("Random Angle Variation")]
    public float pitchVariation = 5f; // Random pitch offset
    public float yawVariation = 5f; // Random yaw offset

    [Header("Capture Settings")]
    public int captureWidth = 640; // Image width (YOLO standard)
    public int captureHeight = 640; // Image height (YOLO standard)

    [Range(0, 100)]
    public int jpegQuality = 85; // JPEG compression quality

    [Header("YOLO Label Settings")]
    public bool generateLabels = true; // Auto-generate YOLO labels
    public string[] classNames; // Class names for mapping (e.g., "red", "blue", "green")

    [Range(0f, 1f)]
    [Tooltip(
        "Minimum visible area fraction (0-1) for object to be labeled. 0 = any visibility, 1 = fully visible only"
    )]
    public float minVisibleAreaThreshold = 0.1f;

    [Range(0f, 0.1f)]
    [Tooltip(
        "Minimum bbox width/height in viewport space (0-1). Filters out tiny slivers at edges. Default 0.02 = 2% of image"
    )]
    public float minBBoxSize = 0.02f;

    [Tooltip("Require object center to be in viewport for labeling")]
    public bool requireCenterInViewport = true;

    [Tooltip("Enable occlusion detection - skip objects hidden behind others")]
    public bool enableOcclusionDetection = true;

    [Range(0f, 1f)]
    [Tooltip(
        "Minimum fraction of sampled points that must be visible (0 = any point visible, 1 = all points visible)"
    )]
    public float occlusionVisibilityThreshold = 0.3f;

    private int captureCount = 0;
    private Dictionary<string, int> classNameToId;
    public Camera cam;
    private RenderTexture renderTexture;
    private Texture2D screenShot;

    void Start()
    {
        // Create train/test/valid subdirectories for images
        Directory.CreateDirectory(Path.Combine(outputDir, "train"));
        Directory.CreateDirectory(Path.Combine(outputDir, "test"));
        Directory.CreateDirectory(Path.Combine(outputDir, "valid"));

        if (generateLabels)
        {
            // Create train/test/valid subdirectories for labels
            Directory.CreateDirectory(Path.Combine(labelsDir, "train"));
            Directory.CreateDirectory(Path.Combine(labelsDir, "test"));
            Directory.CreateDirectory(Path.Combine(labelsDir, "valid"));

            InitializeClassMapping();
            CreateClassesFile();
        }

        // Validate split ratios
        float totalSplit = trainSplit + testSplit + validSplit;
        if (Mathf.Abs(totalSplit - 1f) > 0.01f)
        {
            Debug.LogWarning(
                $"Split ratios don't sum to 1.0 (current: {totalSplit:F2}). Normalizing..."
            );
            trainSplit /= totalSplit;
            testSplit /= totalSplit;
            validSplit /= totalSplit;
        }

        // Get camera component
        if (cam == null)
        {
            Debug.LogError("SaveScreenshots: No Camera component found!");
            enabled = false;
            return;
        }

        // Create RenderTexture for fixed resolution capture
        renderTexture = new RenderTexture(captureWidth, captureHeight, 24);
        screenShot = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);

        InvokeRepeating("CaptureRandomScene", 1f, 0.2f);
    }

    /// <summary>
    /// Initializes the class name to ID mapping
    /// </summary>
    void InitializeClassMapping()
    {
        classNameToId = new Dictionary<string, int>();

        if (classNames != null && classNames.Length > 0)
        {
            // Use provided class names
            for (int i = 0; i < classNames.Length; i++)
            {
                classNameToId[classNames[i].ToLower()] = i;
            }
        }
        else
        {
            // Auto-detect class names from labeled objects
            HashSet<string> uniqueNames = new HashSet<string>();
            foreach (var obj in objectsToLabel)
            {
                if (obj != null)
                {
                    string className = ExtractClassName(obj.name);
                    uniqueNames.Add(className);
                }
            }

            classNames = new string[uniqueNames.Count];
            int index = 0;
            foreach (var name in uniqueNames)
            {
                classNames[index] = name;
                classNameToId[name.ToLower()] = index;
                index++;
            }
        }

        Debug.Log($"Class mapping initialized: {string.Join(", ", classNames)}");
    }

    /// <summary>
    /// Creates classes.txt file for YOLO training
    /// </summary>
    void CreateClassesFile()
    {
        string classesFile = Path.Combine(Path.GetDirectoryName(labelsDir), "classes.txt");
        File.WriteAllLines(classesFile, classNames);
        Debug.Log($"Created classes file at: {classesFile}");
    }

    /// <summary>
    /// Extracts class name from GameObject name (e.g., "RedCube" -> "red", "BlueCube_01" -> "blue")
    /// </summary>
    string ExtractClassName(string objectName)
    {
        // Remove numbers and underscores, get the color/type part
        string cleaned = objectName.ToLower();

        // IMPORTANT: Check more specific patterns FIRST to avoid substring conflicts
        // (e.g., "gripperbase" must be checked before "base")

        // Gripper parts (most specific)
        if (cleaned.Contains("gripperbase"))
            return "gripperbase";
        if (cleaned.Contains("gripperjoint"))
            return "gripperjoint";
        if (cleaned.Contains("jawleft"))
            return "gripperjawleft";
        if (cleaned.Contains("jawright"))
            return "gripperjawright";

        // Wrist parts
        if (cleaned.Contains("wrist1"))
            return "wrist1";
        if (cleaned.Contains("wrist2"))
            return "wrist2";
        if (cleaned.Contains("wrist3"))
            return "wrist3";

        // Other robot parts
        if (cleaned.Contains("shoulder"))
            return "shoulder";
        if (cleaned.Contains("elbow"))
            return "elbow";
        if (cleaned.Contains("plate"))
            return "plate";
        if (cleaned.Contains("base"))
            return "base"; // After gripperbase check
        if (cleaned.Contains("joint"))
            return "joint"; // After gripperjoint check

        // Generic robot
        if (cleaned.Contains("robot"))
            return "robot";

        // Field objects
        if (cleaned.Contains("fielda"))
            return "fielda";
        if (cleaned.Contains("fieldb"))
            return "fieldb";
        if (cleaned.Contains("fieldc"))
            return "fieldc";

        // Cube colors (check last to avoid conflicts with robot parts)
        if (cleaned.Contains("red"))
            return "red";
        if (cleaned.Contains("blue"))
            return "blue";

        // Fallback: return first word
        string[] parts = objectName.Split('_', ' ');
        return parts[0].ToLower();
    }

    /// <summary>
    /// Gets class ID from GameObject name
    /// </summary>
    int GetClassId(string objectName)
    {
        string className = ExtractClassName(objectName);
        if (classNameToId.TryGetValue(className, out int classId))
        {
            return classId;
        }
        return 0; // Default to class 0 if not found
    }

    void OnDestroy()
    {
        // Clean up resources
        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
        }
        if (screenShot != null)
        {
            Destroy(screenShot);
        }
    }

    /// <summary>
    /// Captures a screenshot with randomized cube positions and camera angle
    /// </summary>
    void CaptureRandomScene()
    {
        if (captureCount >= numScreenshots)
        {
            CancelInvoke();
            Debug.Log($"Captured {captureCount} screenshots");
            return;
        }

        // Randomize only Target-tagged object positions
        foreach (var obj in objectsToLabel)
        {
            if (obj.CompareTag("Target"))
            {
                obj.transform.position = new Vector3(
                    Random.Range(-0.35f, 0.35f),
                    Random.Range(0.0f, 0.35f),
                    Random.Range(-0.35f, 0.35f)
                );
                obj.transform.rotation = Random.rotation;
            }
        }

        // Randomize robot joint targets
        RandomizeRobotJointTargets();

        // Position camera in orbit around cubes
        PositionCameraInOrbit();

        // Add slight random angle variation
        transform.Rotate(
            Random.Range(-pitchVariation, pitchVariation), // Pitch
            Random.Range(-yawVariation, yawVariation), // Yaw
            0f,
            Space.Self
        );

        CaptureToFile();

        captureCount++;
    }

    /// <summary>
    /// Waits for specified delay then captures screenshot
    /// </summary>
    IEnumerator CaptureAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);

        // Capture screenshot at fixed resolution
        CaptureToFile();

        captureCount++;
    }

    /// <summary>
    /// Determines which dataset split (train/test/valid) to use for the current capture
    /// </summary>
    string GetDatasetSplit()
    {
        float random = Random.value;

        if (random < trainSplit)
            return "train";
        else if (random < trainSplit + testSplit)
            return "test";
        else
            return "valid";
    }

    /// <summary>
    /// Captures camera view to file at specified resolution
    /// </summary>
    void CaptureToFile()
    {
        // Set camera to render to our RenderTexture
        RenderTexture currentRT = RenderTexture.active;
        cam.targetTexture = renderTexture;

        // Render the camera
        cam.Render();

        // Determine which dataset split to use
        string split = GetDatasetSplit();

        // Generate YOLO labels WHILE camera is targeting RenderTexture
        // This ensures WorldToViewportPoint uses correct projection
        string imageFilename = $"image_{captureCount:D4}.jpg";
        if (generateLabels)
        {
            GenerateYOLOLabels(imageFilename, split);
        }

        // Read pixels from RenderTexture
        RenderTexture.active = renderTexture;
        screenShot.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        screenShot.Apply();

        // Restore camera settings
        cam.targetTexture = null;
        RenderTexture.active = currentRT;

        // Encode to JPEG and save
        byte[] bytes = screenShot.EncodeToJPG(jpegQuality);
        string imagePath = Path.Combine(outputDir, split, imageFilename);
        File.WriteAllBytes(imagePath, bytes);

        Debug.Log($"Saved {imageFilename} to {split} set");
    }

    /// <summary>
    /// Generates YOLO format label file for current scene with proper visibility filtering
    /// </summary>
    void GenerateYOLOLabels(string imageFilename, string split)
    {
        List<string> labels = new List<string>();

        foreach (var obj in objectsToLabel)
        {
            if (obj == null || !obj.activeInHierarchy)
                continue;

            // Get world-space points (mesh vertices for tight bbox on rotated objects)
            Vector3[] worldPoints = GetObjectWorldPoints(obj);
            if (worldPoints.Length == 0)
                continue;

            // Check occlusion - skip if object is hidden behind other objects
            if (enableOcclusionDetection && IsObjectOccluded(obj, worldPoints))
                continue;

            // Calculate 2D bounding box in Unity viewport space (bottom-left origin)
            Vector2 min = new Vector2(float.MaxValue, float.MaxValue);
            Vector2 max = new Vector2(float.MinValue, float.MinValue);

            bool anyBehindCamera = false;
            int cornersInViewport = 0;

            foreach (var worldPoint in worldPoints)
            {
                // WorldToViewportPoint now uses RenderTexture projection (camera.targetTexture is set)
                Vector3 viewportPoint = cam.WorldToViewportPoint(worldPoint);

                // Check if any point is behind camera
                if (viewportPoint.z < 0)
                {
                    anyBehindCamera = true;
                    break;
                }

                // Count points actually inside viewport [0,1]
                if (
                    viewportPoint.x >= 0f
                    && viewportPoint.x <= 1f
                    && viewportPoint.y >= 0f
                    && viewportPoint.y <= 1f
                )
                {
                    cornersInViewport++;
                }

                // Accumulate min/max for tight 2D bounding box (before clipping)
                min.x = Mathf.Min(min.x, viewportPoint.x);
                min.y = Mathf.Min(min.y, viewportPoint.y);
                max.x = Mathf.Max(max.x, viewportPoint.x);
                max.y = Mathf.Max(max.y, viewportPoint.y);
            }

            // Skip if any point is behind camera (object spans camera near plane)
            if (anyBehindCamera)
                continue;

            // Skip if all points are completely outside viewport on the same side
            // This filters objects that are entirely off-screen but would create slivers after clipping
            if (cornersInViewport == 0)
            {
                // Check if all points are on the same side (completely outside)
                bool allLeft = max.x < 0f;
                bool allRight = min.x > 1f;
                bool allBelow = max.y < 0f;
                bool allAbove = min.y > 1f;

                if (allLeft || allRight || allBelow || allAbove)
                    continue;
            }

            // Calculate unclipped area for visibility threshold
            float unclippedWidth = max.x - min.x;
            float unclippedHeight = max.y - min.y;
            float unclippedArea = unclippedWidth * unclippedHeight;

            // Clip bounding box to viewport bounds [0,1]
            min.x = Mathf.Clamp(min.x, 0f, 1f);
            min.y = Mathf.Clamp(min.y, 0f, 1f);
            max.x = Mathf.Clamp(max.x, 0f, 1f);
            max.y = Mathf.Clamp(max.y, 0f, 1f);

            // Calculate clipped dimensions in Unity viewport space
            float width_unity = max.x - min.x;
            float height_unity = max.y - min.y;

            // Skip if bbox has no area after clipping
            if (width_unity <= 0.001f || height_unity <= 0.001f)
                continue;

            // Filter out tiny slivers (absolute size check)
            if (width_unity < minBBoxSize || height_unity < minBBoxSize)
                continue;

            // Calculate visible area fraction
            float clippedArea = width_unity * height_unity;
            float visibleFraction = (unclippedArea > 0) ? (clippedArea / unclippedArea) : 0f;

            // Apply visibility threshold filter (relative size check)
            if (visibleFraction < minVisibleAreaThreshold)
                continue;

            // Calculate center in Unity viewport space (bottom-left origin)
            float x_center_unity = (min.x + max.x) / 2f;
            float y_center_unity = (min.y + max.y) / 2f;

            // Optional: require center to be in viewport
            if (requireCenterInViewport)
            {
                if (
                    x_center_unity < 0f
                    || x_center_unity > 1f
                    || y_center_unity < 0f
                    || y_center_unity > 1f
                )
                    continue;
            }

            // Transform to YOLO coordinate system (top-left origin)
            // YOLO uses top-left origin, Unity viewport uses bottom-left
            // Flip Y-axis: y_yolo = 1 - y_unity
            float x_center_yolo = x_center_unity;
            float y_center_yolo = 1f - y_center_unity;
            float width_yolo = width_unity;
            float height_yolo = height_unity;

            // Get class ID based on object name
            int classId = GetClassId(obj.name);
            string className = ExtractClassName(obj.name);

            // Debug: log what we're detecting
            Debug.Log(
                $"Labeled {className} object '{obj.name}': center=({x_center_yolo:F3}, {y_center_yolo:F3}), size=({width_yolo:F3}, {height_yolo:F3}), visible={visibleFraction:F2}"
            );

            // Format: class_id x_center y_center width height
            string label =
                $"{classId} {x_center_yolo:F6} {y_center_yolo:F6} {width_yolo:F6} {height_yolo:F6}";
            labels.Add(label);
        }

        // Save label file in the appropriate split directory
        if (labels.Count > 0)
        {
            string labelFilename = Path.GetFileNameWithoutExtension(imageFilename) + ".txt";
            string labelPath = Path.Combine(labelsDir, split, labelFilename);
            File.WriteAllLines(labelPath, labels.ToArray());
        }
    }

    /// <summary>
    /// Gets the combined bounds of all renderers in an object
    /// </summary>
    Bounds GetObjectBounds(GameObject obj)
    {
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
            return new Bounds(obj.transform.position, Vector3.zero);

        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }
        return bounds;
    }

    /// <summary>
    /// Gets world-space points for tight bounding box calculation
    /// Uses only the primary mesh renderer to avoid including unintended child objects
    /// </summary>
    Vector3[] GetObjectWorldPoints(GameObject obj)
    {
        // Try to get the primary MeshFilter (not children)
        MeshFilter meshFilter = obj.GetComponent<MeshFilter>();

        if (meshFilter != null && meshFilter.sharedMesh != null)
        {
            Mesh mesh = meshFilter.sharedMesh;
            Transform transform = meshFilter.transform;

            // Get all vertices for accurate tight bbox (cubes typically have 24 vertices)
            Vector3[] vertices = mesh.vertices;
            List<Vector3> worldPoints = new List<Vector3>(vertices.Length);

            for (int i = 0; i < vertices.Length; i++)
            {
                // Transform vertex to world space
                Vector3 worldPoint = transform.TransformPoint(vertices[i]);
                worldPoints.Add(worldPoint);
            }

            return worldPoints.ToArray();
        }

        // Fallback: Try to get first child MeshFilter if no direct component
        MeshFilter[] childFilters = obj.GetComponentsInChildren<MeshFilter>();
        if (childFilters.Length > 0 && childFilters[0].sharedMesh != null)
        {
            MeshFilter firstFilter = childFilters[0];
            Mesh mesh = firstFilter.sharedMesh;
            Transform transform = firstFilter.transform;

            Vector3[] vertices = mesh.vertices;
            List<Vector3> worldPoints = new List<Vector3>(vertices.Length);

            for (int i = 0; i < vertices.Length; i++)
            {
                Vector3 worldPoint = transform.TransformPoint(vertices[i]);
                worldPoints.Add(worldPoint);
            }

            return worldPoints.ToArray();
        }

        // Last fallback: use renderer bounds corners
        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer == null)
            renderer = obj.GetComponentInChildren<Renderer>();

        if (renderer != null)
        {
            return GetBoundsCorners(renderer.bounds);
        }

        // No mesh or renderer found, return empty
        return new Vector3[0];
    }

    /// <summary>
    /// Gets the 8 corners of a bounding box
    /// </summary>
    Vector3[] GetBoundsCorners(Bounds bounds)
    {
        Vector3 center = bounds.center;
        Vector3 extents = bounds.extents;

        return new Vector3[]
        {
            center + new Vector3(-extents.x, -extents.y, -extents.z),
            center + new Vector3(-extents.x, -extents.y, extents.z),
            center + new Vector3(-extents.x, extents.y, -extents.z),
            center + new Vector3(-extents.x, extents.y, extents.z),
            center + new Vector3(extents.x, -extents.y, -extents.z),
            center + new Vector3(extents.x, -extents.y, extents.z),
            center + new Vector3(extents.x, extents.y, -extents.z),
            center + new Vector3(extents.x, extents.y, extents.z),
        };
    }

    /// <summary>
    /// Checks if an object is occluded (hidden behind other objects) from the camera
    /// </summary>
    bool IsObjectOccluded(GameObject obj, Vector3[] worldPoints)
    {
        if (worldPoints.Length == 0)
            return true;

        // Sample a subset of points for performance
        int sampleCount = Mathf.Min(12, worldPoints.Length);
        int step = Mathf.Max(1, worldPoints.Length / sampleCount);

        int visiblePoints = 0;
        int totalSampled = 0;

        for (int i = 0; i < worldPoints.Length; i += step)
        {
            if (totalSampled >= sampleCount)
                break;

            Vector3 worldPoint = worldPoints[i];
            Vector3 directionToCamera = cam.transform.position - worldPoint;
            float distanceToCamera = directionToCamera.magnitude;

            // Raycast from point towards camera
            if (
                Physics.Raycast(
                    worldPoint,
                    directionToCamera.normalized,
                    out RaycastHit hit,
                    distanceToCamera
                )
            )
            {
                // Check if the hit object is part of our target object or its children
                if (
                    hit.collider.gameObject == obj
                    || hit.collider.transform.IsChildOf(obj.transform)
                )
                {
                    visiblePoints++;
                }
                // Also check if we hit a parent of our object (robot arm with child parts)
                else if (obj.transform.IsChildOf(hit.collider.transform))
                {
                    visiblePoints++;
                }
            }
            else
            {
                // No hit means direct line of sight to camera
                visiblePoints++;
            }

            totalSampled++;
        }

        // Calculate visibility ratio
        float visibilityRatio = totalSampled > 0 ? (float)visiblePoints / totalSampled : 0f;

        // Object is occluded if visibility is below threshold
        return visibilityRatio < occlusionVisibilityThreshold;
    }

    /// <summary>
    /// Randomizes all robot joint targets within their safe limits
    /// Uses middle 50% of range for more natural/stable poses
    /// </summary>
    void RandomizeRobotJointTargets()
    {
        if (robots == null || robots.Length == 0)
            return;

        foreach (var robotController in robots)
        {
            if (robotController == null || robotController.robotJoints == null)
                continue;

            int jointCount = robotController.robotJoints.Length;

            for (int i = 0; i < jointCount; i++)
            {
                var joint = robotController.robotJoints[i];
                var drive = joint.xDrive;

                // Get safe range from joint limits (in degrees)
                float lowerLimit = drive.lowerLimit;
                float upperLimit = drive.upperLimit;

                // Calculate middle 50% of the range for more natural poses
                float center = (lowerLimit + upperLimit) / 2f;
                float fullRange = upperLimit - lowerLimit;
                float halfMiddleRange = fullRange * 0.25f; // 50% / 2 = 25% on each side

                float restrictedMin = center - halfMiddleRange;
                float restrictedMax = center + halfMiddleRange;

                // Randomize within restricted range
                float randomTarget = Random.Range(restrictedMin, restrictedMax);

                // Apply the new target
                drive.target = randomTarget;
                joint.xDrive = drive;
            }
        }
    }

    /// <summary>
    /// Positions camera in a random orbital position around the center point
    /// </summary>
    void PositionCameraInOrbit()
    {
        // Random azimuth angle (0-360 degrees around Y axis)
        float azimuth = Random.Range(0f, 360f);

        // Random elevation angle (looking down at cubes)
        float elevation = Random.Range(minElevation, maxElevation);

        // Convert spherical coordinates to Cartesian
        float elevationRad = elevation * Mathf.Deg2Rad;
        float azimuthRad = azimuth * Mathf.Deg2Rad;

        Vector3 offset = new Vector3(
            orbitRadius * Mathf.Cos(elevationRad) * Mathf.Cos(azimuthRad),
            orbitRadius * Mathf.Sin(elevationRad),
            orbitRadius * Mathf.Cos(elevationRad) * Mathf.Sin(azimuthRad)
        );

        // Set camera position
        transform.position = orbitCenter + offset;

        // Make camera look at center point
        transform.LookAt(orbitCenter);
    }
}
