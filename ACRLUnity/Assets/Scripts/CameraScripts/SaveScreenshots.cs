using System.Collections;
using System.Collections.Generic;
using System.IO;
using Robotics;
using UnityEngine;

/// <summary>
/// Generates a YOLO training dataset by capturing screenshots from a fixed stereo camera position.
/// Randomizes scene contents (object positions, robot joint poses) between captures while keeping
/// the camera at its real mounting position with small jitter for augmentation.
/// </summary>
public class SaveScreenshots : MonoBehaviour
{
    public int numScreenshots = 1000;
    public string outputDir = "YOLODataset/images";
    public string labelsDir = "YOLODataset/labels";
    public GameObject[] objectsToLabel;
    public RobotController[] robots;

    [Header("Dataset Split Settings")]
    [Range(0f, 1f)]
    public float trainSplit = 0.8f;

    [Range(0f, 1f)]
    public float testSplit = 0.1f;

    [Range(0f, 1f)]
    public float validSplit = 0.1f;

    [Header("Camera Jitter (augmentation around fixed mount position)")]
    [Tooltip("Max random position offset in meters per axis")]
    public float positionJitter = 0.02f;

    [Tooltip("Max random rotation offset in degrees per axis")]
    public float rotationJitter = 2f;

    [Header("Object Randomization")]
    [Tooltip("Enable random repositioning of Target-tagged objects each capture. Disable to keep objects at their scene positions.")]
    public bool randomizeObjectPositions = true;

    [Tooltip("Min/max bounds for randomizing Target-tagged object positions (matches robot workspace)")]
    public Vector3 objectSpawnMin = new Vector3(-0.35f, 0.0f, -0.35f);
    public Vector3 objectSpawnMax = new Vector3(0.35f, 0.4f, 0.35f);

    [Tooltip("Bias cube spawning toward the far half of the table to generate more small/distant training examples")]
    public bool biasSpawnToFar = false;

    [Range(0f, 1f)]
    [Tooltip("When biasSpawnToFar is enabled, probability that a cube spawns in the far half (z > midpoint). 0.5 = uniform.")]
    public float farSpawnBias = 0.75f;

    [Header("Visibility Randomization")]
    [Tooltip("All cube GameObjects — a random subset will be shown each capture")]
    public GameObject[] allCubes;

    [Tooltip("All field GameObjects — a random subset will be shown each capture")]
    public GameObject[] allFields;

    [Range(0f, 1f)]
    [Tooltip("Probability that any individual cube is visible in a given capture")]
    public float cubeVisibilityChance = 0.6f;

    [Range(0f, 1f)]
    [Tooltip("Probability that any individual field is visible in a given capture")]
    public float fieldVisibilityChance = 0.7f;

    [Range(0f, 1f)]
    [Tooltip("Probability that any individual robot is visible in a given capture")]
    public float robotVisibilityChance = 0.8f;

    [Header("Capture Settings")]
    public int captureWidth = 1920;
    public int captureHeight = 1080;

    [Range(0, 100)]
    public int jpegQuality = 85;

    [Header("YOLO Label Settings")]
    public bool generateLabels = true;
    public string[] classNames;

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

    public Camera cam;

    private int captureCount = 0;
    private Dictionary<string, int> classNameToId;
    private Dictionary<int, (int classId, string className)> _objectClassCache;
    private RenderTexture renderTexture;
    private Texture2D screenShot;

    // Stored on Start() from the camera's actual scene transform
    private Vector3 baseCameraPosition;
    private Quaternion baseCameraRotation;

    void Start()
    {
        Directory.CreateDirectory(Path.Combine(outputDir, "train"));
        Directory.CreateDirectory(Path.Combine(outputDir, "test"));
        Directory.CreateDirectory(Path.Combine(outputDir, "valid"));

        if (generateLabels)
        {
            Directory.CreateDirectory(Path.Combine(labelsDir, "train"));
            Directory.CreateDirectory(Path.Combine(labelsDir, "test"));
            Directory.CreateDirectory(Path.Combine(labelsDir, "valid"));

            InitializeClassMapping();
            CreateClassesFile();

            _objectClassCache = new Dictionary<int, (int, string)>();
            foreach (var obj in objectsToLabel)
            {
                if (obj == null)
                    continue;
                string cn = ExtractClassName(obj.name);
                if (!classNameToId.TryGetValue(cn, out int cid))
                    Debug.LogError($"[SaveScreenshots] No class ID for '{cn}' (GameObject: '{obj.name}') — fix the name or classNames array!");
                _objectClassCache[obj.GetInstanceID()] = (cid, cn);
            }
        }

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

        if (cam == null)
        {
            Debug.LogError("SaveScreenshots: No Camera assigned!");
            enabled = false;
            return;
        }

        // Store the camera's real mounting position/rotation from the scene
        baseCameraPosition = cam.transform.position;
        baseCameraRotation = cam.transform.rotation;

        renderTexture = new RenderTexture(captureWidth, captureHeight, 24);
        screenShot = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);

        Debug.Log(
            $"SaveScreenshots: Starting capture of {numScreenshots} images. "
            + $"Camera base position: {baseCameraPosition}, rotation: {baseCameraRotation.eulerAngles}"
        );

        StartCoroutine(CaptureLoop());
    }

    /// <summary>
    /// Coroutine that captures screenshots, waiting for physics to settle between each capture.
    /// </summary>
    IEnumerator CaptureLoop()
    {
        yield return new WaitForSeconds(1f);

        while (captureCount < numScreenshots)
        {
            SetupRandomScene();

            // Wait for physics to settle after repositioning objects and joints
            for (int i = 0; i < 10; i++)
                yield return new WaitForFixedUpdate();

            CaptureToFile();
            captureCount++;

            yield return null;
        }

        Debug.Log($"Dataset generation complete: captured {captureCount} screenshots");
    }

    /// <summary>
    /// Initializes the class name to ID mapping.
    /// </summary>
    void InitializeClassMapping()
    {
        classNameToId = new Dictionary<string, int>();

        if (classNames != null && classNames.Length > 0)
        {
            for (int i = 0; i < classNames.Length; i++)
            {
                classNameToId[classNames[i].ToLower()] = i;
            }
        }
        else
        {
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
    /// Creates classes.txt file for YOLO training.
    /// </summary>
    void CreateClassesFile()
    {
        string classesFile = Path.Combine(Path.GetDirectoryName(labelsDir), "classes.txt");
        File.WriteAllLines(classesFile, classNames);
        Debug.Log($"Created classes file at: {classesFile}");
    }

    /// <summary>
    /// Extracts class name from GameObject name (e.g., "RedCube" -> "red_cube", "GripperBase_01" -> "gripperbase").
    /// </summary>
    string ExtractClassName(string objectName)
    {
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
            return "base";
        if (cleaned.Contains("joint"))
            return "joint";

        // Generic robot
        if (cleaned.Contains("robot"))
            return "robot";

        // Field objects
        if (cleaned.Contains("fielda"))
            return "field_a";
        if (cleaned.Contains("fieldb"))
            return "field_b";
        if (cleaned.Contains("fieldc"))
            return "field_c";
        if (cleaned.Contains("fieldd"))
            return "field_d";
        if (cleaned.Contains("fielde"))
            return "field_e";
        if (cleaned.Contains("fieldf"))
            return "field_f";
        if (cleaned.Contains("fieldg"))
            return "field_g";
        if (cleaned.Contains("fieldh"))
            return "field_h";
        if (cleaned.Contains("fieldi"))
            return "field_i";

        // Cube colors — check multi-word colors before short ones
        if (cleaned.Contains("magenta"))
            return "magenta_cube";
        if (cleaned.Contains("orange"))
            return "orange_cube";
        if (cleaned.Contains("purple"))
            return "purple_cube";
        if (cleaned.Contains("yellow"))
            return "yellow_cube";
        if (cleaned.Contains("green"))
            return "green_cube";
        if (cleaned.Contains("cyan"))
            return "cyan_cube";
        if (cleaned.Contains("blue"))
            return "blue_cube";
        if (cleaned.Contains("red"))
            return "red_cube";

        // Fallback: return first word
        string[] parts = objectName.Split('_', ' ');
        return parts[0].ToLower();
    }

    /// <summary>
    /// Gets class ID from GameObject name.
    /// </summary>
    int GetClassId(string objectName)
    {
        string className = ExtractClassName(objectName);
        if (classNameToId.TryGetValue(className, out int classId))
        {
            return classId;
        }
        return 0;
    }

    void OnDestroy()
    {
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
    /// Samples a Z position within spawn bounds, optionally biased toward the far half of the table
    /// to produce more small/distant cube examples in the training dataset.
    /// </summary>
    float SampleZWithBias()
    {
        if (!biasSpawnToFar)
            return Random.Range(objectSpawnMin.z, objectSpawnMax.z);

        float midZ = (objectSpawnMin.z + objectSpawnMax.z) / 2f;
        if (Random.value < farSpawnBias)
            return Random.Range(midZ, objectSpawnMax.z);
        else
            return Random.Range(objectSpawnMin.z, midZ);
    }

    /// <summary>
    /// Randomizes scene contents and applies small camera jitter around the fixed mount position.
    /// </summary>
    void SetupRandomScene()
    {
        // Randomize only Target-tagged object positions within table bounds
        if (randomizeObjectPositions)
        {
            foreach (var obj in objectsToLabel)
            {
                if (obj == null)
                    continue;
                if (obj.CompareTag("Target"))
                {
                    obj.transform.position = new Vector3(
                        Random.Range(objectSpawnMin.x, objectSpawnMax.x),
                        Random.Range(objectSpawnMin.y, objectSpawnMax.y),
                        SampleZWithBias()
                    );
                    obj.transform.rotation = Random.rotation;
                }
            }
        }

        // Randomize which cubes and fields are visible this capture
        RandomizeObjectVisibility();

        // Randomize robot joint targets
        RandomizeRobotJointTargets();

        // Apply small jitter around the real camera mount position
        ApplyCameraJitter();
    }

    /// <summary>
    /// Randomly shows or hides individual cubes and fields each capture so the model
    /// learns to detect each object independently, regardless of what else is in the scene.
    /// </summary>
    void RandomizeObjectVisibility()
    {
        if (allCubes != null)
        {
            foreach (var cube in allCubes)
            {
                if (cube != null)
                    cube.SetActive(Random.value < cubeVisibilityChance);
            }
        }

        if (allFields != null)
        {
            foreach (var field in allFields)
            {
                if (field != null)
                    field.SetActive(Random.value < fieldVisibilityChance);
            }
        }

        if (robots != null)
        {
            foreach (var robot in robots)
            {
                if (robot == null)
                    continue;
                bool visible = Random.value < robotVisibilityChance;
                foreach (var renderer in robot.GetComponentsInChildren<Renderer>())
                    renderer.enabled = visible;
            }
        }
    }

    /// <summary>
    /// Applies small random position and rotation offsets to the camera around its fixed mount pose.
    /// Simulates minor mounting tolerance and provides augmentation diversity.
    /// </summary>
    void ApplyCameraJitter()
    {
        Vector3 posOffset = new Vector3(
            Random.Range(-positionJitter, positionJitter),
            Random.Range(-positionJitter, positionJitter),
            Random.Range(-positionJitter, positionJitter)
        );
        cam.transform.position = baseCameraPosition + posOffset;

        cam.transform.rotation = baseCameraRotation;
        cam.transform.Rotate(
            Random.Range(-rotationJitter, rotationJitter),
            Random.Range(-rotationJitter, rotationJitter),
            0f,
            Space.Self
        );
    }

    /// <summary>
    /// Determines which dataset split (train/test/valid) to use for the current capture.
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
    /// Captures camera view to file at specified resolution.
    /// </summary>
    void CaptureToFile()
    {
        RenderTexture currentRT = RenderTexture.active;
        cam.targetTexture = renderTexture;

        cam.Render();

        string split = GetDatasetSplit();

        // Generate YOLO labels WHILE camera is targeting RenderTexture
        // so WorldToViewportPoint uses the correct projection
        string imageFilename = $"image_{captureCount:D4}.jpg";
        if (generateLabels)
        {
            GenerateYOLOLabels(imageFilename, split);
        }

        RenderTexture.active = renderTexture;
        screenShot.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        screenShot.Apply();

        cam.targetTexture = null;
        RenderTexture.active = currentRT;

        byte[] bytes = screenShot.EncodeToJPG(jpegQuality);
        string imagePath = Path.Combine(outputDir, split, imageFilename);
        File.WriteAllBytes(imagePath, bytes);

        if (captureCount % 100 == 0)
            Debug.Log($"Progress: {captureCount}/{numScreenshots} ({split})");
    }

    /// <summary>
    /// Generates YOLO format label file for current scene with proper visibility filtering.
    /// </summary>
    void GenerateYOLOLabels(string imageFilename, string split)
    {
        List<string> labels = new List<string>();

        foreach (var obj in objectsToLabel)
        {
            if (obj == null || !obj.activeInHierarchy)
                continue;

            Vector3[] worldPoints = GetObjectWorldPoints(obj);
            if (worldPoints.Length == 0)
                continue;

            if (enableOcclusionDetection && IsObjectOccluded(obj, worldPoints))
                continue;

            // Calculate 2D bounding box in Unity viewport space (bottom-left origin)
            Vector2 min = new Vector2(float.MaxValue, float.MaxValue);
            Vector2 max = new Vector2(float.MinValue, float.MinValue);

            bool anyBehindCamera = false;
            int cornersInViewport = 0;

            foreach (var worldPoint in worldPoints)
            {
                Vector3 viewportPoint = cam.WorldToViewportPoint(worldPoint);

                if (viewportPoint.z < 0)
                {
                    anyBehindCamera = true;
                    break;
                }

                if (
                    viewportPoint.x >= 0f
                    && viewportPoint.x <= 1f
                    && viewportPoint.y >= 0f
                    && viewportPoint.y <= 1f
                )
                {
                    cornersInViewport++;
                }

                min.x = Mathf.Min(min.x, viewportPoint.x);
                min.y = Mathf.Min(min.y, viewportPoint.y);
                max.x = Mathf.Max(max.x, viewportPoint.x);
                max.y = Mathf.Max(max.y, viewportPoint.y);
            }

            if (anyBehindCamera)
                continue;

            // Skip if all points are completely outside viewport on the same side
            if (cornersInViewport == 0)
            {
                bool allLeft = max.x < 0f;
                bool allRight = min.x > 1f;
                bool allBelow = max.y < 0f;
                bool allAbove = min.y > 1f;

                if (allLeft || allRight || allBelow || allAbove)
                    continue;
            }

            float unclippedWidth = max.x - min.x;
            float unclippedHeight = max.y - min.y;
            float unclippedArea = unclippedWidth * unclippedHeight;

            // Check unclipped center before clamping (post-clamp center is always in [0,1])
            if (requireCenterInViewport)
            {
                float unclippedCenterX = (min.x + max.x) / 2f;
                float unclippedCenterY = (min.y + max.y) / 2f;
                if (
                    unclippedCenterX < 0f
                    || unclippedCenterX > 1f
                    || unclippedCenterY < 0f
                    || unclippedCenterY > 1f
                )
                    continue;
            }

            // Clip bounding box to viewport bounds [0,1]
            min.x = Mathf.Clamp(min.x, 0f, 1f);
            min.y = Mathf.Clamp(min.y, 0f, 1f);
            max.x = Mathf.Clamp(max.x, 0f, 1f);
            max.y = Mathf.Clamp(max.y, 0f, 1f);

            float width_unity = max.x - min.x;
            float height_unity = max.y - min.y;

            if (width_unity <= 0.001f || height_unity <= 0.001f)
                continue;

            if (width_unity < minBBoxSize || height_unity < minBBoxSize)
                continue;

            float clippedArea = width_unity * height_unity;
            float visibleFraction = (unclippedArea > 0) ? (clippedArea / unclippedArea) : 0f;

            if (visibleFraction < minVisibleAreaThreshold)
                continue;

            // Calculate center in Unity viewport space (bottom-left origin, post-clip)
            float x_center_unity = (min.x + max.x) / 2f;
            float y_center_unity = (min.y + max.y) / 2f;

            // Transform to YOLO coordinate system (top-left origin): flip Y
            float x_center_yolo = x_center_unity;
            float y_center_yolo = 1f - y_center_unity;
            float width_yolo = width_unity;
            float height_yolo = height_unity;

            var (classId, className) = _objectClassCache.TryGetValue(obj.GetInstanceID(), out var info)
                ? info
                : (0, ExtractClassName(obj.name));

            string label =
                $"{classId} {x_center_yolo:F6} {y_center_yolo:F6} {width_yolo:F6} {height_yolo:F6}";
            labels.Add(label);
        }

        // Always write label file (empty file = explicit negative sample for YOLO)
        string labelFilename = Path.GetFileNameWithoutExtension(imageFilename) + ".txt";
        string labelPath = Path.Combine(labelsDir, split, labelFilename);
        File.WriteAllLines(labelPath, labels.ToArray());
    }

    /// <summary>
    /// Gets world-space points for bounding box calculation using the 8 local AABB corners.
    /// </summary>
    Vector3[] GetObjectWorldPoints(GameObject obj)
    {
        MeshFilter meshFilter = obj.GetComponent<MeshFilter>();

        if (meshFilter != null && meshFilter.sharedMesh != null)
        {
            return GetLocalBoundsCorners(meshFilter);
        }

        MeshFilter[] childFilters = obj.GetComponentsInChildren<MeshFilter>();
        if (childFilters.Length > 0 && childFilters[0].sharedMesh != null)
        {
            return GetLocalBoundsCorners(childFilters[0]);
        }

        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer == null)
            renderer = obj.GetComponentInChildren<Renderer>();

        if (renderer != null)
        {
            return GetBoundsCorners(renderer.bounds);
        }

        return new Vector3[0];
    }

    /// <summary>
    /// Returns the 8 world-space corners of a mesh's local AABB.
    /// </summary>
    Vector3[] GetLocalBoundsCorners(MeshFilter meshFilter)
    {
        Bounds localBounds = meshFilter.sharedMesh.bounds;
        Transform t = meshFilter.transform;
        Vector3 c = localBounds.center;
        Vector3 e = localBounds.extents;

        return new Vector3[]
        {
            t.TransformPoint(c + new Vector3(-e.x, -e.y, -e.z)),
            t.TransformPoint(c + new Vector3(-e.x, -e.y,  e.z)),
            t.TransformPoint(c + new Vector3(-e.x,  e.y, -e.z)),
            t.TransformPoint(c + new Vector3(-e.x,  e.y,  e.z)),
            t.TransformPoint(c + new Vector3( e.x, -e.y, -e.z)),
            t.TransformPoint(c + new Vector3( e.x, -e.y,  e.z)),
            t.TransformPoint(c + new Vector3( e.x,  e.y, -e.z)),
            t.TransformPoint(c + new Vector3( e.x,  e.y,  e.z)),
        };
    }

    /// <summary>
    /// Gets the 8 corners of a bounding box.
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
    /// Checks if an object is occluded (hidden behind other objects) from the camera.
    /// </summary>
    bool IsObjectOccluded(GameObject obj, Vector3[] worldPoints)
    {
        if (worldPoints.Length == 0)
            return true;

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

            if (
                Physics.Raycast(
                    worldPoint,
                    directionToCamera.normalized,
                    out RaycastHit hit,
                    distanceToCamera
                )
            )
            {
                if (
                    hit.collider.gameObject == obj
                    || hit.collider.transform.IsChildOf(obj.transform)
                )
                {
                    visiblePoints++;
                }
                else if (obj.transform.IsChildOf(hit.collider.transform))
                {
                    visiblePoints++;
                }
            }
            else
            {
                visiblePoints++;
            }

            totalSampled++;
        }

        float visibilityRatio = totalSampled > 0 ? (float)visiblePoints / totalSampled : 0f;

        return visibilityRatio < occlusionVisibilityThreshold;
    }

    /// <summary>
    /// Randomizes all robot joint targets within the middle 50% of their safe limits.
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

                float lowerLimit = drive.lowerLimit;
                float upperLimit = drive.upperLimit;

                // Middle 30% of range for subtle poses
                float center = (lowerLimit + upperLimit) / 2f;
                float fullRange = upperLimit - lowerLimit;
                float halfMiddleRange = fullRange * 0.20f;

                float restrictedMin = center - halfMiddleRange;
                float restrictedMax = center + halfMiddleRange;

                drive.target = Random.Range(restrictedMin, restrictedMax);
                joint.xDrive = drive;
            }
        }
    }
}
