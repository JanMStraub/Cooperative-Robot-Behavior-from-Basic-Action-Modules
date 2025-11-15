using System;
using UnityEngine;

namespace PythonCommunication
{
    // ============================================================================
    // LLM RESULT DATA MODELS (from LLMResultsReceiver)
    // ============================================================================

    /// <summary>
    /// Data structure for LLM analysis results received from Python
    /// </summary>
    [Serializable]
    public class LLMResult
    {
        public bool success;
        public string response;
        public string camera_id;
        public string timestamp;
        public LLMMetadata metadata;
    }

    [Serializable]
    public class LLMMetadata
    {
        public string model;
        public float duration_seconds;
        public int image_count;
        public string[] camera_ids;
        public string prompt;
        public string full_prompt;
    }

    // ============================================================================
    // DEPTH RESULT DATA MODELS (from DepthResultsReceiver)
    // ============================================================================

    /// <summary>
    /// Data structure for stereo detection results with 3D depth information
    /// </summary>
    [Serializable]
    public class DepthResult
    {
        public bool success;
        public string camera_id;
        public string timestamp;
        public ObjectDetection[] detections;
        public DepthMetadata metadata;
    }

    [Serializable]
    public class ObjectDetection
    {
        public string color;
        public float confidence;
        public DetectionBoundingBox bbox;
        public DetectionPixelCoords pixel_center;
        public Detection3DPosition world_position;
        public float depth_m;
        public float disparity;
    }

    [Serializable]
    public class DetectionBoundingBox
    {
        public int x;
        public int y;
        public int width;
        public int height;
    }

    [Serializable]
    public class DetectionPixelCoords
    {
        public int x;
        public int y;
    }

    [Serializable]
    public class Detection3DPosition
    {
        public float x;
        public float y;
        public float z;
    }

    [Serializable]
    public class DepthMetadata
    {
        public float processing_time_seconds;
        public string prompt;
        public float camera_baseline_m;
        public float camera_fov_deg;
        public string detection_mode;
    }

    // ============================================================================
    // DETECTION RESULT DATA MODELS (original DetectionDataModels)
    // ============================================================================

    /// <summary>
    /// Complete detection result from Python detector including all detected cubes in a single frame
    /// </summary>
    [Serializable]
    public class DetectionResult
    {
        public bool success;
        public string camera_id;
        public string timestamp;
        public int image_width;
        public int image_height;
        public DetectionObject[] detections;
        public DetectionMetadata metadata;

        private const string _logPrefix = "[DETECTION_RESULT]";

        public override string ToString()
        {
            int count = detections?.Length ?? 0;
            return $"{_logPrefix} [camera={camera_id}, detections={count}]";
        }
    }

    /// <summary>
    /// Single detected object (cube) with pixel coordinates and color.
    /// May optionally include 3D world position from stereo depth estimation.
    /// </summary>
    [Serializable]
    public class DetectionObject
    {
        public int id;
        public string color; // "red" or "blue"
        public BoundingBoxPx bbox_px;
        public Vector2Int center_px;
        public float confidence;
        public WorldPosition world_position; // Optional, from stereo depth estimation

        private const string _logPrefix = "[DETECTION_OBJECT]";

        /// <summary>
        /// Converts pixel coordinates to Unity world coordinates using raycasting
        /// </summary>
        /// <param name="camera">The camera that captured this detection</param>
        /// <param name="imageWidth">Width of the captured image</param>
        /// <param name="imageHeight">Height of the captured image</param>
        /// <param name="worldPosition">Output world position (if hit)</param>
        /// <param name="hitObject">Output GameObject that was hit (if any)</param>
        /// <returns>True if a world position was found via raycast</returns>
        public bool TryGetWorldPosition(
            Camera camera,
            int imageWidth,
            int imageHeight,
            out Vector3 worldPosition,
            out GameObject hitObject
        )
        {
            worldPosition = Vector3.zero;
            hitObject = null;

            if (camera == null)
            {
                Debug.LogWarning($"{_logPrefix} Camera is null");
                return false;
            }

            // Convert pixel coordinates to viewport coordinates (0-1 normalized)
            // Note: Unity's screen space has Y=0 at bottom, but image Y=0 is at top
            // So we need to flip the Y coordinate
            float viewportX = center_px.x / (float)imageWidth;
            float viewportY = 1.0f - (center_px.y / (float)imageHeight); // Flip Y

            Vector3 viewportPoint = new Vector3(viewportX, viewportY, 0);

            // Create ray from camera through this viewport point
            Ray ray = camera.ViewportPointToRay(viewportPoint);

            // Perform raycast to find world position
            if (Physics.Raycast(ray, out RaycastHit hit, maxDistance: 100f))
            {
                worldPosition = hit.point;
                hitObject = hit.collider.gameObject;
                return true;
            }

            return false;
        }

        public override string ToString()
        {
            return $"{_logPrefix} {color} cube at ({center_px.x}, {center_px.y}) conf={confidence:F2}";
        }
    }

    /// <summary>
    /// Bounding box in pixel coordinates
    /// </summary>
    [Serializable]
    public class BoundingBoxPx
    {
        public int x;
        public int y;
        public int width;
        public int height;

        private const string _logPrefix = "[BOUNDING_BOX_PX]";

        /// <summary>
        /// Get the center point of the bounding box
        /// </summary>
        public Vector2Int Center => new Vector2Int(x + width / 2, y + height / 2);

        /// <summary>
        /// Get the area of the bounding box in pixels
        /// </summary>
        public int Area => width * height;

        public override string ToString()
        {
            return $"{_logPrefix} BBox({x}, {y}, {width}×{height})";
        }
    }

    /// <summary>
    /// 3D world position in meters (from stereo depth estimation)
    /// </summary>
    [Serializable]
    public class WorldPosition
    {
        public float x; // X coordinate in meters (right positive)
        public float y; // Y coordinate in meters (up positive)
        public float z; // Z coordinate in meters (forward positive)

        private const string _logPrefix = "[WORLD_POSITION]";

        /// <summary>
        /// Convert to Unity Vector3
        /// </summary>
        public Vector3 ToVector3()
        {
            return new Vector3(x, y, z);
        }

        /// <summary>
        /// Check if this world position is valid (non-null and not NaN)
        /// </summary>
        public bool IsValid()
        {
            return !float.IsNaN(x)
                && !float.IsNaN(y)
                && !float.IsNaN(z)
                && !float.IsInfinity(x)
                && !float.IsInfinity(y)
                && !float.IsInfinity(z);
        }

        public override string ToString()
        {
            return $"{_logPrefix} WorldPos({x:F3}, {y:F3}, {z:F3})m";
        }
    }

    /// <summary>
    /// Metadata about the detection result
    /// </summary>
    [Serializable]
    public class DetectionMetadata
    {
        public string server_timestamp;
        public float processing_time_seconds;
        public string prompt;
        public float camera_baseline_m; // Stereo baseline distance
        public float camera_fov_deg; // Camera field of view
        public string detection_mode; // "mono_2d" or "stereo_3d"

        private const string _logPrefix = "[DETECTION_METADATA]";

        public override string ToString()
        {
            string mode = !string.IsNullOrEmpty(detection_mode) ? $" mode={detection_mode}" : "";
            return $"{_logPrefix} Metadata[timestamp={server_timestamp}{mode}]";
        }
    }

    /// <summary>
    /// Enhanced detection result with Unity world coordinates automatically calculated
    /// </summary>
    public class DetectionResultWithWorld
    {
        public DetectionResult OriginalResult { get; private set; }
        public DetectedCubeWithWorld[] CubesWithWorldCoords { get; private set; }
        public Camera SourceCamera { get; private set; }

        public DetectionResultWithWorld(
            DetectionResult result,
            Camera sourceCamera,
            DetectedCubeWithWorld[] cubesWithWorld
        )
        {
            OriginalResult = result;
            SourceCamera = sourceCamera;
            CubesWithWorldCoords = cubesWithWorld;
        }
    }

    /// <summary>
    /// Detected cube with both pixel and world coordinates
    /// </summary>
    public class DetectedCubeWithWorld
    {
        public DetectionObject OriginalDetection { get; private set; }
        public Vector3 WorldPosition { get; private set; }
        public GameObject HitObject { get; private set; }
        public bool HasWorldPosition { get; private set; }

        private const string _logPrefix = "[DETECTED_CUBE_WITH_WORLD]";

        public DetectedCubeWithWorld(
            DetectionObject detection,
            Vector3 worldPosition,
            GameObject hitObject,
            bool hasWorldPosition
        )
        {
            OriginalDetection = detection;
            WorldPosition = worldPosition;
            HitObject = hitObject;
            HasWorldPosition = hasWorldPosition;
        }

        public override string ToString()
        {
            if (HasWorldPosition)
            {
                string objName = HitObject != null ? HitObject.name : "none";
                return $"{_logPrefix} {OriginalDetection.color} cube at world {WorldPosition} (hit: {objName})";
            }
            else
            {
                return $"{_logPrefix} {OriginalDetection.color} cube at pixel ({OriginalDetection.center_px.x}, {OriginalDetection.center_px.y}) - no world position";
            }
        }
    }
}
