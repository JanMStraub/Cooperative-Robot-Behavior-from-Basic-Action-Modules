# Stereo Image Reconstruction

A Python package for 3D point cloud reconstruction from stereoscopic image pairs using OpenCV's Semi-Global Block Matching (SGBM) algorithm.

## Features

- **Stereo Reconstruction**: Generate 3D point clouds from stereo image pairs
- **Feature Matching**: ORB-based feature detection and matching for disparity estimation
- **Real-time Streaming**: TCP server for receiving and processing stereo images from Unity
- **Configurable**: Centralized configuration system for all parameters
- **Type-Safe**: Full type hints throughout the codebase

## Installation

```bash
# Install dependencies
pip install opencv-python numpy open3d

# For development
pip install -e .
```

## Usage

### Basic Stereo Reconstruction

```python
from StereoImageReconstruction import (
    load_images,
    stereo_reconstruct,
    CameraConfig,
    ReconstructionConfig,
    OutputConfig,
)

# Load stereo images
imgL, imgR = load_images("left.png", "right.png")

# Configure camera parameters
camera_config = CameraConfig(
    fov=60.0,        # Field of view in degrees
    baseline=0.1,    # Distance between cameras in meters
)

# Configure reconstruction
recon_config = ReconstructionConfig(
    max_disparity=64,        # Maximum disparity (auto-estimate if None)
    mask_edges=False,        # Mask out strong edges
)

# Configure output
output_config = OutputConfig(
    output_base_dir="./output",
    save_disparity=True,
    save_point_cloud=True,
)

# Perform reconstruction
point_cloud = stereo_reconstruct(
    imgL, imgR,
    camera_config=camera_config,
    recon_config=recon_config,
    output_config=output_config,
)

print(f"Generated {len(point_cloud.verts)} points")
```

### Feature Matching

```python
from StereoImageReconstruction import find_matches, visualize_matches
import cv2

# Load grayscale images
imgL = cv2.imread("left.png", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread("right.png", cv2.IMREAD_GRAYSCALE)

# Find matches
matches, kp1, kp2 = find_matches(imgL, imgR)
print(f"Found {len(matches)} matches")

# Visualize matches
visualize_matches(imgL, imgR)
```

### Streaming Server

```python
from StereoImageReconstruction import (
    StereoStreamingServer,
    CameraConfig,
    ServerConfig,
)

# Configure server
camera_config = CameraConfig(fov=60.0, baseline=0.1)
server_config = ServerConfig(host="127.0.0.1", port=5005)

# Create and run server
server = StereoStreamingServer(
    camera_config=camera_config,
    server_config=server_config,
)

server.run()  # Starts TCP server and visualization
```

Or run from command line:

```bash
python -m StereoImageReconstruction.streaming_server \
    --host 127.0.0.1 \
    --port 5005 \
    --fov 60.0 \
    --baseline 0.1
```

## Command-Line Interface

### Reconstruct from Images

```bash
python -m StereoImageReconstruction.reconstruct \
    --l left.png \
    --r right.png \
    --fov 60 \
    --cam_dist 0.1 \
    --output_dir ./output
```

### Visualize Feature Matches

```bash
python -m StereoImageReconstruction.feature_match left.png right.png
```

## Configuration

All modules use centralized configuration through dataclasses:

### CameraConfig

```python
CameraConfig(
    fov=60.0,               # Field of view in degrees
    focal_length=None,      # Focal length in mm (alternative to fov)
    sensor_width=None,      # Sensor width in mm (alternative to fov)
    baseline=0.1,           # Stereo baseline in meters
)
```

### ReconstructionConfig

```python
ReconstructionConfig(
    window_size=2,              # SGBM window size
    min_disparity=0,            # Minimum disparity
    max_disparity=None,         # Maximum disparity (auto if None)
    uniqueness_ratio=40,        # SGBM uniqueness ratio
    speckle_window_size=20,     # Speckle filter window
    speckle_range=1,            # Speckle filter range
    mask_edges=False,           # Mask out strong edges
    min_depth_threshold=0.0,    # Minimum depth threshold
)
```

### FeatureMatchConfig

```python
FeatureMatchConfig(
    n_features=5000,            # Max ORB features
    scale_factor=1.1,           # ORB scale factor
    n_levels=2,                 # ORB pyramid levels
    lowe_ratio=0.5,             # Lowe's ratio test threshold
    max_y_diff=0.5,             # Max vertical disparity
    match_keep_ratio=0.7,       # Keep top 70% matches
)
```

### ServerConfig

```python
ServerConfig(
    host="127.0.0.1",           # Server host
    port=5005,                  # Server port
    max_connections=5,          # Max simultaneous clients
    window_width=1000,          # Visualization window width
    window_height=1000,         # Visualization window height
    voxel_downsample_size=0.02, # Voxel downsampling size
)
```

## Architecture

### Module Structure

```
StereoImageReconstruction/
├── __init__.py              # Package exports
├── config.py                # Configuration dataclasses
├── reconstruct.py           # Core reconstruction functions
├── feature_match.py         # Feature matching utilities
├── streaming_server.py      # Real-time streaming server
└── README.md                # This file
```

### Data Flow

1. **Input**: Stereo image pair (left and right cameras)
2. **Disparity Calculation**: SGBM algorithm computes disparity map
3. **3D Reprojection**: Convert disparity to 3D coordinates
4. **Point Cloud**: Output colored 3D point cloud (PLY format)

### Disparity to Depth Formula

```
Z = (f * T) / d

Where:
- Z: Depth (distance from camera)
- f: Focal length in pixels
- T: Baseline (distance between cameras)
- d: Disparity in pixels
```

## Integration with Unity

The streaming server receives stereo images from Unity via TCP:

**Protocol**: `[camera_id (1 byte)][image_size (4 bytes)][PNG image data]`

1. Unity sends left image (camera_id='L')
2. Unity sends right image (camera_id='R')
3. Server reconstructs point cloud
4. Server visualizes point cloud in Open3D

## Performance Tips

1. **Downscale images** for faster processing:
   ```python
   imgL, imgR = load_images("left.png", "right.png", downscale=2)
   ```

2. **Limit max disparity** to reduce computation:
   ```python
   recon_config = ReconstructionConfig(max_disparity=64)
   ```

3. **Enable voxel downsampling** for smoother visualization:
   ```python
   server_config = ServerConfig(voxel_downsample_size=0.02)
   ```

4. **Mask edges** to reduce noise (at cost of detail):
   ```python
   recon_config = ReconstructionConfig(mask_edges=True)
   ```

## Troubleshooting

### Empty Point Cloud

- Check that images are properly aligned (stereo rectified)
- Verify camera parameters (FOV, baseline)
- Try increasing `max_disparity`
- Disable edge masking

### Poor Matches

- Ensure sufficient texture in scene
- Adjust `n_features` in FeatureMatchConfig
- Check for proper lighting
- Verify images are from stereo pair (not reversed)

### Server Connection Issues

- Check firewall settings for TCP port
- Verify host/port match Unity configuration
- Ensure server is started before Unity connects

## References

- OpenCV SGBM Documentation: https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html
- Stereo Vision Tutorial: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
- Open3D Documentation: http://www.open3d.org/docs/

## License

Part of the ACRL (Auto-Cooperative Robot Learning) project.
