import argparse
import math
import cv2
import numpy as np


def load_images(path_left, path_right, downscale=1):
    """
    Load two stereoscopic images from the specified paths.
    :param path_left: Path to the image taken from the left camera.
    :param path_right: Path to the image taken from the right camera.
    :param downscale: Downscale factor (default: 1)
    :raises ValueError: If either image cannot be loaded from the given path.
    """
    imgL = cv2.imread(path_left)
    imgR = cv2.imread(path_right)

    if imgL is None:
        raise ValueError(f"Could not load left image from path: {path_left}")
    if imgR is None:
        raise ValueError(f"Could not load right image from path: {path_right}")

    shape = (int(imgL.shape[1] / downscale), int(imgL.shape[0] / downscale))
    imgL = cv2.resize(imgL, shape, interpolation=cv2.INTER_AREA)
    imgR = cv2.resize(imgR, shape, interpolation=cv2.INTER_AREA)

    print("Image shape:", imgL.shape, imgR.shape)
    return imgL, imgR


def calc_disparity(imgL, imgR, max_disp=None):
    """
    Calculate the disparity map between two images using StereoSGBM.
    :param imgL: Grayscale image from left camera.
    :param imgR: Grayscale image from right camera.
    :param max_disp: The maximum disparity (must be divisible by 16).
    :raises ValueError: If images are not single-channel (grayscale).
    """
    if imgL.ndim != 2 or imgR.ndim != 2:
        raise ValueError(
            "calc_disparity expects grayscale (single-channel) images. "
            "Convert with cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) first."
        )

    window_size = 5  # Increased from 2 for better smoothness

    if max_disp is None:
        # Default fallback if max_disp isn't provided.
        # 128 is a safe default for typical stereo setups.
        max_disp = 128

    # StereoSGBM requires numDisparities to be divisible by 16
    max_disp = max_disp if max_disp % 16 == 0 else max_disp + (16 - max_disp % 16)

    print(f"Using maximum disparity: {max_disp}")

    # Tuned parameters for standard robotic manipulation scenes
    stereo = cv2.StereoSGBM_create(  # type: ignore
        minDisparity=0,
        numDisparities=max_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,  # OpenCV recommended formula
        P2=32 * 3 * window_size**2,  # OpenCV recommended formula
        disp12MaxDiff=1,
        uniquenessRatio=10,  # Standard uniqueness (10-15)
        speckleWindowSize=100,  # Filter out small noise
        speckleRange=32,  # Standard speckle range
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,  # Faster than full HH, better than standard
    )
    disp = (stereo.compute(imgL, imgR).astype(np.float32)) / 16.0
    return np.where(disp >= 0.0, disp, np.nan)


def calc_normals(depth_map):
    """
    Compute a surface normal map from a depth map using Sobel gradients.
    :param depth_map: 2D float32 depth map (may contain NaN for invalid pixels).
    :return: uint8 RGB image encoding surface normals, remapped to [0, 255].
    """
    print("computing normal map...")
    ksize = 1
    grad_x = cv2.Sobel(
        depth_map, cv2.CV_32F, 1, 0, ksize=ksize
    )  # Gradient in X direction
    grad_y = cv2.Sobel(
        depth_map, cv2.CV_32F, 0, 1, ksize=ksize
    )  # Gradient in Y direction

    abs_max = max(
        np.abs(np.nan_to_num(grad_x)).max(), np.abs(np.nan_to_num(grad_y)).max()
    )

    # Normal map
    grad_z = np.full_like(depth_map, abs_max / 50)
    normals = np.stack((grad_z, -grad_y, -grad_x), axis=-1)  # OpenCV uses BGR
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    # Guard against division by zero on flat/invalid regions
    normals /= np.where(norm > 0, norm, 1.0)

    res = ((normals + 1) / 2.0) * 255
    return res.astype(np.uint8)


def make_3d(depth_map, imgL, focal_length, min_disp=0.0, cam_dist=0.1):
    """
    Generate a 3D point cloud from a depth map.
    :param depth_map: A grayscale image representing the depth of each pixel. Darker = further away.
    :param imgL: A reference image to color the point cloud.
    :param focal_length: The focal length of the camera.
    :param min_disp: The minimum disparity
    :param cam_dist: The distance between the two cameras.
    """
    depth_map = np.nan_to_num(depth_map, nan=-1).astype(np.float32)

    h, w = imgL.shape[:2]
    f = focal_length
    T_x = cam_dist
    # Q-matrix for reprojection: X is negated to match Unity's left-handed coordinate system.
    Q = np.array(
        [[-1, 0, 0, 0.5 * w], [0, 1, 0, -0.5 * h], [0, 0, 0, f], [0, 0, -1 / T_x, 0]],
        dtype=np.float32,
    )
    points = cv2.reprojectImageTo3D(depth_map, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = depth_map > min_disp
    out_points = points[mask]
    out_colors = colors[mask]

    return {"points": out_points, "colors": out_colors}


def remove_edges(image, ksize=5):
    """
    Mask out the strongest edges (changes in brightness) in the image.
    :param image: Input float32 image (may contain NaN).
    :param ksize: Kernel size used for both the median blur and Laplacian operator.
    :return: Image with edge pixels replaced by NaN.
    """
    img = np.copy(image)
    img = cv2.medianBlur(img, ksize)
    img = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)

    # Mask the top 10% of the strongest edges
    threshold_value = np.nanpercentile(np.asarray(img, dtype=np.float64), 90)
    img = np.where(img < threshold_value, 0.0, 1.0)

    # Thicken the edge mask a bit.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    img = cv2.dilate(img, kernel, iterations=1)

    return np.where(img != 0, np.nan, image)


def f_from_fov(fov):
    """
    Compute the normalised focal length from a field-of-view angle.
    :param fov: Horizontal field of view in degrees.
    :return: Normalised focal length (focal_length / sensor_width).
    """
    return 1 / (2 * math.tan(math.radians(fov / 2)))


def reconstruct(
    disparity,
    imgL_rgb,
    fov=None,
    focal_length=None,
    sensor_width=None,
    min_disp=0.0,
    cam_dist=0.1,
):
    """
    Reconstruct a point cloud from a disparity map.
    :param disparity: The disparity map between the two images.
    :param imgL_rgb: The reference image to color the point cloud.
    :param fov: The field of view of the camera. If provided, focal_length and sensor_width are ignored.
    :param focal_length: The focal length of the camera. Required if fov is not provided.
    :param sensor_width: The width of the camera sensor. Required if fov is not provided.
    :param min_disp: The minimum disparity. (Values below this are ignored in the resulting pointcloud).
    :param cam_dist: The distance between the two cameras.
    """
    if fov is not None:
        f = f_from_fov(fov)
    elif focal_length is not None and sensor_width is not None:
        f = focal_length / sensor_width
    else:
        raise ValueError(
            "Either fov or focal_length and sensor_width must be provided."
        )

    f = f * disparity.shape[1]
    point_cloud = make_3d(disparity, imgL_rgb, f, min_disp=min_disp, cam_dist=cam_dist)

    return point_cloud


def stereo_reconstruct(
    imgL,
    imgR,
    fov=None,
    focal_length=None,
    sensor_width=None,
    min_disp=0.0,
    max_disp=None,
    cam_dist=0.1,
    mask_edges=False,
):
    """
    Reconstruct a point cloud from two stereoscopic images.
    :param imgL: Image taken from the left camera.
    :param imgR: Image taken from the right camera.
    :param fov: The field of view of the camera. If provided, focal_length and sensor_width are ignored.
    :param focal_length: The focal length of the camera. Required if fov is not provided.
    :param sensor_width: The width of the camera sensor. Required if fov is not provided.
    :param min_disp: The minimum disparity. (Values below this are ignored in the resulting pointcloud).
    :param max_disp: The maximum disparity.
    :param cam_dist: The distance between the two cameras.
    :param mask_edges: Mask out strong edges in the depth map.
    """
    imgL_rgb = imgL.copy()
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    print("computing disparity...")
    disparity = calc_disparity(imgL, imgR, max_disp=max_disp)
    if mask_edges:
        disparity = remove_edges(disparity)

    print("generating 3d point cloud...")
    point_cloud = reconstruct(
        disparity,
        imgL_rgb,
        fov,
        focal_length,
        sensor_width,
        min_disp,
        cam_dist,
    )

    return point_cloud


def stereo_reconstruct_stream(
    imgL,
    imgR,
    fov=None,
    focal_length=None,
    sensor_width=None,
    min_disp=0.0,
    max_disp=None,
    cam_dist=0.1,
):
    """
    Reconstruct a point cloud from two stereoscopic images received via streaming.
    This function uses the existing reconstruction functions (e.g. calc_disparity, reconstruct)
    but does not write any data to disk.

    :param imgL: Left camera image (BGR format)
    :param imgR: Right camera image (BGR format)
    :param fov: Field of view of the camera (default: 60)
    :param focal_length: (Optional) Focal length of the camera
    :param sensor_width: (Optional) Width of the camera sensor
    :param min_disp: Minimum disparity to consider
    :param max_disp: Maximum disparity (passed to calc_disparity; defaults to 128 if None)
    :param cam_dist: Distance between the two cameras
    :return: A dict containing 'points' and 'colors' arrays (point cloud data)
    """
    # Preserve the original color image for point cloud coloring.
    imgL_rgb = imgL.copy()
    # Convert images to grayscale for disparity calculation.
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    print("Computing disparity...")
    disparity = calc_disparity(imgL_gray, imgR_gray, max_disp=max_disp)

    print("Generating 3D point cloud...")
    point_cloud = reconstruct(
        disparity, imgL_rgb, fov, focal_length, sensor_width, min_disp, cam_dist
    )
    return point_cloud


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process two images.")
    parser.add_argument(
        "--l",
        type=str,
        default="./misc/robot_scene/LeftCamera_3.png",
        help="Path to the image taken from the left camera",
    )
    parser.add_argument(
        "--r",
        type=str,
        default="./misc/robot_scene/RightCamera_3.png",
        help="Path to the image taken from the right camera",
    )
    parser.add_argument(
        "--downscale", type=int, default=1, help="Downscale factor (default: 1)"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Field of view of the camera (provide this or focal_length and sensor_width)",
    )
    parser.add_argument("--focal_length", type=float, help="Focal length of the camera")
    parser.add_argument("--sensor_width", type=float, help="Width of the camera sensor")
    parser.add_argument("--min_disp", type=int, default=0, help="Minimum disparity")
    parser.add_argument("--max_disp", type=int, default=None, help="Maximum disparity")
    parser.add_argument(
        "--cam_dist", type=float, default=0.1, help="Distance between the two cameras"
    )
    parser.add_argument(
        "--mask_edges",
        action="store_true",
        help="Mask out strong edges in the depth map.",
    )

    args = parser.parse_args()

    imgL, imgR = load_images(args.l, args.r, args.downscale)

    point_cloud = stereo_reconstruct(
        imgL,
        imgR,
        fov=args.fov,
        focal_length=args.focal_length,
        sensor_width=args.sensor_width,
        min_disp=args.min_disp,
        max_disp=args.max_disp,
        cam_dist=args.cam_dist,
        mask_edges=args.mask_edges,
    )
    print(f"Point cloud: {len(point_cloud['points'])} points")
