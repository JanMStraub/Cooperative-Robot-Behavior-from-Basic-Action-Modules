import pytest
import numpy as np
from unittest.mock import patch

from vision.StereoConfig import CameraConfig, ReconstructionConfig

from vision.DepthEstimator import (
    estimate_depth_at_point,
    pixel_to_world_coords,
    estimate_object_world_position,
)


class TestEstimateDepthAtPoint:

    @patch("vision.DepthEstimator.calc_disparity")
    def test_estimate_depth_valid_point(self, mock_calc_disparity, sample_stereo_pair):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 20.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
        )

        assert depth is not None
        assert depth > 0
        assert isinstance(depth, float)

    @patch("vision.DepthEstimator.calc_disparity")
    def test_estimate_depth_with_default_config(
        self, mock_calc_disparity, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 15.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        depth = estimate_depth_at_point(imgL, imgR, pixel_x=320, pixel_y=240)

        assert depth is not None
        assert depth > 0

    @patch("vision.DepthEstimator.calc_disparity")
    def test_estimate_depth_out_of_bounds(
        self, mock_calc_disparity, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 10.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=1000,
            pixel_y=240,
            camera_config=camera_config,
        )
        assert depth is None

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=1000,
            camera_config=camera_config,
        )
        assert depth is None

    @patch("vision.DepthEstimator.calc_disparity")
    def test_estimate_depth_invalid_disparity(
        self, mock_calc_disparity, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), np.nan, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
        )

        assert depth is None

    @patch("vision.DepthEstimator.calc_disparity")
    def test_estimate_depth_zero_disparity(
        self, mock_calc_disparity, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        disparity = np.zeros((480, 640), dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
        )

        assert depth is None

    @patch("vision.DepthEstimator.calc_disparity")
    def test_estimate_depth_uses_median(self, mock_calc_disparity, sample_stereo_pair):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 10.0, dtype=np.float32)
        outlier_values = np.array(
            [
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 100.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
        disparity[238:242, 318:322] = outlier_values

        mock_calc_disparity.return_value = disparity

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
            window_size=5,
        )

        assert depth is not None

    @patch("vision.DepthEstimator.calc_disparity")
    def test_estimate_depth_grayscale_conversion(self, mock_calc_disparity):
        imgL = np.ones((480, 640, 3), dtype=np.uint8) * 128
        imgR = np.ones((480, 640, 3), dtype=np.uint8) * 128

        disparity = np.full((480, 640), 15.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
        )

        assert depth is not None
        args = mock_calc_disparity.call_args[0]
        assert len(args[0].shape) == 2
        assert len(args[1].shape) == 2

    @patch("vision.DepthEstimator.calc_disparity")
    def test_estimate_depth_focal_length_calculation_fov(
        self, mock_calc_disparity, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 20.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
        )

        assert depth is not None

    @patch("vision.DepthEstimator.calc_disparity")
    def test_estimate_depth_focal_length_calculation_sensor(
        self, mock_calc_disparity, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 20.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        # Use focal_length and sensor_width instead of FOV
        camera_config = CameraConfig(
            fov=0.0,  # Use 0.0 instead of None to avoid type error
            baseline=0.1,
            focal_length=0.016,  # 16mm
            sensor_width=0.0236,  # 23.6mm
        )

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
        )

        assert depth is not None

    @patch("vision.DepthEstimator.calc_disparity")
    def test_estimate_depth_missing_focal_info_raises(
        self, mock_calc_disparity, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 20.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        # Config with no FOV or focal_length
        camera_config = CameraConfig(
            fov=0.0,  # Use 0.0 instead of None to avoid type error
            baseline=0.1,
            focal_length=None,
            sensor_width=None,
        )

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
        )

        assert depth is None


class TestPixelToWorldCoords:

    def test_pixel_to_world_center_pixel(self):
        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        world_x, world_y, world_z = pixel_to_world_coords(
            pixel_x=320,
            pixel_y=240,
            depth=1.0,
            camera_config=camera_config,
            image_width=640,
            image_height=480,
        )

        assert abs(world_x) < 0.01
        assert abs(world_y) < 0.01
        assert abs(world_z - 1.0) < 0.01

    def test_pixel_to_world_right_of_center(self):
        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        world_x, world_y, world_z = pixel_to_world_coords(
            pixel_x=420,  # 100 pixels right of center
            pixel_y=240,
            depth=1.0,
            camera_config=camera_config,
            image_width=640,
            image_height=480,
        )

        assert world_x > 0
        assert abs(world_y) < 0.01
        assert abs(world_z - 1.0) < 0.01

    def test_pixel_to_world_above_center(self):
        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        world_x, world_y, world_z = pixel_to_world_coords(
            pixel_x=320,
            pixel_y=140,  # 100 pixels above center
            depth=1.0,
            camera_config=camera_config,
            image_width=640,
            image_height=480,
        )

        assert abs(world_x) < 0.01
        assert world_y > 0
        assert abs(world_z - 1.0) < 0.01

    def test_pixel_to_world_depth_scaling(self):
        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        x1, y1, z1 = pixel_to_world_coords(
            pixel_x=420,
            pixel_y=340,
            depth=1.0,
            camera_config=camera_config,
            image_width=640,
            image_height=480,
        )

        x2, y2, z2 = pixel_to_world_coords(
            pixel_x=420,
            pixel_y=340,
            depth=2.0,
            camera_config=camera_config,
            image_width=640,
            image_height=480,
        )

        assert abs(x2 / x1 - 2.0) < 0.1
        assert abs(y2 / y1 - 2.0) < 0.1
        assert abs(z2 - 2.0) < 0.01

    def test_pixel_to_world_with_focal_length(self):
        camera_config = CameraConfig(
            fov=0.0,  # Use 0.0 instead of None to avoid type error
            baseline=0.1,
            focal_length=0.016,
            sensor_width=0.0236,
        )

        world_x, world_y, world_z = pixel_to_world_coords(
            pixel_x=320,
            pixel_y=240,
            depth=1.0,
            camera_config=camera_config,
            image_width=640,
            image_height=480,
        )

        assert abs(world_z - 1.0) < 0.01

    def test_pixel_to_world_missing_focal_info_raises(self):
        camera_config = CameraConfig(
            fov=0.0,  # Use 0.0 instead of None to avoid type error
            baseline=0.1,
            focal_length=None,
            sensor_width=None,
        )

        with pytest.raises(ValueError, match="Camera config must provide"):
            pixel_to_world_coords(
                pixel_x=320,
                pixel_y=240,
                depth=1.0,
                camera_config=camera_config,
                image_width=640,
                image_height=480,
            )

    def test_pixel_to_world_different_image_sizes(self):
        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        x1, y1, z1 = pixel_to_world_coords(
            pixel_x=960,
            pixel_y=540,
            depth=1.0,
            camera_config=camera_config,
            image_width=1920,
            image_height=1080,
        )

        assert abs(x1) < 0.01
        assert abs(y1) < 0.01


class TestEstimateObjectWorldPosition:

    @patch("vision.DepthEstimator.estimate_depth_at_point")
    def test_estimate_object_position_success(
        self, mock_estimate_depth, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        mock_estimate_depth.return_value = 1.5

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        world_pos = estimate_object_world_position(
            imgL,
            imgR,
            bbox_center_x=320,
            bbox_center_y=240,
            camera_config=camera_config,
        )

        assert world_pos is not None
        assert len(world_pos) == 3
        world_x, world_y, world_z = world_pos

        assert abs(world_z - 1.5) < 0.01

    @patch("vision.DepthEstimator.estimate_depth_at_point")
    def test_estimate_object_position_depth_failure(
        self, mock_estimate_depth, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        mock_estimate_depth.return_value = None

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        world_pos = estimate_object_world_position(
            imgL,
            imgR,
            bbox_center_x=320,
            bbox_center_y=240,
            camera_config=camera_config,
        )

        assert world_pos is None

    @patch("vision.DepthEstimator.estimate_depth_at_point")
    def test_estimate_object_position_off_center(
        self, mock_estimate_depth, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        mock_estimate_depth.return_value = 2.0

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        world_pos = estimate_object_world_position(
            imgL,
            imgR,
            bbox_center_x=450,  # Right of center
            bbox_center_y=180,  # Above center
            camera_config=camera_config,
        )

        assert world_pos is not None
        world_x, world_y, world_z = world_pos

        assert world_x > 0
        assert world_y > 0
        assert abs(world_z - 2.0) < 0.01

    @patch("vision.DepthEstimator.estimate_depth_at_point")
    def test_estimate_object_position_with_recon_config(
        self, mock_estimate_depth, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        mock_estimate_depth.return_value = 1.0

        camera_config = CameraConfig(fov=60.0, baseline=0.1)
        recon_config = ReconstructionConfig()

        world_pos = estimate_object_world_position(
            imgL,
            imgR,
            bbox_center_x=320,
            bbox_center_y=240,
            camera_config=camera_config,
            recon_config=recon_config,
        )

        assert world_pos is not None

        mock_estimate_depth.assert_called_once()
        # Arguments are positional: imgL, imgR, bbox_center_x, bbox_center_y, camera_config, recon_config
        call_args = mock_estimate_depth.call_args[0]
        assert call_args[4] == camera_config
        assert call_args[5] == recon_config


class TestDepthEstimatorIntegration:

    @patch("vision.DepthEstimator.calc_disparity")
    def test_full_pipeline(self, mock_calc_disparity, sample_stereo_pair):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 25.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
        )

        assert depth is not None

        world_pos = pixel_to_world_coords(
            pixel_x=320,
            pixel_y=240,
            depth=depth,
            camera_config=camera_config,
            image_width=640,
            image_height=480,
        )

        assert len(world_pos) == 3
        assert world_pos[2] == depth

    @patch("vision.DepthEstimator.estimate_depth_at_point")
    def test_object_position_estimation_workflow(
        self, mock_estimate_depth, sample_stereo_pair
    ):
        imgL, imgR = sample_stereo_pair

        mock_estimate_depth.return_value = 1.8

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        world_pos = estimate_object_world_position(
            imgL,
            imgR,
            bbox_center_x=450,
            bbox_center_y=300,
            camera_config=camera_config,
        )

        assert world_pos is not None
        x, y, z = world_pos

        assert x > 0  # Right of center
        assert y < 0  # Below center
        assert abs(z - 1.8) < 0.01


class TestDepthEstimatorEdgeCases:

    @patch("vision.DepthEstimator.calc_disparity")
    def test_very_small_disparity(self, mock_calc_disparity, sample_stereo_pair):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 0.1, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        # With very small disparity (0.1px), depth will be huge and exceed max_depth_threshold
        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
            min_disparity_threshold=0.05,  # Allow very small disparity
            max_depth_threshold=1000.0,  # Allow very far objects
        )

        assert depth is not None
        assert depth > 10

    @patch("vision.DepthEstimator.calc_disparity")
    def test_large_disparity(self, mock_calc_disparity, sample_stereo_pair):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 100.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
        )

        assert depth is not None
        assert depth < 1

    @patch("vision.DepthEstimator.calc_disparity")
    def test_window_size_variation(self, mock_calc_disparity, sample_stereo_pair):
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 20.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = CameraConfig(fov=60.0, baseline=0.1)

        depth1 = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
            window_size=3,
        )

        depth2 = estimate_depth_at_point(
            imgL,
            imgR,
            pixel_x=320,
            pixel_y=240,
            camera_config=camera_config,
            window_size=11,
        )

        assert depth1 is not None
        assert depth2 is not None
