#!/usr/bin/env python3
"""
Unit tests for DepthEstimator.py

Tests stereo depth estimation and 3D coordinate conversion
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add LLMCommunication directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LLMCommunication"))

# Mock StereoImageReconstruction imports
sys.modules['StereoImageReconstruction'] = MagicMock()
sys.modules['StereoImageReconstruction.Reconstruct'] = MagicMock()
sys.modules['StereoImageReconstruction.stereo_config'] = MagicMock()


# Create mock CameraConfig for testing
class MockCameraConfig:  # type: ignore - Mock implements CameraConfig interface for testing
    """
    Mock camera configuration that implements the same interface as
    StereoImageReconstruction.config.CameraConfig for testing purposes.
    """
    def __init__(self, fov=60.0, baseline=0.1, focal_length=None, sensor_width=None):
        self.fov = fov
        self.baseline = baseline
        self.focal_length = focal_length
        self.sensor_width = sensor_width


# Create mock ReconstructionConfig
class MockReconstructionConfig:  # type: ignore - Mock implements ReconstructionConfig interface
    """
    Mock reconstruction configuration that implements the same interface as
    StereoImageReconstruction.config.ReconstructionConfig for testing purposes.
    """
    def __init__(self):
        self.algorithm = "SGBM"
        self.min_disparity = 0
        self.num_disparities = 64


# Patch the imports
sys.modules['StereoImageReconstruction.Reconstruct'].calc_disparity = MagicMock()
sys.modules['StereoImageReconstruction.stereo_config'].CameraConfig = MockCameraConfig
sys.modules['StereoImageReconstruction.stereo_config'].ReconstructionConfig = MockReconstructionConfig
sys.modules['StereoImageReconstruction.stereo_config'].DEFAULT_CAMERA_CONFIG = MockCameraConfig()
sys.modules['StereoImageReconstruction.stereo_config'].DEFAULT_RECONSTRUCTION_CONFIG = MockReconstructionConfig()

from LLMCommunication.vision.DepthEstimator import (
    estimate_depth_at_point,
    pixel_to_world_coords,
    estimate_object_world_position
)


class TestEstimateDepthAtPoint:
    """Test estimate_depth_at_point function"""

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_estimate_depth_valid_point(self, mock_calc_disparity, sample_stereo_pair):
        """Test depth estimation at valid pixel coordinate"""
        imgL, imgR = sample_stereo_pair

        # Mock disparity map with known values
        disparity = np.full((480, 640), 20.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert depth is not None
        assert depth > 0
        assert isinstance(depth, float)

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_estimate_depth_with_default_config(self, mock_calc_disparity, sample_stereo_pair):
        """Test depth estimation with default camera config"""
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 15.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240
        )

        assert depth is not None
        assert depth > 0

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_estimate_depth_out_of_bounds(self, mock_calc_disparity, sample_stereo_pair):
        """Test depth estimation with out-of-bounds coordinates"""
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 10.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        # Out of bounds X
        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=1000, pixel_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )
        assert depth is None

        # Out of bounds Y
        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=1000,
            camera_config=camera_config  # type: ignore[arg-type]
        )
        assert depth is None

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_estimate_depth_invalid_disparity(self, mock_calc_disparity, sample_stereo_pair):
        """Test depth estimation with invalid disparity values"""
        imgL, imgR = sample_stereo_pair

        # Create disparity map with NaN values
        disparity = np.full((480, 640), np.nan, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert depth is None

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_estimate_depth_zero_disparity(self, mock_calc_disparity, sample_stereo_pair):
        """Test depth estimation with zero/negative disparity"""
        imgL, imgR = sample_stereo_pair

        # Zero disparity (infinite depth)
        disparity = np.zeros((480, 640), dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert depth is None

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_estimate_depth_uses_median(self, mock_calc_disparity, sample_stereo_pair):
        """Test that depth estimation uses median for robustness"""
        imgL, imgR = sample_stereo_pair

        # Create disparity with outliers
        disparity = np.full((480, 640), 10.0, dtype=np.float32)
        # Add outliers in the center region (4x4 = 16 values)
        outlier_values = np.array([
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 100.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0]
        ], dtype=np.float32)
        disparity[238:242, 318:322] = outlier_values

        mock_calc_disparity.return_value = disparity

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config,  # type: ignore[arg-type]
            window_size=5
        )

        # Should use median, not affected by outlier
        assert depth is not None

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_estimate_depth_grayscale_conversion(self, mock_calc_disparity):
        """Test that color images are converted to grayscale"""
        # Color images
        imgL = np.ones((480, 640, 3), dtype=np.uint8) * 128
        imgR = np.ones((480, 640, 3), dtype=np.uint8) * 128

        disparity = np.full((480, 640), 15.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert depth is not None
        # Verify calc_disparity was called with grayscale
        args = mock_calc_disparity.call_args[0]
        assert len(args[0].shape) == 2  # Grayscale
        assert len(args[1].shape) == 2  # Grayscale

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_estimate_depth_focal_length_calculation_fov(self, mock_calc_disparity, sample_stereo_pair):
        """Test focal length calculation using FOV"""
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 20.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert depth is not None

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_estimate_depth_focal_length_calculation_sensor(self, mock_calc_disparity, sample_stereo_pair):
        """Test focal length calculation using sensor width"""
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 20.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        # Use focal_length and sensor_width instead of FOV
        camera_config = MockCameraConfig(
            fov=0.0,  # Use 0.0 instead of None to avoid type error
            baseline=0.1,
            focal_length=0.016,  # 16mm
            sensor_width=0.0236  # 23.6mm
        )

        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert depth is not None

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_estimate_depth_missing_focal_info_raises(self, mock_calc_disparity, sample_stereo_pair):
        """Test that missing focal length info returns None"""
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 20.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        # Config with no FOV or focal_length
        camera_config = MockCameraConfig(
            fov=0.0,  # Use 0.0 instead of None to avoid type error
            baseline=0.1,
            focal_length=None,
            sensor_width=None
        )

        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert depth is None


class TestPixelToWorldCoords:
    """Test pixel_to_world_coords function"""

    def test_pixel_to_world_center_pixel(self):
        """Test conversion of center pixel to world coordinates"""
        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        # Center pixel at 1 meter depth
        world_x, world_y, world_z = pixel_to_world_coords(
            pixel_x=320,
            pixel_y=240,
            depth=1.0,
            camera_config=camera_config,  # type: ignore[arg-type]
            image_width=640,
            image_height=480
        )

        # Center pixel should map to (0, 0, depth)
        assert abs(world_x) < 0.01
        assert abs(world_y) < 0.01
        assert abs(world_z - 1.0) < 0.01

    def test_pixel_to_world_right_of_center(self):
        """Test conversion of pixel to the right of center"""
        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        # Pixel to the right of center
        world_x, world_y, world_z = pixel_to_world_coords(
            pixel_x=420,  # 100 pixels right of center
            pixel_y=240,
            depth=1.0,
            camera_config=camera_config,  # type: ignore[arg-type]
            image_width=640,
            image_height=480
        )

        # X should be positive (right)
        assert world_x > 0
        assert abs(world_y) < 0.01  # Still centered vertically
        assert abs(world_z - 1.0) < 0.01

    def test_pixel_to_world_above_center(self):
        """Test conversion of pixel above center"""
        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        # Pixel above center
        world_x, world_y, world_z = pixel_to_world_coords(
            pixel_x=320,
            pixel_y=140,  # 100 pixels above center
            depth=1.0,
            camera_config=camera_config,  # type: ignore[arg-type]
            image_width=640,
            image_height=480
        )

        # Y should be positive (up, due to Y negation)
        assert abs(world_x) < 0.01
        assert world_y > 0  # Above center
        assert abs(world_z - 1.0) < 0.01

    def test_pixel_to_world_depth_scaling(self):
        """Test that world coordinates scale with depth"""
        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        # Same pixel at different depths
        x1, y1, z1 = pixel_to_world_coords(
            pixel_x=420, pixel_y=340,
            depth=1.0,
            camera_config=camera_config,  # type: ignore[arg-type]
            image_width=640, image_height=480
        )

        x2, y2, z2 = pixel_to_world_coords(
            pixel_x=420, pixel_y=340,
            depth=2.0,
            camera_config=camera_config,  # type: ignore[arg-type]
            image_width=640, image_height=480
        )

        # At 2x depth, X and Y should be ~2x
        assert abs(x2 / x1 - 2.0) < 0.1
        assert abs(y2 / y1 - 2.0) < 0.1
        assert abs(z2 - 2.0) < 0.01

    def test_pixel_to_world_with_focal_length(self):
        """Test conversion using focal_length instead of FOV"""
        camera_config = MockCameraConfig(
            fov=0.0,  # Use 0.0 instead of None to avoid type error
            baseline=0.1,
            focal_length=0.016,
            sensor_width=0.0236
        )

        world_x, world_y, world_z = pixel_to_world_coords(
            pixel_x=320,
            pixel_y=240,
            depth=1.0,
            camera_config=camera_config,  # type: ignore[arg-type]
            image_width=640,
            image_height=480
        )

        assert abs(world_z - 1.0) < 0.01

    def test_pixel_to_world_missing_focal_info_raises(self):
        """Test that missing focal info raises ValueError"""
        camera_config = MockCameraConfig(
            fov=0.0,  # Use 0.0 instead of None to avoid type error
            baseline=0.1,
            focal_length=None,
            sensor_width=None
        )

        with pytest.raises(ValueError, match="Invalid camera configuration"):
            pixel_to_world_coords(
                pixel_x=320,
                pixel_y=240,
                depth=1.0,
                camera_config=camera_config,  # type: ignore[arg-type]
                image_width=640,
                image_height=480
            )

    def test_pixel_to_world_different_image_sizes(self):
        """Test conversion with different image sizes"""
        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        # HD image
        x1, y1, z1 = pixel_to_world_coords(
            pixel_x=960, pixel_y=540,
            depth=1.0,
            camera_config=camera_config,  # type: ignore[arg-type]
            image_width=1920, image_height=1080
        )

        # Center should still be (0, 0, 1)
        assert abs(x1) < 0.01
        assert abs(y1) < 0.01


class TestEstimateObjectWorldPosition:
    """Test estimate_object_world_position function"""

    @patch('LLMCommunication.vision.DepthEstimator.estimate_depth_at_point')
    def test_estimate_object_position_success(self, mock_estimate_depth, sample_stereo_pair):
        """Test successful object position estimation"""
        imgL, imgR = sample_stereo_pair

        # Mock depth estimation
        mock_estimate_depth.return_value = 1.5

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        world_pos = estimate_object_world_position(
            imgL, imgR,
            bbox_center_x=320,
            bbox_center_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert world_pos is not None
        assert len(world_pos) == 3
        world_x, world_y, world_z = world_pos

        # Z should match the mocked depth
        assert abs(world_z - 1.5) < 0.01

    @patch('LLMCommunication.vision.DepthEstimator.estimate_depth_at_point')
    def test_estimate_object_position_depth_failure(self, mock_estimate_depth, sample_stereo_pair):
        """Test when depth estimation fails"""
        imgL, imgR = sample_stereo_pair

        # Mock depth estimation failure
        mock_estimate_depth.return_value = None

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        world_pos = estimate_object_world_position(
            imgL, imgR,
            bbox_center_x=320,
            bbox_center_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert world_pos is None

    @patch('LLMCommunication.vision.DepthEstimator.estimate_depth_at_point')
    def test_estimate_object_position_off_center(self, mock_estimate_depth, sample_stereo_pair):
        """Test object position estimation for off-center detection"""
        imgL, imgR = sample_stereo_pair

        mock_estimate_depth.return_value = 2.0

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        world_pos = estimate_object_world_position(
            imgL, imgR,
            bbox_center_x=450,  # Right of center
            bbox_center_y=180,  # Above center
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert world_pos is not None
        world_x, world_y, world_z = world_pos

        # Should be to the right and up
        assert world_x > 0
        assert world_y > 0
        assert abs(world_z - 2.0) < 0.01

    @patch('LLMCommunication.vision.DepthEstimator.estimate_depth_at_point')
    def test_estimate_object_position_with_recon_config(self, mock_estimate_depth, sample_stereo_pair):
        """Test with custom reconstruction config"""
        imgL, imgR = sample_stereo_pair

        mock_estimate_depth.return_value = 1.0

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)
        recon_config = MockReconstructionConfig()

        world_pos = estimate_object_world_position(
            imgL, imgR,
            bbox_center_x=320,
            bbox_center_y=240,
            camera_config=camera_config,  # type: ignore[arg-type]
            recon_config=recon_config  # type: ignore[arg-type]
        )

        assert world_pos is not None

        # Verify estimate_depth_at_point was called with correct params
        mock_estimate_depth.assert_called_once()
        # Arguments are positional: imgL, imgR, bbox_center_x, bbox_center_y, camera_config, recon_config
        call_args = mock_estimate_depth.call_args[0]
        assert call_args[4] == camera_config  # 5th positional arg
        assert call_args[5] == recon_config  # 6th positional arg


class TestDepthEstimatorIntegration:
    """Integration tests for depth estimation workflow"""

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_full_pipeline(self, mock_calc_disparity, sample_stereo_pair):
        """Test complete depth estimation pipeline"""
        imgL, imgR = sample_stereo_pair

        # Mock disparity map
        disparity = np.full((480, 640), 25.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        # Estimate depth
        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert depth is not None

        # Convert to world coordinates
        world_pos = pixel_to_world_coords(
            pixel_x=320, pixel_y=240,
            depth=depth,
            camera_config=camera_config,  # type: ignore[arg-type]
            image_width=640, image_height=480
        )

        assert len(world_pos) == 3
        assert world_pos[2] == depth

    @patch('LLMCommunication.vision.DepthEstimator.estimate_depth_at_point')
    def test_object_position_estimation_workflow(self, mock_estimate_depth, sample_stereo_pair):
        """Test object position estimation workflow"""
        imgL, imgR = sample_stereo_pair

        # Simulate detection at (450, 300) with depth 1.8m
        mock_estimate_depth.return_value = 1.8

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        world_pos = estimate_object_world_position(
            imgL, imgR,
            bbox_center_x=450,
            bbox_center_y=300,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        assert world_pos is not None
        x, y, z = world_pos

        # Verify position makes sense
        assert x > 0  # Right of center
        assert y < 0  # Below center
        assert abs(z - 1.8) < 0.01


class TestDepthEstimatorEdgeCases:
    """Test edge cases and error conditions"""

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_very_small_disparity(self, mock_calc_disparity, sample_stereo_pair):
        """Test with very small disparity (far object)"""
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 0.1, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        # Should work but depth will be very large
        assert depth is not None
        assert depth > 10  # Far away

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_large_disparity(self, mock_calc_disparity, sample_stereo_pair):
        """Test with large disparity (close object)"""
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 100.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        depth = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config  # type: ignore[arg-type]
        )

        # Should work, depth will be small
        assert depth is not None
        assert depth < 1  # Close

    @patch('LLMCommunication.vision.DepthEstimator.calc_disparity')
    def test_window_size_variation(self, mock_calc_disparity, sample_stereo_pair):
        """Test different window sizes for depth estimation"""
        imgL, imgR = sample_stereo_pair

        disparity = np.full((480, 640), 20.0, dtype=np.float32)
        mock_calc_disparity.return_value = disparity

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)

        # Small window
        depth1 = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config,  # type: ignore[arg-type]
            window_size=3
        )

        # Large window
        depth2 = estimate_depth_at_point(
            imgL, imgR,
            pixel_x=320, pixel_y=240,
            camera_config=camera_config,  # type: ignore[arg-type]
            window_size=11
        )

        # Both should work
        assert depth1 is not None
        assert depth2 is not None
