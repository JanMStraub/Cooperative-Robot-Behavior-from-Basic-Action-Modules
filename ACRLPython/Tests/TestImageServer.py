#!/usr/bin/env python3
"""
Unit tests for ImageStorage (formerly ImageServer)

Tests the image storage and retrieval system including:
- Image storage and retrieval
- Camera ID management
- Stereo pair handling
- Image age tracking
- Memory management
- Concurrent image reception
"""

import pytest
import numpy as np
import cv2
import time
import threading
from unittest.mock import Mock, patch

from servers.StreamingServer import ImageStorage


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def image_storage():
    """
    Create a fresh ImageStorage instance for testing.

    Returns:
        ImageStorage instance
    """
    # Reset singleton
    ImageStorage._instance = None
    ImageStorage._cameras = {}
    storage = ImageStorage.get_instance()
    return storage


@pytest.fixture
def sample_image():
    """
    Create a sample test image.

    Returns:
        numpy array representing a 100x100 RGB image
    """
    # Create a simple 100x100 RGB image with some color
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = [128, 64, 32]  # BGR color
    return image


@pytest.fixture
def sample_stereo_pair():
    """
    Create a sample stereo image pair.

    Returns:
        Tuple of (left_image, right_image)
    """
    left = np.zeros((100, 100, 3), dtype=np.uint8)
    left[:, :] = [255, 0, 0]  # Blue

    right = np.zeros((100, 100, 3), dtype=np.uint8)
    right[:, :] = [0, 255, 0]  # Green

    return left, right


# ============================================================================
# Test ImageStorage Singleton
# ============================================================================

class TestImageStorageSingleton:
    """Test ImageStorage singleton behavior"""

    def test_singleton_instance(self, image_storage):
        """Test that ImageStorage is a singleton"""
        instance1 = ImageStorage.get_instance()
        instance2 = ImageStorage.get_instance()

        assert instance1 is instance2
        assert instance1 is image_storage

    def test_singleton_thread_safe(self):
        """Test singleton is thread-safe"""
        ImageStorage._instance = None
        instances = []

        def get_instance():
            instances.append(ImageStorage.get_instance())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same
        assert all(inst is instances[0] for inst in instances)


# ============================================================================
# Test Image Storage and Retrieval
# ============================================================================

class TestImageStorageBasics:
    """Test basic image storage and retrieval"""

    def test_store_and_retrieve_image(self, image_storage, sample_image):
        """Test storing and retrieving a single image"""
        camera_id = "main_camera"
        prompt = "detect objects"

        image_storage.store_image(camera_id, sample_image, prompt)

        retrieved = image_storage.get_camera_image(camera_id)

        assert retrieved is not None
        assert isinstance(retrieved, np.ndarray)
        assert retrieved.shape == sample_image.shape
        np.testing.assert_array_equal(retrieved, sample_image)

    def test_retrieve_nonexistent_camera(self, image_storage):
        """Test retrieving from non-existent camera returns None"""
        retrieved = image_storage.get_camera_image("nonexistent")
        assert retrieved is None

    def test_store_overwrites_previous_image(self, image_storage, sample_image):
        """Test that storing a new image overwrites the previous one"""
        camera_id = "main_camera"

        # Store first image
        image_storage.store_image(camera_id, sample_image, "first")

        # Store second image
        new_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
        image_storage.store_image(camera_id, new_image, "second")

        # Should retrieve the second image
        retrieved = image_storage.get_camera_image(camera_id)
        assert retrieved.shape == new_image.shape
        np.testing.assert_array_equal(retrieved, new_image)

    def test_retrieved_image_is_copy(self, image_storage, sample_image):
        """Test that retrieved image is a copy, not reference"""
        camera_id = "main_camera"
        image_storage.store_image(camera_id, sample_image)

        retrieved1 = image_storage.get_camera_image(camera_id)
        retrieved2 = image_storage.get_camera_image(camera_id)

        # Should be separate copies
        assert retrieved1 is not retrieved2
        # But with same data
        np.testing.assert_array_equal(retrieved1, retrieved2)


# ============================================================================
# Test Camera ID Management
# ============================================================================

class TestCameraIDManagement:
    """Test camera ID tracking and management"""

    def test_get_all_camera_ids_empty(self, image_storage):
        """Test getting camera IDs when storage is empty"""
        camera_ids = image_storage.get_all_camera_ids()
        assert camera_ids == []

    def test_get_all_camera_ids_single(self, image_storage, sample_image):
        """Test getting camera IDs with one camera"""
        image_storage.store_image("cam1", sample_image)

        camera_ids = image_storage.get_all_camera_ids()
        assert camera_ids == ["cam1"]

    def test_get_all_camera_ids_multiple(self, image_storage, sample_image):
        """Test getting camera IDs with multiple cameras"""
        image_storage.store_image("cam1", sample_image)
        image_storage.store_image("cam2", sample_image)
        image_storage.store_image("cam3", sample_image)

        camera_ids = image_storage.get_all_camera_ids()
        assert set(camera_ids) == {"cam1", "cam2", "cam3"}


# ============================================================================
# Test Stereo Pair Handling
# ============================================================================

class TestStereoPairHandling:
    """Test handling of stereo image pairs"""

    def test_store_stereo_pair(self, image_storage, sample_stereo_pair):
        """Test storing left and right stereo images"""
        left_image, right_image = sample_stereo_pair

        image_storage.store_image("left_camera", left_image)
        image_storage.store_image("right_camera", right_image)

        left_retrieved = image_storage.get_camera_image("left_camera")
        right_retrieved = image_storage.get_camera_image("right_camera")

        assert left_retrieved is not None
        assert right_retrieved is not None
        np.testing.assert_array_equal(left_retrieved, left_image)
        np.testing.assert_array_equal(right_retrieved, right_image)

    def test_stereo_cameras_independent(self, image_storage, sample_stereo_pair):
        """Test that stereo cameras are stored independently"""
        left_image, right_image = sample_stereo_pair

        image_storage.store_image("left_camera", left_image)

        # Right camera should not exist yet
        assert image_storage.get_camera_image("right_camera") is None

        image_storage.store_image("right_camera", right_image)

        # Now both should exist
        assert image_storage.get_camera_image("left_camera") is not None
        assert image_storage.get_camera_image("right_camera") is not None


# ============================================================================
# Test Prompt and Age Tracking
# ============================================================================

class TestPromptAndAgeTracking:
    """Test prompt and timestamp tracking"""

    def test_get_camera_prompt(self, image_storage, sample_image):
        """Test retrieving prompt associated with image"""
        camera_id = "main_camera"
        prompt = "detect red cubes"

        image_storage.store_image(camera_id, sample_image, prompt)

        retrieved_prompt = image_storage.get_camera_prompt(camera_id)
        assert retrieved_prompt == prompt

    def test_get_camera_prompt_empty(self, image_storage, sample_image):
        """Test getting prompt when none was provided"""
        camera_id = "main_camera"

        image_storage.store_image(camera_id, sample_image, "")

        retrieved_prompt = image_storage.get_camera_prompt(camera_id)
        assert retrieved_prompt == ""

    def test_get_camera_prompt_nonexistent(self, image_storage):
        """Test getting prompt for non-existent camera"""
        prompt = image_storage.get_camera_prompt("nonexistent")
        assert prompt is None

    def test_get_camera_age(self, image_storage, sample_image):
        """Test calculating image age"""
        camera_id = "main_camera"

        image_storage.store_image(camera_id, sample_image)

        # Wait a small amount of time
        time.sleep(0.1)

        age = image_storage.get_camera_age(camera_id)

        assert age is not None
        assert age >= 0.1
        assert age < 1.0  # Should be recent

    def test_get_camera_age_nonexistent(self, image_storage):
        """Test getting age for non-existent camera"""
        age = image_storage.get_camera_age("nonexistent")
        assert age is None


# ============================================================================
# Test Memory Management
# ============================================================================

class TestMemoryManagement:
    """Test cleanup and memory management"""

    def test_cleanup_old_images(self, image_storage, sample_image):
        """Test cleaning up old images"""
        # Store image with old timestamp
        camera_id = "old_camera"
        image_storage.store_image(camera_id, sample_image)

        # Manually set old timestamp
        with image_storage._lock:
            if camera_id in image_storage._cameras:
                img, _, prompt = image_storage._cameras[camera_id]
                # Set timestamp to 400 seconds ago
                image_storage._cameras[camera_id] = (img, time.time() - 400, prompt)

        # Cleanup images older than 300 seconds
        image_storage.cleanup_old_images(max_age_seconds=300.0)

        # Old image should be removed
        assert image_storage.get_camera_image(camera_id) is None

    def test_cleanup_keeps_recent_images(self, image_storage, sample_image):
        """Test cleanup keeps recent images"""
        camera_id = "recent_camera"
        image_storage.store_image(camera_id, sample_image)

        # Cleanup with 300 second threshold
        image_storage.cleanup_old_images(max_age_seconds=300.0)

        # Recent image should still exist
        assert image_storage.get_camera_image(camera_id) is not None

    def test_cleanup_multiple_cameras(self, image_storage, sample_image):
        """Test cleanup with multiple cameras"""
        # Store recent camera
        image_storage.store_image("recent", sample_image)

        # Store old camera
        image_storage.store_image("old", sample_image)
        with image_storage._lock:
            img, _, prompt = image_storage._cameras["old"]
            image_storage._cameras["old"] = (img, time.time() - 400, prompt)

        # Cleanup
        image_storage.cleanup_old_images(max_age_seconds=300.0)

        # Recent should exist, old should not
        assert image_storage.get_camera_image("recent") is not None
        assert image_storage.get_camera_image("old") is None


# ============================================================================
# Test Concurrent Access
# ============================================================================

class TestConcurrentAccess:
    """Test thread-safe concurrent operations"""

    def test_concurrent_image_storage(self, image_storage):
        """Test storing images concurrently from multiple threads"""
        num_threads = 10
        images_per_thread = 5

        def store_images(thread_id):
            for i in range(images_per_thread):
                camera_id = f"cam_{thread_id}_{i}"
                image = np.ones((50, 50, 3), dtype=np.uint8) * thread_id
                image_storage.store_image(camera_id, image)

        threads = [threading.Thread(target=store_images, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All images should be stored
        camera_ids = image_storage.get_all_camera_ids()
        assert len(camera_ids) == num_threads * images_per_thread

    def test_concurrent_read_write(self, image_storage, sample_image):
        """Test concurrent reads and writes"""
        camera_id = "shared_camera"
        image_storage.store_image(camera_id, sample_image)

        results = []

        def read_image():
            for _ in range(100):
                img = image_storage.get_camera_image(camera_id)
                results.append(img is not None)

        def write_image():
            for i in range(100):
                new_img = np.ones((50, 50, 3), dtype=np.uint8) * i
                image_storage.store_image(camera_id, new_img)

        readers = [threading.Thread(target=read_image) for _ in range(3)]
        writers = [threading.Thread(target=write_image) for _ in range(2)]

        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()

        # All reads should have succeeded
        assert all(results)
