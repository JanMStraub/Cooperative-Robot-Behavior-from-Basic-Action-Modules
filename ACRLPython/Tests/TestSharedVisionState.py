#!/usr/bin/env python3
"""
test_shared_vision_state.py - Unit tests for SharedVisionState

Tests the thread-safe shared vision state for multi-robot coordination.
"""

import sys
import os
import unittest
import time
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from operations.SharedVisionState import (
    SharedVisionState,
    ClaimedObject,
    get_shared_vision_state,
)
from vision.DetectionDataModels import DetectionObject


class TestClaimedObject(unittest.TestCase):
    """Test ClaimedObject dataclass"""

    def test_claimed_object_creation(self):
        """Test ClaimedObject initialization"""
        obj = ClaimedObject(
            object_id="blue_cube_1",
            color="blue_cube",
            world_position=(0.3, 0.1, 0.0),
            claimed_by="Robot1",
            claim_timestamp=time.time(),
            track_id=5,
            confidence=0.95,
            depth_m=0.8,
        )

        self.assertEqual(obj.object_id, "blue_cube_1")
        self.assertEqual(obj.color, "blue_cube")
        self.assertEqual(obj.world_position, (0.3, 0.1, 0.0))
        self.assertEqual(obj.claimed_by, "Robot1")
        self.assertEqual(obj.track_id, 5)
        self.assertEqual(obj.confidence, 0.95)
        self.assertEqual(obj.depth_m, 0.8)

    def test_claimed_object_defaults(self):
        """Test ClaimedObject default values"""
        obj = ClaimedObject(
            object_id="test", color="red", world_position=(0.0, 0.0, 0.0)
        )

        self.assertIsNone(obj.claimed_by)
        self.assertEqual(obj.claim_timestamp, 0.0)
        self.assertIsNone(obj.track_id)
        self.assertEqual(obj.confidence, 1.0)
        self.assertIsNone(obj.depth_m)


class TestSharedVisionState(unittest.TestCase):
    """Test SharedVisionState class"""

    def setUp(self):
        """Create fresh SharedVisionState for each test"""
        self.state = SharedVisionState(claim_timeout=10.0)

    def tearDown(self):
        """Clean up after each test"""
        self.state.clear()

    def test_initialization(self):
        """Test SharedVisionState initialization"""
        self.assertEqual(len(self.state.detections), 0)
        self.assertEqual(self.state.claim_timeout, 10.0)
        self.assertIsNotNone(self.state.lock)

    def test_update_detections_new_objects(self):
        """Test updating state with new detections"""
        detections = [
            DetectionObject(
                object_id=1,
                color="blue_cube",
                bbox=(100, 100, 50, 50),
                confidence=0.9,
                world_position=(0.3, 0.1, 0.0),
                track_id=1,
            ),
            DetectionObject(
                object_id=2,
                color="red_cube",
                bbox=(200, 100, 50, 50),
                confidence=0.85,
                world_position=(0.5, 0.1, 0.2),
                track_id=2,
            ),
        ]

        self.state.update_detections(detections)

        self.assertEqual(len(self.state.detections), 2)

        # Check object IDs are generated with track_id
        obj_ids = list(self.state.detections.keys())
        self.assertIn("blue_cube_track_1", obj_ids)
        self.assertIn("red_cube_track_2", obj_ids)

    def test_update_detections_preserves_claims(self):
        """Test updating detections preserves existing claims"""
        # Add initial detection
        det1 = DetectionObject(
            object_id=1,
            color="blue_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            world_position=(0.3, 0.1, 0.0),
            track_id=1,
        )
        self.state.update_detections([det1])

        # Claim the object
        object_id = "blue_cube_track_1"
        self.state.claim_object(object_id, "Robot1")

        # Update with new detection (same track_id, moved position)
        det2 = DetectionObject(
            object_id=2,
            color="blue_cube",
            bbox=(105, 102, 50, 50),
            confidence=0.95,
            world_position=(0.32, 0.12, 0.0),
            track_id=1,  # Same track
        )
        self.state.update_detections([det2])

        # Claim should be preserved
        obj = self.state.detections[object_id]
        self.assertEqual(obj.claimed_by, "Robot1")
        # Position should be updated
        self.assertEqual(obj.world_position, (0.32, 0.12, 0.0))

    def test_claim_object_success(self):
        """Test claiming an available object"""
        # Add detection
        det = DetectionObject(
            object_id=1,
            color="blue_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            world_position=(0.3, 0.1, 0.0),
            track_id=1,
        )
        self.state.update_detections([det])

        # Claim object
        object_id = "blue_cube_track_1"
        success = self.state.claim_object(object_id, "Robot1")

        self.assertTrue(success)
        obj = self.state.detections[object_id]
        self.assertEqual(obj.claimed_by, "Robot1")
        self.assertGreater(obj.claim_timestamp, 0)

    def test_claim_object_already_claimed(self):
        """Test claiming an already claimed object fails"""
        # Add and claim object
        det = DetectionObject(
            object_id=1,
            color="blue_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            world_position=(0.3, 0.1, 0.0),
            track_id=1,
        )
        self.state.update_detections([det])

        object_id = "blue_cube_track_1"
        self.state.claim_object(object_id, "Robot1")

        # Try to claim with different robot
        success = self.state.claim_object(object_id, "Robot2")

        self.assertFalse(success)
        # Should still be claimed by Robot1
        obj = self.state.detections[object_id]
        self.assertEqual(obj.claimed_by, "Robot1")

    def test_claim_object_refresh_timestamp(self):
        """Test claiming same object by same robot refreshes timestamp"""
        # Add and claim object
        det = DetectionObject(
            object_id=1,
            color="blue_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            world_position=(0.3, 0.1, 0.0),
            track_id=1,
        )
        self.state.update_detections([det])

        object_id = "blue_cube_track_1"
        self.state.claim_object(object_id, "Robot1")

        first_timestamp = self.state.detections[object_id].claim_timestamp

        # Wait a bit
        time.sleep(0.1)

        # Claim again (refresh)
        success = self.state.claim_object(object_id, "Robot1")

        self.assertTrue(success)
        second_timestamp = self.state.detections[object_id].claim_timestamp
        self.assertGreater(second_timestamp, first_timestamp)

    def test_release_object_success(self):
        """Test releasing a claimed object"""
        # Add and claim object
        det = DetectionObject(
            object_id=1,
            color="blue_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            world_position=(0.3, 0.1, 0.0),
            track_id=1,
        )
        self.state.update_detections([det])

        object_id = "blue_cube_track_1"
        self.state.claim_object(object_id, "Robot1")

        # Release
        success = self.state.release_object(object_id, "Robot1")

        self.assertTrue(success)
        obj = self.state.detections[object_id]
        self.assertIsNone(obj.claimed_by)
        self.assertEqual(obj.claim_timestamp, 0.0)

    def test_release_object_wrong_robot(self):
        """Test releasing object by wrong robot fails"""
        # Add and claim object
        det = DetectionObject(
            object_id=1,
            color="blue_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            world_position=(0.3, 0.1, 0.0),
            track_id=1,
        )
        self.state.update_detections([det])

        object_id = "blue_cube_track_1"
        self.state.claim_object(object_id, "Robot1")

        # Try to release with different robot
        success = self.state.release_object(object_id, "Robot2")

        self.assertFalse(success)
        # Should still be claimed by Robot1
        obj = self.state.detections[object_id]
        self.assertEqual(obj.claimed_by, "Robot1")

    def test_get_available_objects_all(self):
        """Test getting all available objects"""
        # Add two objects
        detections = [
            DetectionObject(
                object_id=1,
                color="blue_cube",
                bbox=(100, 100, 50, 50),
                confidence=0.9,
                world_position=(0.3, 0.1, 0.0),
                track_id=1,
            ),
            DetectionObject(
                object_id=2,
                color="red_cube",
                bbox=(200, 100, 50, 50),
                confidence=0.85,
                world_position=(0.5, 0.1, 0.2),
                track_id=2,
            ),
        ]
        self.state.update_detections(detections)

        # Claim one object
        self.state.claim_object("blue_cube_track_1", "Robot1")

        # Get available (should only return red cube)
        available = self.state.get_available_objects()

        self.assertEqual(len(available), 1)
        self.assertEqual(available[0].color, "red_cube")

    def test_get_available_objects_by_color(self):
        """Test filtering available objects by color"""
        # Add multiple objects
        detections = [
            DetectionObject(
                object_id=1,
                color="blue_cube",
                bbox=(100, 100, 50, 50),
                confidence=0.9,
                world_position=(0.3, 0.1, 0.0),
                track_id=1,
            ),
            DetectionObject(
                object_id=2,
                color="red_cube",
                bbox=(200, 100, 50, 50),
                confidence=0.85,
                world_position=(0.5, 0.1, 0.2),
                track_id=2,
            ),
            DetectionObject(
                object_id=3,
                color="blue_cube",
                bbox=(300, 100, 50, 50),
                confidence=0.88,
                world_position=(0.7, 0.1, 0.0),
                track_id=3,
            ),
        ]
        self.state.update_detections(detections)

        # Get available blue cubes
        available = self.state.get_available_objects(color="blue")

        self.assertEqual(len(available), 2)
        for obj in available:
            self.assertIn("blue", obj.color)

    def test_get_claimed_objects(self):
        """Test getting objects claimed by specific robot"""
        # Add two objects
        detections = [
            DetectionObject(
                object_id=1,
                color="blue_cube",
                bbox=(100, 100, 50, 50),
                confidence=0.9,
                world_position=(0.3, 0.1, 0.0),
                track_id=1,
            ),
            DetectionObject(
                object_id=2,
                color="red_cube",
                bbox=(200, 100, 50, 50),
                confidence=0.85,
                world_position=(0.5, 0.1, 0.2),
                track_id=2,
            ),
        ]
        self.state.update_detections(detections)

        # Robot1 claims blue, Robot2 claims red
        self.state.claim_object("blue_cube_track_1", "Robot1")
        self.state.claim_object("red_cube_track_2", "Robot2")

        # Check Robot1's claims
        robot1_claims = self.state.get_claimed_objects("Robot1")
        self.assertEqual(len(robot1_claims), 1)
        self.assertEqual(robot1_claims[0].color, "blue_cube")

        # Check Robot2's claims
        robot2_claims = self.state.get_claimed_objects("Robot2")
        self.assertEqual(len(robot2_claims), 1)
        self.assertEqual(robot2_claims[0].color, "red_cube")

    def test_cleanup_stale_claims(self):
        """Test stale claims are automatically cleaned up"""
        # Create state with short timeout
        state = SharedVisionState(claim_timeout=0.2)

        # Add and claim object
        det = DetectionObject(
            object_id=1,
            color="blue_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            world_position=(0.3, 0.1, 0.0),
            track_id=1,
        )
        state.update_detections([det])

        object_id = "blue_cube_track_1"
        state.claim_object(object_id, "Robot1")

        # Verify claimed
        self.assertEqual(state.detections[object_id].claimed_by, "Robot1")

        # Wait for timeout
        time.sleep(0.3)

        # Get available objects (triggers cleanup)
        available = state.get_available_objects()

        # Should be available now (claim released)
        self.assertEqual(len(available), 1)
        self.assertIsNone(state.detections[object_id].claimed_by)

    def test_resolve_conflict_closest_robot(self):
        """Test conflict resolution by closest robot"""
        # Add object
        det = DetectionObject(
            object_id=1,
            color="blue_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            world_position=(0.5, 0.1, 0.0),
            track_id=1,
        )
        self.state.update_detections([det])

        object_id = "blue_cube_track_1"

        # Robot1 is closer (0.2m away)
        robot1_pos = (0.5, 0.1, 0.2)
        # Robot2 is farther (0.5m away)
        robot2_pos = (0.5, 0.1, 0.5)

        winner = self.state.resolve_conflict(
            object_id, "Robot1", "Robot2", robot1_pos, robot2_pos
        )

        self.assertEqual(winner, "Robot1")

    def test_resolve_conflict_tie_breaker(self):
        """Test conflict resolution when distances are equal"""
        # Add object
        det = DetectionObject(
            object_id=1,
            color="blue_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            world_position=(0.5, 0.1, 0.0),
            track_id=1,
        )
        self.state.update_detections([det])

        object_id = "blue_cube_track_1"

        # Both robots same distance
        robot1_pos = (0.5, 0.1, 0.2)
        robot2_pos = (0.5, 0.1, 0.2)

        winner = self.state.resolve_conflict(
            object_id, "Robot1", "Robot2", robot1_pos, robot2_pos
        )

        # Should use alphabetical tie-breaker
        self.assertEqual(winner, "Robot1")

    def test_resolve_conflict_existing_claim(self):
        """Test conflict resolution honors existing claims"""
        # Add and claim object
        det = DetectionObject(
            object_id=1,
            color="blue_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            world_position=(0.5, 0.1, 0.0),
            track_id=1,
        )
        self.state.update_detections([det])

        object_id = "blue_cube_track_1"
        self.state.claim_object(object_id, "Robot2")

        # Robot1 is closer, but Robot2 already claimed
        robot1_pos = (0.5, 0.1, 0.2)
        robot2_pos = (0.5, 0.1, 0.5)

        winner = self.state.resolve_conflict(
            object_id, "Robot1", "Robot2", robot1_pos, robot2_pos
        )

        # Should honor existing claim
        self.assertEqual(winner, "Robot2")

    def test_get_stats(self):
        """Test statistics reporting"""
        # Add objects
        detections = [
            DetectionObject(
                object_id=1,
                color="blue_cube",
                bbox=(100, 100, 50, 50),
                confidence=0.9,
                world_position=(0.3, 0.1, 0.0),
                track_id=1,
            ),
            DetectionObject(
                object_id=2,
                color="red_cube",
                bbox=(200, 100, 50, 50),
                confidence=0.85,
                world_position=(0.5, 0.1, 0.2),
                track_id=2,
            ),
        ]
        self.state.update_detections(detections)

        # Claim one
        self.state.claim_object("blue_cube_track_1", "Robot1")

        stats = self.state.get_stats()

        self.assertEqual(stats["total_objects"], 2)
        self.assertEqual(stats["claimed_objects"], 1)
        self.assertEqual(stats["available_objects"], 1)
        self.assertEqual(stats["claim_timeout"], 10.0)

    def test_clear(self):
        """Test clearing all detections and claims"""
        # Add objects
        detections = [
            DetectionObject(
                object_id=1,
                color="blue_cube",
                bbox=(100, 100, 50, 50),
                confidence=0.9,
                world_position=(0.3, 0.1, 0.0),
                track_id=1,
            )
        ]
        self.state.update_detections(detections)

        self.assertEqual(len(self.state.detections), 1)

        # Clear
        self.state.clear()

        self.assertEqual(len(self.state.detections), 0)


class TestSharedVisionStateThreadSafety(unittest.TestCase):
    """Test thread safety of SharedVisionState"""

    def test_concurrent_claims(self):
        """Test multiple robots claiming objects concurrently"""
        state = SharedVisionState()

        # Add 10 objects
        detections = [
            DetectionObject(
                object_id=i,
                color=f"cube_{i}",
                bbox=(100 * i, 100, 50, 50),
                confidence=0.9,
                world_position=(0.3 * i, 0.1, 0.0),
                track_id=i,
            )
            for i in range(1, 11)
        ]
        state.update_detections(detections)

        claim_results = {"Robot1": [], "Robot2": []}

        def claim_objects(robot_id, object_ids):
            """Claim multiple objects"""
            for obj_id in object_ids:
                success = state.claim_object(obj_id, robot_id)
                claim_results[robot_id].append((obj_id, success))

        # Robot1 tries to claim objects 1-5
        robot1_objects = [f"cube_{i}_track_{i}" for i in range(1, 6)]

        # Robot2 tries to claim objects 4-8 (overlap with Robot1)
        robot2_objects = [f"cube_{i}_track_{i}" for i in range(4, 9)]

        # Start concurrent claims
        t1 = threading.Thread(target=claim_objects, args=("Robot1", robot1_objects))
        t2 = threading.Thread(target=claim_objects, args=("Robot2", robot2_objects))

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # Check that only one robot claimed each object
        for i in range(1, 9):
            obj_id = f"cube_{i}_track_{i}"
            if obj_id in state.detections:
                claimed_by = state.detections[obj_id].claimed_by
                if claimed_by:
                    # Count how many robots successfully claimed this object
                    robot1_success = any(
                        r[0] == obj_id and r[1] for r in claim_results["Robot1"]
                    )
                    robot2_success = any(
                        r[0] == obj_id and r[1] for r in claim_results["Robot2"]
                    )

                    # Exactly one should have succeeded
                    self.assertTrue(robot1_success ^ robot2_success)


class TestSharedVisionStateSingleton(unittest.TestCase):
    """Test singleton pattern for SharedVisionState"""

    def test_get_shared_vision_state_singleton(self):
        """Test get_shared_vision_state returns same instance"""
        state1 = get_shared_vision_state()
        state2 = get_shared_vision_state()

        self.assertIs(state1, state2)


if __name__ == "__main__":
    unittest.main()
