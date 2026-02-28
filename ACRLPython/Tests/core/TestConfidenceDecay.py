"""
Test suite for object confidence decay and liveness tracking.

Tests the WorldState confidence decay mechanism, including:
- Confidence decay when objects disappear from detection
- TTL-based cleanup of stale objects
- Flicker scenarios (objects appearing/disappearing/reappearing)
- Staleness threshold marking
"""

import math
import unittest
import time
from operations.WorldState import WorldState, get_world_state
from config.Robot import (
    CONFIDENCE_DECAY_PER_FRAME,
    STALE_CONFIDENCE_THRESHOLD,
    OBJECT_TTL_SECONDS,
)

# Tolerance for float confidence comparisons. Accumulated IEEE-754 rounding over
# many repeated subtractions can produce errors in the range of 1e-15 to 1e-12,
# so 1e-9 gives a safe margin while keeping tests strict enough to catch real bugs.
_CONFIDENCE_TOL = 1e-9


class TestConfidenceDecay(unittest.TestCase):
    """Test confidence decay and liveness tracking."""

    def setUp(self):
        """Reset world state before each test."""
        self.world_state = get_world_state()
        self.world_state.reset()

    def test_confidence_refresh_on_detection(self):
        """Test that detected objects have confidence refreshed to 1.0."""
        # Register an object with low confidence
        self.world_state.register_object(
            "obj1", position=(0.1, 0.2, 0.3), confidence=0.5
        )

        # Simulate detection frame with object present
        self.world_state.decay_object_confidence({"obj1"})

        # Verify confidence is refreshed
        obj = self.world_state._objects["obj1"]
        self.assertEqual(obj.confidence, 1.0)
        self.assertFalse(obj.stale)

    def test_confidence_decay_on_miss(self):
        """Test that missing objects have confidence decayed."""
        # Register an object
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))

        # Simulate 5 detection frames without seeing object
        for _ in range(5):
            self.world_state.decay_object_confidence(set())

        # Verify confidence decayed by 5 * CONFIDENCE_DECAY_PER_FRAME
        obj = self.world_state._objects["obj1"]
        expected_confidence = 1.0 - (5 * CONFIDENCE_DECAY_PER_FRAME)
        self.assertAlmostEqual(obj.confidence, expected_confidence, delta=_CONFIDENCE_TOL)

    def test_confidence_cannot_go_negative(self):
        """Test that confidence is clamped to 0.0."""
        # Register an object
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))

        # Simulate many frames (more than 1.0 / CONFIDENCE_DECAY_PER_FRAME)
        for _ in range(20):
            self.world_state.decay_object_confidence(set())

        # Verify confidence is 0.0, not negative
        obj = self.world_state._objects["obj1"]
        self.assertEqual(obj.confidence, 0.0)

    def test_stale_threshold_marking(self):
        """Test that objects are marked stale when confidence drops below threshold."""
        # Register an object
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))

        # With defaults: CONFIDENCE_DECAY_PER_FRAME=0.1, STALE_CONFIDENCE_THRESHOLD=0.3
        # Sequence: 1.0 → 0.9 → 0.8 → 0.7 → 0.6 → 0.5 → 0.4 (6 frames)
        # Next: 0.4 → 0.3 (7 frames, at threshold, not stale)
        # Next: 0.3 → 0.2 (8 frames, below threshold, stale)

        # Decay to just above threshold (6 frames)
        for _ in range(6):
            self.world_state.decay_object_confidence(set())

        obj = self.world_state._objects["obj1"]
        self.assertAlmostEqual(obj.confidence, 0.4, delta=_CONFIDENCE_TOL)
        self.assertFalse(obj.stale, "Should not be stale above threshold")

        # One more frame brings us to threshold (7 frames total)
        self.world_state.decay_object_confidence(set())
        obj = self.world_state._objects["obj1"]
        self.assertAlmostEqual(obj.confidence, STALE_CONFIDENCE_THRESHOLD, delta=_CONFIDENCE_TOL)
        self.assertFalse(obj.stale, "Should not be stale at threshold (< not <=)")

        # One more frame makes it stale (8 frames total)
        self.world_state.decay_object_confidence(set())
        obj = self.world_state._objects["obj1"]
        self.assertAlmostEqual(obj.confidence, 0.2, delta=_CONFIDENCE_TOL)
        self.assertTrue(obj.stale, "Should be stale below threshold")

    def test_ttl_based_deletion(self):
        """Test that objects are deleted after TTL expires."""
        # Register an object
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))

        # Set last_seen to past (beyond TTL)
        obj = self.world_state._objects["obj1"]
        obj.last_seen = time.time() - (OBJECT_TTL_SECONDS + 0.1)

        # Trigger decay
        self.world_state.decay_object_confidence(set())

        # Verify object was deleted
        self.assertNotIn("obj1", self.world_state._objects)

    def test_ttl_not_expired_stays(self):
        """Test that objects within TTL are not deleted."""
        # Register an object
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))

        # Set last_seen to recent (within TTL)
        obj = self.world_state._objects["obj1"]
        obj.last_seen = time.time() - (OBJECT_TTL_SECONDS * 0.5)

        # Trigger decay
        self.world_state.decay_object_confidence(set())

        # Verify object still exists
        self.assertIn("obj1", self.world_state._objects)

    def test_flicker_scenario_appears_disappears_reappears(self):
        """Test object that flickers in and out of detection."""
        # Register object
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))

        # Appears (frame 1)
        self.world_state.decay_object_confidence({"obj1"})
        obj = self.world_state._objects["obj1"]
        self.assertEqual(obj.confidence, 1.0)

        # Disappears (frame 2-3)
        self.world_state.decay_object_confidence(set())
        self.world_state.decay_object_confidence(set())
        obj = self.world_state._objects["obj1"]
        self.assertAlmostEqual(obj.confidence, 1.0 - (2 * CONFIDENCE_DECAY_PER_FRAME))

        # Reappears (frame 4)
        self.world_state.decay_object_confidence({"obj1"})
        obj = self.world_state._objects["obj1"]
        self.assertEqual(obj.confidence, 1.0, "Confidence should be refreshed")
        self.assertFalse(obj.stale, "Should not be stale")

    def test_multiple_objects_independent_decay(self):
        """Test that multiple objects decay independently."""
        # Register two objects
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        self.world_state.register_object("obj2", position=(0.4, 0.5, 0.6))

        # Frame 1: Only obj1 seen
        self.world_state.decay_object_confidence({"obj1"})
        obj1 = self.world_state._objects["obj1"]
        obj2 = self.world_state._objects["obj2"]
        self.assertEqual(obj1.confidence, 1.0)
        self.assertAlmostEqual(obj2.confidence, 1.0 - CONFIDENCE_DECAY_PER_FRAME)

        # Frame 2: Only obj2 seen
        self.world_state.decay_object_confidence({"obj2"})
        obj1 = self.world_state._objects["obj1"]
        obj2 = self.world_state._objects["obj2"]
        self.assertAlmostEqual(obj1.confidence, 1.0 - CONFIDENCE_DECAY_PER_FRAME)
        self.assertEqual(obj2.confidence, 1.0)

    def test_exactly_at_stale_threshold(self):
        """Test behavior when confidence is exactly at stale threshold."""
        # Register object with confidence exactly at threshold
        self.world_state.register_object(
            "obj1", position=(0.1, 0.2, 0.3), confidence=STALE_CONFIDENCE_THRESHOLD
        )

        # Object should not be stale yet (threshold is exclusive)
        obj = self.world_state._objects["obj1"]
        obj.stale = obj.confidence < STALE_CONFIDENCE_THRESHOLD
        self.assertFalse(obj.stale, "Exactly at threshold should not be stale")

        # One decay should make it stale
        self.world_state.decay_object_confidence(set())
        self.assertTrue(obj.stale, "Below threshold should be stale")

    def test_rapid_updates_preserve_liveness(self):
        """Test that rapid detection updates keep object alive."""
        # Register object
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))

        # Simulate rapid updates (every frame sees object)
        for _ in range(100):
            self.world_state.decay_object_confidence({"obj1"})

        # Verify object is still alive and confident
        obj = self.world_state._objects["obj1"]
        self.assertEqual(obj.confidence, 1.0)
        self.assertFalse(obj.stale)
        self.assertIn("obj1", self.world_state._objects)

    def test_last_seen_timestamp_updated(self):
        """Test that last_seen timestamp is updated on detection."""
        # Register object
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))

        # Get initial last_seen
        initial_last_seen = self.world_state._objects["obj1"].last_seen

        # Wait a bit
        time.sleep(0.1)

        # Trigger detection
        self.world_state.decay_object_confidence({"obj1"})

        # Verify last_seen was updated
        new_last_seen = self.world_state._objects["obj1"].last_seen
        self.assertGreater(new_last_seen, initial_last_seen)

    def test_empty_seen_set(self):
        """Test decay with empty seen set (no detections)."""
        # Register multiple objects
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        self.world_state.register_object("obj2", position=(0.4, 0.5, 0.6))

        # Trigger decay with no detections
        self.world_state.decay_object_confidence(set())

        # Verify all objects decayed
        for obj in self.world_state._objects.values():
            self.assertAlmostEqual(obj.confidence, 1.0 - CONFIDENCE_DECAY_PER_FRAME)

    def test_all_objects_seen(self):
        """Test decay when all objects are detected."""
        # Register multiple objects
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        self.world_state.register_object("obj2", position=(0.4, 0.5, 0.6))

        # Trigger decay with all objects seen
        self.world_state.decay_object_confidence({"obj1", "obj2"})

        # Verify all objects refreshed
        for obj in self.world_state._objects.values():
            self.assertEqual(obj.confidence, 1.0)
            self.assertFalse(obj.stale)


if __name__ == "__main__":
    unittest.main()
