import pytest
import time
from operations.WorldState import get_world_state
from config.Robot import (
    CONFIDENCE_DECAY_PER_FRAME,
    STALE_CONFIDENCE_THRESHOLD,
    OBJECT_TTL_SECONDS,
)

# Tolerance for float confidence comparisons. Accumulated IEEE-754 rounding over
# many repeated subtractions can produce errors in the range of 1e-15 to 1e-12,
# so 1e-9 gives a safe margin while keeping tests strict enough to catch real bugs.
_CONFIDENCE_TOL = 1e-9


class TestConfidenceDecay:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.world_state = get_world_state()
        self.world_state.reset()

    def test_confidence_refresh_on_detection(self):
        self.world_state.register_object(
            "obj1", position=(0.1, 0.2, 0.3), confidence=0.5
        )
        self.world_state.decay_object_confidence({"obj1"})
        obj = self.world_state._objects["obj1"]
        assert obj.confidence == 1.0
        assert not obj.stale

    def test_confidence_decay_on_miss(self):
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        for _ in range(5):
            self.world_state.decay_object_confidence(set())
        obj = self.world_state._objects["obj1"]
        expected_confidence = 1.0 - (5 * CONFIDENCE_DECAY_PER_FRAME)
        assert abs(obj.confidence - expected_confidence) <= _CONFIDENCE_TOL

    def test_confidence_cannot_go_negative(self):
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        for _ in range(20):
            self.world_state.decay_object_confidence(set())
        obj = self.world_state._objects["obj1"]
        assert obj.confidence == 0.0

    def test_stale_threshold_marking(self):
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))

        # With defaults: CONFIDENCE_DECAY_PER_FRAME=0.1, STALE_CONFIDENCE_THRESHOLD=0.3
        # Sequence: 1.0 → 0.9 → 0.8 → 0.7 → 0.6 → 0.5 → 0.4 (6 frames)
        # Next: 0.4 → 0.3 (7 frames, at threshold, not stale)
        # Next: 0.3 → 0.2 (8 frames, below threshold, stale)

        for _ in range(6):
            self.world_state.decay_object_confidence(set())

        obj = self.world_state._objects["obj1"]
        assert abs(obj.confidence - 0.4) <= _CONFIDENCE_TOL
        assert not obj.stale, "Should not be stale above threshold"

        self.world_state.decay_object_confidence(set())
        obj = self.world_state._objects["obj1"]
        assert abs(obj.confidence - STALE_CONFIDENCE_THRESHOLD) <= _CONFIDENCE_TOL
        assert not obj.stale, "Should not be stale at threshold (< not <=)"

        self.world_state.decay_object_confidence(set())
        obj = self.world_state._objects["obj1"]
        assert abs(obj.confidence - 0.2) <= _CONFIDENCE_TOL
        assert obj.stale, "Should be stale below threshold"

    def test_ttl_based_deletion(self):
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        obj = self.world_state._objects["obj1"]
        obj.last_seen = time.time() - (OBJECT_TTL_SECONDS + 0.1)
        self.world_state.decay_object_confidence(set())
        assert "obj1" not in self.world_state._objects

    def test_ttl_not_expired_stays(self):
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        obj = self.world_state._objects["obj1"]
        obj.last_seen = time.time() - (OBJECT_TTL_SECONDS * 0.5)
        self.world_state.decay_object_confidence(set())
        assert "obj1" in self.world_state._objects

    def test_flicker_scenario_appears_disappears_reappears(self):
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))

        self.world_state.decay_object_confidence({"obj1"})
        obj = self.world_state._objects["obj1"]
        assert obj.confidence == 1.0

        self.world_state.decay_object_confidence(set())
        self.world_state.decay_object_confidence(set())
        obj = self.world_state._objects["obj1"]
        assert (
            abs(obj.confidence - (1.0 - (2 * CONFIDENCE_DECAY_PER_FRAME)))
            <= _CONFIDENCE_TOL
        )

        self.world_state.decay_object_confidence({"obj1"})
        obj = self.world_state._objects["obj1"]
        assert obj.confidence == 1.0, "Confidence should be refreshed"
        assert not obj.stale, "Should not be stale"

    def test_multiple_objects_independent_decay(self):
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        self.world_state.register_object("obj2", position=(0.4, 0.5, 0.6))

        self.world_state.decay_object_confidence({"obj1"})
        obj1 = self.world_state._objects["obj1"]
        obj2 = self.world_state._objects["obj2"]
        assert obj1.confidence == 1.0
        assert (
            abs(obj2.confidence - (1.0 - CONFIDENCE_DECAY_PER_FRAME)) <= _CONFIDENCE_TOL
        )

        self.world_state.decay_object_confidence({"obj2"})
        obj1 = self.world_state._objects["obj1"]
        obj2 = self.world_state._objects["obj2"]
        assert (
            abs(obj1.confidence - (1.0 - CONFIDENCE_DECAY_PER_FRAME)) <= _CONFIDENCE_TOL
        )
        assert obj2.confidence == 1.0

    def test_exactly_at_stale_threshold(self):
        self.world_state.register_object(
            "obj1", position=(0.1, 0.2, 0.3), confidence=STALE_CONFIDENCE_THRESHOLD
        )
        obj = self.world_state._objects["obj1"]
        obj.stale = obj.confidence < STALE_CONFIDENCE_THRESHOLD
        assert not obj.stale, "Exactly at threshold should not be stale"

        self.world_state.decay_object_confidence(set())
        assert obj.stale, "Below threshold should be stale"

    def test_rapid_updates_preserve_liveness(self):
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        for _ in range(100):
            self.world_state.decay_object_confidence({"obj1"})
        obj = self.world_state._objects["obj1"]
        assert obj.confidence == 1.0
        assert not obj.stale
        assert "obj1" in self.world_state._objects

    def test_last_seen_timestamp_updated(self):
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        initial_last_seen = self.world_state._objects["obj1"].last_seen
        time.sleep(0.1)
        self.world_state.decay_object_confidence({"obj1"})
        new_last_seen = self.world_state._objects["obj1"].last_seen
        assert new_last_seen > initial_last_seen

    def test_empty_seen_set(self):
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        self.world_state.register_object("obj2", position=(0.4, 0.5, 0.6))
        self.world_state.decay_object_confidence(set())
        for obj in self.world_state._objects.values():
            assert (
                abs(obj.confidence - (1.0 - CONFIDENCE_DECAY_PER_FRAME))
                <= _CONFIDENCE_TOL
            )

    def test_all_objects_seen(self):
        self.world_state.register_object("obj1", position=(0.1, 0.2, 0.3))
        self.world_state.register_object("obj2", position=(0.4, 0.5, 0.6))
        self.world_state.decay_object_confidence({"obj1", "obj2"})
        for obj in self.world_state._objects.values():
            assert obj.confidence == 1.0
            assert not obj.stale
