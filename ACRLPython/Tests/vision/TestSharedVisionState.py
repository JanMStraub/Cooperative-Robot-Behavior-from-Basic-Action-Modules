import time
import threading

from operations.SharedVisionState import (
    SharedVisionState,
    ClaimedObject,
    get_shared_vision_state,
)
from vision.DetectionDataModels import DetectionObject


class TestClaimedObject:

    def test_claimed_object_creation(self):
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

        assert obj.object_id == "blue_cube_1"
        assert obj.color == "blue_cube"
        assert obj.world_position == (0.3, 0.1, 0.0)
        assert obj.claimed_by == "Robot1"
        assert obj.track_id == 5
        assert obj.confidence == 0.95
        assert obj.depth_m == 0.8

    def test_claimed_object_defaults(self):
        obj = ClaimedObject(
            object_id="test", color="red", world_position=(0.0, 0.0, 0.0)
        )

        assert obj.claimed_by is None
        assert obj.claim_timestamp == 0.0
        assert obj.track_id is None
        assert obj.confidence == 1.0
        assert obj.depth_m is None


class TestSharedVisionState:

    def setup_method(self):
        self.state = SharedVisionState(claim_timeout=10.0)

    def teardown_method(self):
        self.state.clear()

    def test_initialization(self):
        assert len(self.state.detections) == 0
        assert self.state.claim_timeout == 10.0
        assert self.state.lock is not None

    def test_update_detections_new_objects(self):
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

        assert len(self.state.detections) == 2
        obj_ids = list(self.state.detections.keys())
        assert "blue_cube_track_1" in obj_ids
        assert "red_cube_track_2" in obj_ids

    def test_update_detections_preserves_claims(self):
        det1 = DetectionObject(
            object_id=1,
            color="blue_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            world_position=(0.3, 0.1, 0.0),
            track_id=1,
        )
        self.state.update_detections([det1])

        object_id = "blue_cube_track_1"
        self.state.claim_object(object_id, "Robot1")

        det2 = DetectionObject(
            object_id=2,
            color="blue_cube",
            bbox=(105, 102, 50, 50),
            confidence=0.95,
            world_position=(0.32, 0.12, 0.0),
            track_id=1,
        )
        self.state.update_detections([det2])

        obj = self.state.detections[object_id]
        assert obj.claimed_by == "Robot1"
        assert obj.world_position == (0.32, 0.12, 0.0)

    def test_claim_object_success(self):
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
        success = self.state.claim_object(object_id, "Robot1")

        assert success
        obj = self.state.detections[object_id]
        assert obj.claimed_by == "Robot1"
        assert obj.claim_timestamp > 0

    def test_claim_object_already_claimed(self):
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

        success = self.state.claim_object(object_id, "Robot2")

        assert not success
        obj = self.state.detections[object_id]
        assert obj.claimed_by == "Robot1"

    def test_claim_object_refresh_timestamp(self):
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

        time.sleep(0.1)

        success = self.state.claim_object(object_id, "Robot1")

        assert success
        second_timestamp = self.state.detections[object_id].claim_timestamp
        assert second_timestamp > first_timestamp

    def test_release_object_success(self):
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

        success = self.state.release_object(object_id, "Robot1")

        assert success
        obj = self.state.detections[object_id]
        assert obj.claimed_by is None
        assert obj.claim_timestamp == 0.0

    def test_release_object_wrong_robot(self):
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

        success = self.state.release_object(object_id, "Robot2")

        assert not success
        obj = self.state.detections[object_id]
        assert obj.claimed_by == "Robot1"

    def test_get_available_objects_all(self):
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

        self.state.claim_object("blue_cube_track_1", "Robot1")

        available = self.state.get_available_objects()

        assert len(available) == 1
        assert available[0].color == "red_cube"

    def test_get_available_objects_by_color(self):
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

        available = self.state.get_available_objects(color="blue")

        assert len(available) == 2
        for obj in available:
            assert "blue" in obj.color

    def test_get_claimed_objects(self):
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

        self.state.claim_object("blue_cube_track_1", "Robot1")
        self.state.claim_object("red_cube_track_2", "Robot2")

        robot1_claims = self.state.get_claimed_objects("Robot1")
        assert len(robot1_claims) == 1
        assert robot1_claims[0].color == "blue_cube"

        robot2_claims = self.state.get_claimed_objects("Robot2")
        assert len(robot2_claims) == 1
        assert robot2_claims[0].color == "red_cube"

    def test_cleanup_stale_claims(self):
        state = SharedVisionState(claim_timeout=0.2)

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

        assert state.detections[object_id].claimed_by == "Robot1"

        time.sleep(0.3)

        available = state.get_available_objects()

        assert len(available) == 1
        assert state.detections[object_id].claimed_by is None

    def test_resolve_conflict_closest_robot(self):
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
        robot1_pos = (0.5, 0.1, 0.2)
        robot2_pos = (0.5, 0.1, 0.5)

        winner = self.state.resolve_conflict(
            object_id, "Robot1", "Robot2", robot1_pos, robot2_pos
        )

        assert winner == "Robot1"

    def test_resolve_conflict_tie_breaker(self):
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
        robot1_pos = (0.5, 0.1, 0.2)
        robot2_pos = (0.5, 0.1, 0.2)

        winner = self.state.resolve_conflict(
            object_id, "Robot1", "Robot2", robot1_pos, robot2_pos
        )

        assert winner == "Robot1"

    def test_resolve_conflict_existing_claim(self):
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

        robot1_pos = (0.5, 0.1, 0.2)
        robot2_pos = (0.5, 0.1, 0.5)

        winner = self.state.resolve_conflict(
            object_id, "Robot1", "Robot2", robot1_pos, robot2_pos
        )

        assert winner == "Robot2"

    def test_get_stats(self):
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

        self.state.claim_object("blue_cube_track_1", "Robot1")

        stats = self.state.get_stats()

        assert stats["total_objects"] == 2
        assert stats["claimed_objects"] == 1
        assert stats["available_objects"] == 1
        assert stats["claim_timeout"] == 10.0

    def test_clear(self):
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

        assert len(self.state.detections) == 1

        self.state.clear()

        assert len(self.state.detections) == 0


class TestSharedVisionStateThreadSafety:

    def test_concurrent_claims(self):
        state = SharedVisionState()

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
            for obj_id in object_ids:
                success = state.claim_object(obj_id, robot_id)
                claim_results[robot_id].append((obj_id, success))

        robot1_objects = [f"cube_{i}_track_{i}" for i in range(1, 6)]
        robot2_objects = [f"cube_{i}_track_{i}" for i in range(4, 9)]

        t1 = threading.Thread(target=claim_objects, args=("Robot1", robot1_objects))
        t2 = threading.Thread(target=claim_objects, args=("Robot2", robot2_objects))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        for i in range(1, 9):
            obj_id = f"cube_{i}_track_{i}"
            if obj_id in state.detections:
                claimed_by = state.detections[obj_id].claimed_by
                if claimed_by:
                    robot1_success = any(
                        r[0] == obj_id and r[1] for r in claim_results["Robot1"]
                    )
                    robot2_success = any(
                        r[0] == obj_id and r[1] for r in claim_results["Robot2"]
                    )
                    assert robot1_success ^ robot2_success


class TestSharedVisionStateSingleton:

    def test_get_shared_vision_state_singleton(self):
        state1 = get_shared_vision_state()
        state2 = get_shared_vision_state()

        assert state1 is state2
