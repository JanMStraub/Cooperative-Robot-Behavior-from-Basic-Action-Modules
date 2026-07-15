from operations.VisionOperations import resolve_held_object_xz


class TestResolveHeldObjectXZ:
    def test_no_existing_state_uses_fresh(self):
        x, z = resolve_held_object_xz(1.0, 0.03, 2.0, "Robot2", None)
        assert (x, z) == (1.0, 2.0)

    def test_not_held_uses_fresh(self):
        existing = {"grasped_by": None, "position": {"x": 9.0, "y": 0.03, "z": 9.0}}
        x, z = resolve_held_object_xz(1.0, 0.03, 2.0, "Robot2", existing)
        assert (x, z) == (1.0, 2.0)

    def test_held_by_requester_uses_fresh(self):
        existing = {
            "grasped_by": "Robot2",
            "position": {"x": 9.0, "y": 0.03, "z": 9.0},
        }
        x, z = resolve_held_object_xz(1.0, 0.03, 2.0, "Robot2", existing)
        assert (x, z) == (1.0, 2.0)

    def test_held_by_other_robot_unchanged_height_uses_cached(self):
        existing = {
            "grasped_by": "Robot1",
            "position": {"x": 9.0, "y": 0.028, "z": 9.0},
        }
        x, z = resolve_held_object_xz(1.0, 0.03, 2.0, "Robot2", existing)
        assert (x, z) == (9.0, 9.0)

    def test_held_by_other_robot_changed_height_uses_fresh(self):
        existing = {
            "grasped_by": "Robot1",
            "position": {"x": 9.0, "y": 0.028, "z": 9.0},
        }
        x, z = resolve_held_object_xz(1.0, 0.35, 2.0, "Robot2", existing)
        assert (x, z) == (1.0, 2.0)
