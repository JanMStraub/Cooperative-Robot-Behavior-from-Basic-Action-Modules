#!/usr/bin/env python3
"""
Unit tests for WorldState.py
=============================

Tests world state management including singleton pattern, TTL-based caching,
robot state tracking, object management, and workspace allocation.

Coverage:
- Singleton pattern enforcement
- TTL-based caching with expiration
- Robot state tracking and updates
- Object position and state management
- Workspace allocation and conflicts
- Workspace deadlock detection (NEW)
- Concurrent workspace allocation (NEW)
- Command tracking and cleanup

NOT Covered:
- Real Unity integration (requires Unity running)
- Persistence to disk/database
- Network synchronization of world state

Dependencies:
- Mock world state (no external dependencies)
- Threading for concurrency tests
- pytest fixtures for cleanup

Run tests:
    pytest tests/TestWorldState.py -v

Run with coverage:
    pytest tests/TestWorldState.py --cov=operations.WorldState --cov-report=html
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from operations.WorldState import (
    CachedValue,
    RobotState,
    ObjectState,
    WorldState,
    get_world_state,
)


class TestCachedValue:
    """Test CachedValue TTL expiration"""

    def test_cached_value_fresh(self):
        """Test value is fresh within TTL"""
        cached = CachedValue(value=42, ttl=1.0)

        assert cached.is_valid() is True
        assert cached.get() == 42

    def test_cached_value_expired(self):
        """Test value expires after TTL"""
        cached = CachedValue(value=42, ttl=0.1)
        time.sleep(0.15)  # Wait for expiration

        assert cached.is_valid() is False
        assert cached.get() is None

    def test_cached_value_update(self):
        """Test updating cached value resets TTL"""
        cached = CachedValue(value=42, ttl=0.1)
        time.sleep(0.08)  # Wait but don't expire

        cached = CachedValue(value=100, timestamp=time.time(), ttl=0.1)
        assert cached.is_valid() is True
        assert cached.get() == 100


class TestWorldStateSingleton:
    """Test WorldState singleton pattern"""

    def test_singleton_pattern(self, cleanup_world_state):
        """Test only one WorldState instance exists"""
        instance1 = get_world_state()
        instance2 = get_world_state()

        assert instance1 is instance2
        assert id(instance1) == id(instance2)

    def test_singleton_reset(self, cleanup_world_state):
        """Test singleton can be reset"""
        world_state = get_world_state()
        world_state._robot_states["Robot1"] = RobotState("Robot1")

        world_state.reset()

        assert len(world_state._robot_states) == 0
        assert len(world_state._objects) == 0


class TestRobotStatus:
    """Test robot status queries and caching"""

    @patch('operations.WorldState.check_robot_status')
    def test_get_robot_cached(self, mock_check_status, cleanup_world_state):
        """Test robot status returns cached value within TTL"""
        from operations.Base import OperationResult
        world_state = get_world_state()

        # Mock Unity response - returns OperationResult
        mock_check_status.return_value = OperationResult.success_result({
            "robot_id": "Robot1",
            "position": (0.3, 0.15, 0.1),
            "is_moving": False,
            "gripper_state": "open"
        })

        # First call - should query Unity
        status1 = world_state.get_robot_status("Robot1")
        assert status1 is not None
        assert mock_check_status.call_count == 1

        # Second call within TTL - should use cache
        status2 = world_state.get_robot_status("Robot1")
        assert status2 is not None
        assert mock_check_status.call_count == 1  # No additional call

    @patch('operations.WorldState.ROBOT_STATUS_CACHE_TTL', 0.1)
    @patch('operations.WorldState.check_robot_status')
    def test_get_robot_status_expired(self, mock_check_status, cleanup_world_state):
        """Test robot status re-queries after TTL expiration"""
        world_state = get_world_state()

        # Mock Unity response
        from operations.Base import OperationResult
        mock_check_status.return_value = OperationResult.success_result({
            "robot_id": "Robot1",
            "position": (0.3, 0.15, 0.1),
            "is_moving": False
        })

        # First call
        status1 = world_state.get_robot_status("Robot1")
        assert mock_check_status.call_count == 1

        # Wait for expiration
        time.sleep(0.15)

        # Second call - should query Unity again
        status2 = world_state.get_robot_status("Robot1")
        assert mock_check_status.call_count == 2

    @patch('operations.WorldState.check_robot_status')
    def test_get_robot_status_force_update(self, mock_check_status, cleanup_world_state):
        """Test force_update bypasses cache"""
        world_state = get_world_state()

        from operations.Base import OperationResult
        mock_check_status.return_value = OperationResult.success_result({
            "robot_id": "Robot1",
            "position": (0.3, 0.15, 0.1)
        })

        # First call
        status1 = world_state.get_robot_status("Robot1")
        assert mock_check_status.call_count == 1

        # Force update - should bypass cache
        status2 = world_state.get_robot_status("Robot1", force_refresh=True)
        assert mock_check_status.call_count == 2


class TestRobotPosition:
    """Test robot position extraction"""

    @patch('operations.WorldState.check_robot_status')
    def test_get_robot_position(self, mock_check_status, cleanup_world_state):
        """Test extracting position from robot status"""
        world_state = get_world_state()

        from operations.Base import OperationResult
        mock_check_status.return_value = OperationResult.success_result({
            "robot_id": "Robot1",
            "position": (0.3, 0.15, 0.1)
        })

        position = world_state.get_robot_position("Robot1")
        assert position == (0.3, 0.15, 0.1)

    def test_get_robot_position_from_state(self, cleanup_world_state):
        """Test position from cached robot state"""
        world_state = get_world_state()

        # Manually set robot state
        robot_state = RobotState("Robot1")
        robot_state.position = (0.5, 0.2, 0.15)
        world_state._robot_states["Robot1"] = robot_state

        position = world_state.get_robot_position("Robot1")
        assert position == (0.5, 0.2, 0.15)

    def test_get_robot_position_none(self, cleanup_world_state):
        """Test position returns None when unavailable"""
        world_state = get_world_state()

        # Robot not tracked and no Unity query available
        with patch('operations.StatusOperations.check_robot_status', return_value={"success": False}):
            position = world_state.get_robot_position("UnknownRobot")
            assert position is None


class TestObjectManagement:
    """Test object tracking and manipulation"""

    def test_update_object_position(self, cleanup_world_state):
        """Test updating object position"""
        world_state = get_world_state()

        world_state.update_object_position("cube_01", (0.3, 0.2, 0.1), "red")

        assert "cube_01" in world_state._objects
        obj = world_state._objects["cube_01"]
        assert obj.object_id == "cube_01"
        assert obj.position == (0.3, 0.2, 0.1)
        assert obj.color == "red"
        assert obj.grasped_by is None

    def test_get_object_position(self, cleanup_world_state):
        """Test retrieving object position"""
        world_state = get_world_state()

        world_state.update_object_position("cube_01", (0.3, 0.2, 0.1), "red")
        position = world_state.get_object_position("cube_01")

        assert position == (0.3, 0.2, 0.1)

    def test_get_object_position_not_found(self, cleanup_world_state):
        """Test object not found returns None"""
        world_state = get_world_state()

        position = world_state.get_object_position("nonexistent")
        assert position is None

    def test_mark_object_grasped(self, cleanup_world_state):
        """Test marking object as grasped"""
        world_state = get_world_state()

        world_state.update_object_position("cube_01", (0.3, 0.2, 0.1), "red")
        world_state.mark_object_grasped("cube_01", "Robot1")

        obj = world_state._objects["cube_01"]
        assert obj.grasped_by == "Robot1"

    def test_mark_object_released(self, cleanup_world_state):
        """Test releasing grasped object"""
        world_state = get_world_state()

        world_state.update_object_position("cube_01", (0.3, 0.2, 0.1), "red")
        world_state.mark_object_grasped("cube_01", "Robot1")
        world_state.mark_object_released("cube_01")

        obj = world_state._objects["cube_01"]
        assert obj.grasped_by is None

    def test_get_objects_by_color(self, cleanup_world_state):
        """Test filtering objects by color"""
        world_state = get_world_state()

        world_state.update_object_position("cube_01", (0.3, 0.2, 0.1), "red")
        world_state.update_object_position("cube_02", (0.4, 0.3, 0.1), "blue")
        world_state.update_object_position("cube_03", (0.5, 0.4, 0.1), "red")

        red_objects = world_state.get_objects_by_color("red")

        assert len(red_objects) == 2
        # get_objects_by_color returns a list of ObjectState objects
        assert any(obj.object_id == "cube_01" for obj in red_objects)
        assert any(obj.object_id == "cube_03" for obj in red_objects)


class TestWorkspaceAllocation:
    """Test workspace allocation and release"""

    def test_allocate_workspace(self, cleanup_world_state):
        """Test allocating workspace to robot"""
        world_state = get_world_state()

        result = world_state.allocate_workspace("left_workspace", "Robot1")

        assert result is True
        assert world_state.get_workspace_owner("left_workspace") == "Robot1"

    def test_allocate_workspace_conflict(self, cleanup_world_state):
        """Test workspace allocation conflict"""
        world_state = get_world_state()

        # Robot1 allocates workspace
        world_state.allocate_workspace("left_workspace", "Robot1")

        # Robot2 tries to allocate same workspace
        result = world_state.allocate_workspace("left_workspace", "Robot2")

        assert result is False
        assert world_state.get_workspace_owner("left_workspace") == "Robot1"

    def test_release_workspace(self, cleanup_world_state):
        """Test releasing workspace"""
        world_state = get_world_state()

        world_state.allocate_workspace("left_workspace", "Robot1")
        world_state.release_workspace("left_workspace", "Robot1")

        assert world_state.get_workspace_owner("left_workspace") is None

    def test_release_workspace_wrong_owner(self, cleanup_world_state):
        """Test releasing workspace by non-owner"""
        world_state = get_world_state()

        world_state.allocate_workspace("left_workspace", "Robot1")
        world_state.release_workspace("left_workspace", "Robot2")

        # Should still be owned by Robot1
        assert world_state.get_workspace_owner("left_workspace") == "Robot1"

    def test_concurrent_workspace_allocation(self, cleanup_world_state):
        """Test multiple robots competing for workspaces concurrently"""
        import threading
        world_state = get_world_state()

        results = {"Robot1": None, "Robot2": None, "Robot3": None}
        errors = []

        def allocate_workspace(robot_id, workspace_id):
            try:
                result = world_state.allocate_workspace(workspace_id, robot_id)
                results[robot_id] = result  # type: ignore[assignment]
            except Exception as e:
                errors.append((robot_id, str(e)))

        # All 3 robots try to allocate same workspace simultaneously
        threads = [
            threading.Thread(target=allocate_workspace, args=("Robot1", "shared_zone")),
            threading.Thread(target=allocate_workspace, args=("Robot2", "shared_zone")),
            threading.Thread(target=allocate_workspace, args=("Robot3", "shared_zone")),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=2.0)

        # Exactly one robot should succeed
        successful_allocations = sum(1 for v in results.values() if v is True)
        assert successful_allocations == 1

        # No errors should occur
        assert len(errors) == 0

        # Verify workspace has an owner
        owner = world_state.get_workspace_owner("shared_zone")
        assert owner in ["Robot1", "Robot2", "Robot3"]

    def test_workspace_deadlock_detection(self, cleanup_world_state):
        """Test circular workspace contention: each robot holds what the other needs."""
        world_state = get_world_state()

        # Robot1 allocates left_workspace, Robot2 allocates right_workspace
        world_state.allocate_workspace("left_workspace", "Robot1")
        world_state.allocate_workspace("right_workspace", "Robot2")

        # Robot1 tries to get right_workspace (owned by Robot2) - must fail
        result1 = world_state.allocate_workspace("right_workspace", "Robot1")
        assert result1 is False

        # Robot2 tries to get left_workspace (owned by Robot1) - must fail
        result2 = world_state.allocate_workspace("left_workspace", "Robot2")
        assert result2 is False

        # Both robots still hold their original workspaces
        assert world_state.get_workspace_owner("left_workspace") == "Robot1"
        assert world_state.get_workspace_owner("right_workspace") == "Robot2"

    def test_workspace_automatic_release_on_failure(self, cleanup_world_state):
        """Test workspace is released when robot operation fails"""
        world_state = get_world_state()

        # Simulate robot allocating workspace for operation
        world_state.allocate_workspace("center", "Robot1")
        assert world_state.get_workspace_owner("center") == "Robot1"

        # Simulate operation failure and cleanup
        world_state.release_workspace("center", "Robot1")

        # Workspace should now be free
        assert world_state.get_workspace_owner("center") is None

        # Another robot can now allocate it
        result = world_state.allocate_workspace("center", "Robot2")
        assert result is True

    def test_workspace_priority_queue(self, cleanup_world_state):
        """Test workspace re-allocation after release (first-come-first-served)"""
        world_state = get_world_state()

        # Robot1 holds shared_zone
        world_state.allocate_workspace("shared_zone", "Robot1")

        # Other robots cannot allocate the same region
        result2 = world_state.allocate_workspace("shared_zone", "Robot2")
        result3 = world_state.allocate_workspace("shared_zone", "Robot3")
        result4 = world_state.allocate_workspace("shared_zone", "Robot4")

        assert result2 is False
        assert result3 is False
        assert result4 is False

        # Robot1 releases workspace
        world_state.release_workspace("shared_zone", "Robot1")

        # Now Robot2 can allocate
        result_next = world_state.allocate_workspace("shared_zone", "Robot2")
        assert result_next is True
        assert world_state.get_workspace_owner("shared_zone") == "Robot2"

    def test_multiple_workspace_allocation_per_robot(self, cleanup_world_state):
        """Test single robot allocating multiple workspaces"""
        world_state = get_world_state()

        # Robot1 allocates all available regions for a complex operation
        result1 = world_state.allocate_workspace("left_workspace", "Robot1")
        result2 = world_state.allocate_workspace("center", "Robot1")
        result3 = world_state.allocate_workspace("right_workspace", "Robot1")

        assert result1 is True
        assert result2 is True
        assert result3 is True

        # Verify all owned by Robot1
        assert world_state.get_workspace_owner("left_workspace") == "Robot1"
        assert world_state.get_workspace_owner("center") == "Robot1"
        assert world_state.get_workspace_owner("right_workspace") == "Robot1"

        # Release all workspaces
        world_state.release_workspace("left_workspace", "Robot1")
        world_state.release_workspace("center", "Robot1")
        world_state.release_workspace("right_workspace", "Robot1")

        # All should be free
        assert world_state.get_workspace_owner("left_workspace") is None
        assert world_state.get_workspace_owner("center") is None
        assert world_state.get_workspace_owner("right_workspace") is None

    def test_workspace_allocation_stress_test(self, cleanup_world_state):
        """Stress test with 15 robots competing for the 4 available workspace regions"""
        import threading
        world_state = get_world_state()

        workspace_regions = ["left_workspace", "right_workspace", "shared_zone", "center"]
        num_robots = 15
        results = {f"Robot{i}": [] for i in range(num_robots)}
        errors = []

        def robot_workflow(robot_id):
            """Simulate robot trying to allocate workspaces"""
            try:
                for workspace_id in workspace_regions:
                    result = world_state.allocate_workspace(workspace_id, robot_id)
                    results[robot_id].append((workspace_id, result))

                    if result:
                        time.sleep(0.01)
                        world_state.release_workspace(workspace_id, robot_id)
            except Exception as e:
                errors.append((robot_id, str(e)))

        threads = [
            threading.Thread(target=robot_workflow, args=(f"Robot{i}",))
            for i in range(num_robots)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0

        # Verify some robots got some workspaces
        successful_allocations = sum(
            sum(1 for _, success in robot_results if success)
            for robot_results in results.values()
        )
        assert successful_allocations > 0

        # All workspaces should be free at the end
        for workspace_id in workspace_regions:
            owner = world_state.get_workspace_owner(workspace_id)
            assert owner is None or owner.startswith("Robot")


class TestCommandTracking:
    """Test command registration and tracking"""

    def test_register_command(self, cleanup_world_state):
        """Test registering in-flight command"""
        world_state = get_world_state()

        world_state.register_command(1, {"operation": "move_to_coordinate", "robot_id": "Robot1", "params": {"x": 0.3, "y": 0.2, "z": 0.1}})

        assert 1 in world_state._pending_commands
        cmd = world_state._pending_commands[1]
        assert cmd["command"]["robot_id"] == "Robot1"
        assert cmd["command"]["operation"] == "move_to_coordinate"
        assert cmd["status"] == "pending"

    def test_update_command_status(self, cleanup_world_state):
        """Test updating command status"""
        world_state = get_world_state()

        world_state.register_command(1, {"operation": "move_to_coordinate", "robot_id": "Robot1", "params": {}})
        world_state.update_command_status(1, "completed")

        cmd = world_state._pending_commands[1]
        assert cmd["status"] == "completed"

    def test_cleanup_old_commands(self, cleanup_world_state):
        """Test removing old commands"""
        world_state = get_world_state()

        # Register command and mark as completed
        world_state.register_command(1, {"operation": "move_to_coordinate", "robot_id": "Robot1", "params": {}})
        world_state.update_command_status(1, "completed")

        # Mark command as completed with old timestamp
        world_state._pending_commands[1]["completion_time"] = time.time() - 400  # 400s ago

        # Cleanup (default keeps commands for 300s)
        world_state.cleanup_old_commands(max_age_seconds=300)

        assert 1 not in world_state._pending_commands


class TestCacheManagement:
    """Test cache invalidation and reset"""

    @patch('operations.WorldState.check_robot_status')
    def test_clear_cache(self, mock_check_status, cleanup_world_state):
        """Test cache invalidation"""
        world_state = get_world_state()

        from operations.Base import OperationResult
        mock_check_status.return_value = OperationResult.success_result({
            "robot_id": "Robot1",
            "position": (0.3, 0.15, 0.1)
        })

        # Populate cache
        world_state.get_robot_status("Robot1")
        assert len(world_state._robot_cache) == 1

        # Clear cache
        world_state.clear_cache()
        assert len(world_state._robot_cache) == 0

    def test_reset(self, cleanup_world_state):
        """Test full state reset."""
        world_state = get_world_state()

        # Populate state
        world_state.update_object_position("cube_01", (0.3, 0.2, 0.1), "red")
        world_state.allocate_workspace("left_workspace", "Robot1")
        world_state.register_command(1, {"operation": "move", "robot_id": "Robot1", "params": {}})
        robot_state = RobotState("Robot1")
        world_state._robot_states["Robot1"] = robot_state

        # Reset
        world_state.reset()

        assert len(world_state._robot_states) == 0
        assert len(world_state._objects) == 0
        # workspace_allocations has region keys but all values should be None
        assert all(owner is None for owner in world_state._workspace_allocations.values())
        assert len(world_state._pending_commands) == 0
        assert len(world_state._robot_cache) == 0
