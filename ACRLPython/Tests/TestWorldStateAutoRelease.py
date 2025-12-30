#!/usr/bin/env python3
"""
Unit tests for WorldState workspace auto-release (Phase 2 improvement).

Tests the automatic release of stale workspace allocations after timeout period.
Validates that robots don't get indefinitely blocked by expired allocations.

Note: WorldState workspace API uses (region, robot_id) parameter order.
"""

import pytest
import time
from unittest.mock import Mock, patch
from operations.WorldState import (
    WorldState,
    get_world_state,
    WorkspaceAllocation,
)


class TestWorkspaceAutoRelease:
    """Test workspace auto-release with timeout"""

    def test_workspace_allocation_structure(self, cleanup_world_state):
        """Test WorkspaceAllocation dataclass has required fields"""
        allocation = WorkspaceAllocation(
            robot_id="Robot1",
            region="left_workspace"
        )

        assert allocation.robot_id == "Robot1"
        assert allocation.region == "left_workspace"
        assert hasattr(allocation, 'allocated_at')
        assert isinstance(allocation.allocated_at, float)

    def test_allocate_workspace_fresh(self, cleanup_world_state):
        """Test allocating a fresh workspace"""
        world_state = get_world_state()

        # Note: WorldState uses (region, robot_id) parameter order
        result = world_state.allocate_workspace("left_workspace", "Robot1")

        assert result is True
        allocated_robot = world_state.get_workspace_owner("left_workspace")
        assert allocated_robot == "Robot1"

    def test_allocate_workspace_already_owned(self, cleanup_world_state):
        """Test allocating a workspace already owned by the same robot"""
        world_state = get_world_state()

        # First allocation
        world_state.allocate_workspace("left_workspace", "Robot1")

        # Second allocation by same robot
        result = world_state.allocate_workspace("left_workspace", "Robot1")

        assert result is True  # Should succeed (same owner)

    def test_allocate_workspace_already_owned_different_robot(self, cleanup_world_state):
        """Test allocating a workspace owned by another robot"""
        world_state = get_world_state()

        # Robot1 allocates workspace
        world_state.allocate_workspace("left_workspace", "Robot1")

        # Robot2 tries to allocate same workspace
        result = world_state.allocate_workspace("left_workspace", "Robot2")

        assert result is False  # Should fail
        assert world_state.get_workspace_owner("left_workspace") == "Robot1"

    def test_release_workspace(self, cleanup_world_state):
        """Test manual workspace release"""
        world_state = get_world_state()

        # Allocate workspace
        world_state.allocate_workspace("left_workspace", "Robot1")
        assert world_state.get_workspace_owner("left_workspace") == "Robot1"

        # Release workspace
        world_state.release_workspace("left_workspace", "Robot1")
        assert world_state.get_workspace_owner("left_workspace") is None

    def test_cleanup_stale_allocations_timeout(self, cleanup_world_state):
        """Test stale allocations are cleaned up after timeout"""
        world_state = get_world_state()

        # Set short timeout for testing (1 second)
        world_state._workspace_timeout = 1.0

        # Allocate workspace
        world_state.allocate_workspace("left_workspace", "Robot1")
        assert world_state.get_workspace_owner("left_workspace") == "Robot1"

        # Wait for timeout
        time.sleep(1.1)

        # Trigger cleanup
        world_state._cleanup_stale_allocations()

        # Verify workspace was released
        assert world_state.get_workspace_owner("left_workspace") is None

    def test_cleanup_preserves_fresh_allocations(self, cleanup_world_state):
        """Test cleanup preserves fresh allocations"""
        world_state = get_world_state()

        # Set timeout
        world_state._workspace_timeout = 1.0

        # Allocate workspace
        world_state.allocate_workspace("left_workspace", "Robot1")

        # Immediately trigger cleanup (allocation is fresh)
        world_state._cleanup_stale_allocations()

        # Verify workspace is still allocated
        assert world_state.get_workspace_owner("left_workspace") == "Robot1"

    def test_cleanup_multiple_workspaces_partial_stale(self, cleanup_world_state):
        """Test cleanup handles mix of fresh and stale allocations"""
        world_state = get_world_state()

        # Set short timeout
        world_state._workspace_timeout = 0.5

        # Allocate workspaces
        world_state.allocate_workspace("left_workspace", "Robot1")
        world_state.allocate_workspace("right_workspace", "Robot2")

        # Wait for first two to become stale
        time.sleep(0.6)

        # Allocate third workspace (fresh)
        world_state.allocate_workspace("shared_zone", "Robot3")

        # Trigger cleanup
        world_state._cleanup_stale_allocations()

        # Verify stale allocations released, fresh preserved
        assert world_state.get_workspace_owner("left_workspace") is None
        assert world_state.get_workspace_owner("right_workspace") is None
        assert world_state.get_workspace_owner("shared_zone") == "Robot3"

    def test_default_timeout_value(self, cleanup_world_state):
        """Test default workspace timeout is 60 seconds"""
        world_state = get_world_state()

        # Default timeout should be 60 seconds
        assert hasattr(world_state, '_workspace_timeout')
        assert world_state._workspace_timeout == 60.0

    def test_configurable_timeout(self, cleanup_world_state):
        """Test workspace timeout can be configured"""
        world_state = get_world_state()

        # Set custom timeout
        world_state._workspace_timeout = 30.0

        assert world_state._workspace_timeout == 30.0

    def test_allocation_timestamp_updates(self, cleanup_world_state):
        """Test allocation timestamp updates on re-allocation"""
        world_state = get_world_state()

        # First allocation
        world_state.allocate_workspace("left_workspace", "Robot1")
        first_allocation = world_state._workspace_allocations.get("left_workspace")
        first_timestamp = first_allocation.allocated_at if first_allocation else 0

        # Wait briefly
        time.sleep(0.1)

        # Re-allocate same workspace to same robot
        world_state.allocate_workspace("left_workspace", "Robot1")
        second_allocation = world_state._workspace_allocations.get("left_workspace")
        second_timestamp = second_allocation.allocated_at if second_allocation else 0

        # Timestamps should be different (updated)
        assert second_timestamp > first_timestamp

    def test_cleanup_called_periodically(self, cleanup_world_state):
        """Test cleanup is called automatically during operations"""
        world_state = get_world_state()

        # Set short timeout
        world_state._workspace_timeout = 0.5

        # Allocate workspace
        world_state.allocate_workspace("left_workspace", "Robot1")

        # Wait for timeout
        time.sleep(0.6)

        # Trigger any operation that calls cleanup internally
        # (allocate_workspace calls _cleanup_stale_allocations)
        world_state.allocate_workspace("right_workspace", "Robot2")

        # Verify stale allocation was cleaned up
        assert world_state.get_workspace_owner("left_workspace") is None
        assert world_state.get_workspace_owner("right_workspace") == "Robot2"

    def test_workspace_regions_independent(self, cleanup_world_state):
        """Test different workspace regions are tracked independently"""
        world_state = get_world_state()

        # Allocate different regions
        world_state.allocate_workspace("left_workspace", "Robot1")
        world_state.allocate_workspace("right_workspace", "Robot2")
        world_state.allocate_workspace("shared_zone", "Robot3")

        # Verify all are allocated independently
        assert world_state.get_workspace_owner("left_workspace") == "Robot1"
        assert world_state.get_workspace_owner("right_workspace") == "Robot2"
        assert world_state.get_workspace_owner("shared_zone") == "Robot3"

        # Release one
        world_state.release_workspace("right_workspace", "Robot2")

        # Verify only released region is freed
        assert world_state.get_workspace_owner("left_workspace") == "Robot1"
        assert world_state.get_workspace_owner("right_workspace") is None
        assert world_state.get_workspace_owner("shared_zone") == "Robot3"

    def test_reset_clears_workspaces(self, cleanup_world_state):
        """Test reset clears all workspace allocations"""
        world_state = get_world_state()

        # Allocate workspaces
        world_state.allocate_workspace("left_workspace", "Robot1")
        world_state.allocate_workspace("right_workspace", "Robot2")

        # Reset
        world_state.reset()

        # Verify all workspaces cleared
        assert world_state.get_workspace_owner("left_workspace") is None
        assert world_state.get_workspace_owner("right_workspace") is None

    def test_stale_allocation_warning_logged(self, cleanup_world_state, caplog):
        """Test that stale allocation cleanup logs a warning"""
        import logging
        world_state = get_world_state()

        # Set short timeout
        world_state._workspace_timeout = 0.5

        # Allocate workspace
        world_state.allocate_workspace("left_workspace", "Robot1")

        # Wait for timeout
        time.sleep(0.6)

        # Trigger cleanup with logging
        with caplog.at_level(logging.WARNING):
            world_state._cleanup_stale_allocations()

        # Verify warning was logged
        assert any("Auto-releasing stale allocation" in record.message
                   for record in caplog.records)

    def test_multiple_robots_same_region_sequential(self, cleanup_world_state):
        """Test multiple robots can use same region sequentially after timeout"""
        world_state = get_world_state()

        # Set short timeout
        world_state._workspace_timeout = 0.5

        # Robot1 allocates
        result1 = world_state.allocate_workspace("shared_zone", "Robot1")
        assert result1 is True
        assert world_state.get_workspace_owner("shared_zone") == "Robot1"

        # Wait for timeout
        time.sleep(0.6)
        world_state._cleanup_stale_allocations()

        # Robot2 allocates same region (after cleanup)
        result2 = world_state.allocate_workspace("shared_zone", "Robot2")
        assert result2 is True
        assert world_state.get_workspace_owner("shared_zone") == "Robot2"


@pytest.fixture
def cleanup_world_state():
    """Fixture to cleanup WorldState singleton between tests"""
    # Reset before test
    WorldState._instance = None
    yield
    # Reset after test
    if WorldState._instance is not None:
        WorldState._instance.reset()
    WorldState._instance = None
