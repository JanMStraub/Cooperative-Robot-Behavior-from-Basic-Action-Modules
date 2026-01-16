"""
Shared Base Classes for Singleton Pattern Tests
================================================

Provides reusable test mixins for testing singleton patterns across the codebase.
Eliminates duplicate singleton testing code and ensures consistent testing approach.

Usage:
    class TestMySingleton(SingletonTestMixin):
        def get_singleton_instance(self):
            return get_my_singleton()

        def get_cleanup_fixture_name(self):
            return "cleanup_my_singleton"

        # Automatically inherits:
        # - test_singleton_pattern()
        # - test_singleton_thread_safe_initialization()
        # - test_singleton_returns_same_instance()
"""

import threading
import pytest


class SingletonTestMixin:
    """
    Mixin class providing standard singleton pattern tests.

    Subclasses must implement:
    - get_singleton_instance(): Returns singleton instance
    - get_cleanup_fixture_name(): Returns name of cleanup fixture (optional)
    """

    def get_singleton_instance(self):
        """
        Get the singleton instance to test.

        Must be implemented by subclass.

        Returns:
            Singleton instance

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclass must implement get_singleton_instance()")

    def get_cleanup_fixture_name(self):
        """
        Get the name of the cleanup fixture.

        Optional - only needed if cleanup fixture exists.

        Returns:
            str: Name of cleanup fixture, or None if no cleanup needed
        """
        return None

    def test_singleton_pattern(self, request):
        """Test that only one instance of singleton exists."""
        # Get cleanup fixture if specified
        cleanup_fixture = self.get_cleanup_fixture_name()
        if cleanup_fixture and hasattr(request, 'getfixturevalue'):
            try:
                request.getfixturevalue(cleanup_fixture)
            except Exception:
                pass  # Cleanup fixture not available

        instance1 = self.get_singleton_instance()
        instance2 = self.get_singleton_instance()

        assert instance1 is instance2, "Singleton should return same instance"
        assert id(instance1) == id(instance2), "Singleton instances should have same id"

    def test_singleton_returns_same_instance(self, request):
        """Test multiple calls return same instance (alias for clarity)."""
        cleanup_fixture = self.get_cleanup_fixture_name()
        if cleanup_fixture and hasattr(request, 'getfixturevalue'):
            try:
                request.getfixturevalue(cleanup_fixture)
            except Exception:
                pass

        instances = [self.get_singleton_instance() for _ in range(5)]

        # All instances should be the same
        for i in range(1, len(instances)):
            assert instances[i] is instances[0]

    def test_singleton_thread_safe_initialization(self, request):
        """Test singleton is thread-safe during concurrent initialization."""
        cleanup_fixture = self.get_cleanup_fixture_name()
        if cleanup_fixture and hasattr(request, 'getfixturevalue'):
            try:
                request.getfixturevalue(cleanup_fixture)
            except Exception:
                pass

        instances = []
        errors = []
        barrier = threading.Barrier(10)  # Synchronize 10 threads

        def get_instance():
            try:
                barrier.wait()  # Wait for all threads to be ready
                instance = self.get_singleton_instance()
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=2.0)

        # No errors should occur
        assert len(errors) == 0, f"Errors during initialization: {errors}"

        # All threads should get the same instance
        assert len(instances) == 10
        for instance in instances:
            assert instance is instances[0], "All threads should get same singleton instance"


class ResetableSingletonTestMixin(SingletonTestMixin):
    """
    Mixin for testing singletons that support reset/cleanup.

    Subclasses must additionally implement:
    - reset_singleton(): Method to reset singleton state
    """

    def reset_singleton(self):
        """
        Reset the singleton to initial state.

        Must be implemented by subclass if singleton supports reset.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclass must implement reset_singleton() if singleton is resetable")

    def test_singleton_reset(self, request):
        """Test singleton can be reset/reinitialized."""
        cleanup_fixture = self.get_cleanup_fixture_name()
        if cleanup_fixture and hasattr(request, 'getfixturevalue'):
            try:
                request.getfixturevalue(cleanup_fixture)
            except Exception:
                pass

        instance1 = self.get_singleton_instance()

        # Reset singleton
        self.reset_singleton()

        # Get instance again (behavior depends on implementation)
        # Some singletons return same instance, others create new
        instance2 = self.get_singleton_instance()

        # Test passes if no exception raised
        assert instance2 is not None


# ============================================================================
# Example Usage (for documentation)
# ============================================================================

class ExampleSingletonTest(SingletonTestMixin):
    """
    Example showing how to use SingletonTestMixin.

    This class is for documentation only and not run as actual test.
    """

    def get_singleton_instance(self):
        """Return your singleton instance."""
        # Example:
        # from mymodule import get_my_singleton
        # return get_my_singleton()
        pass

    def get_cleanup_fixture_name(self):
        """Return cleanup fixture name if needed."""
        return "cleanup_my_singleton"

    # Now you automatically get:
    # - test_singleton_pattern()
    # - test_singleton_returns_same_instance()
    # - test_singleton_thread_safe_initialization()


# ============================================================================
# Consolidated Singleton Tests
# ============================================================================

class TestConsolidatedSingletons:
    """
    Consolidated singleton pattern tests for all system singletons.

    Tests multiple singletons in one place for easier maintenance.
    """

    def test_all_singletons_thread_safe(self):
        """Test all major singletons are thread-safe."""
        from operations.WorldState import get_world_state
        from servers.CommandServer import get_command_broadcaster

        singletons_to_test = [
            ("WorldState", get_world_state),
            ("CommandBroadcaster", get_command_broadcaster),
        ]

        for name, getter in singletons_to_test:
            instances = []
            errors = []
            barrier = threading.Barrier(5)

            def get_instance():
                try:
                    barrier.wait()
                    instance = getter()
                    instances.append(instance)
                except Exception as e:
                    errors.append((name, e))

            threads = [threading.Thread(target=get_instance) for _ in range(5)]

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=2.0)

            assert len(errors) == 0, f"{name} had errors: {errors}"
            assert len(instances) == 5
            for instance in instances:
                assert instance is instances[0], f"{name} returned different instances"

    def test_all_singletons_return_same_instance(self):
        """Test all singletons consistently return same instance."""
        from operations.WorldState import get_world_state
        from servers.CommandServer import get_command_broadcaster

        singletons_to_test = [
            ("WorldState", get_world_state),
            ("CommandBroadcaster", get_command_broadcaster),
        ]

        for name, getter in singletons_to_test:
            instances = [getter() for _ in range(3)]

            for i in range(1, len(instances)):
                assert instances[i] is instances[0], f"{name} returned different instances"
                assert id(instances[i]) == id(instances[0]), f"{name} instances have different IDs"
