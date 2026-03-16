#!/usr/bin/env python3
"""
AutoRTIntegration.py - Unity ↔ Python AutoRT handler

Provides integration between Unity's AutoRTManager and Python's AutoRTOrchestrator.
Handles task generation WITHOUT automatic execution - Unity approves tasks first.

Architecture:
- Singleton handler integrates with SequenceServer
- Manages background loop thread for continuous task generation
- Caches generated tasks by ID for later execution
- Sends tasks to Unity via AUTORT_RESPONSE messages

Usage:
    handler = AutoRTHandler.get_instance()
    result = handler.generate_tasks(num_tasks=5, robot_ids=["Robot1", "Robot2"])
    result = handler.start_loop(loop_delay=5.0)
    result = handler.execute_task(task_id="task_12345")
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Configure logging
try:
    from core.LoggingSetup import setup_logging

    setup_logging(__name__)
except ImportError:
    from ..core.LoggingSetup import setup_logging

    setup_logging(__name__)

logger = logging.getLogger(__name__)

# Import config
try:
    from config.AutoRT import (
        MAX_TASK_CANDIDATES,
        LOOP_DELAY_SECONDS,
        DEFAULT_ROBOTS,
        ENABLE_COLLABORATIVE_TASKS,
        ENABLE_SAFETY_VALIDATION,
        TASK_CACHE_SIZE,
        TASK_EXPIRATION_SECONDS,
    )
except ImportError:
    from ..config.AutoRT import (
        MAX_TASK_CANDIDATES,
        LOOP_DELAY_SECONDS,
        DEFAULT_ROBOTS,
        ENABLE_COLLABORATIVE_TASKS,
        ENABLE_SAFETY_VALIDATION,
        TASK_CACHE_SIZE,
        TASK_EXPIRATION_SECONDS,
    )


class AutoRTHandler:
    """
    Singleton handler for Unity-integrated AutoRT task generation.

    Manages AutoRTOrchestrator lifecycle and provides Unity-compatible API.
    Tasks are generated but NOT executed - Unity must approve first.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize AutoRT handler (private - use get_instance)."""
        self._orchestrator = None
        self._loop_thread = None
        self._loop_running = False
        self._loop_stop_event = threading.Event()

        # Task caching: task_id -> (task_dict, timestamp)
        self._pending_tasks: Dict[str, tuple] = {}
        self._task_lock = threading.Lock()

        # Loop configuration
        self._loop_delay = LOOP_DELAY_SECONDS
        self._loop_robot_ids = DEFAULT_ROBOTS.copy()
        self._loop_strategy = "balanced"

        # Callback for sending tasks to Unity (set by SequenceServer)
        self._task_callback = None

        # Callback for pushing tasks to WebSocket clients (set by WebUIServer)
        self._web_broadcast_callback = None

        # Bounded thread pool for async task execution (one slot per robot arm)
        self._exec_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="AutoRT-Execute"
        )

        logger.info("AutoRTHandler initialized")

    @classmethod
    def get_instance(cls):
        """Get singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_task_callback(self, callback):
        """
        Set callback function for sending tasks to Unity.

        Args:
            callback: Function(response_dict, request_id) -> None
        """
        self._task_callback = callback
        logger.info("Task callback registered")

    def set_web_broadcast_callback(self, callback):
        """Register callback for pushing tasks to WebSocket clients (web dashboard)."""
        self._web_broadcast_callback = callback
        logger.info("Web broadcast callback registered")

    def _initialize_orchestrator(self):
        """Lazy-initialize AutoRT orchestrator when needed."""
        if self._orchestrator is not None:
            return

        try:
            # Import here to avoid circular dependencies
            from autort.AutoRTLoop import AutoRTOrchestrator

            self._orchestrator = AutoRTOrchestrator()
            logger.info("AutoRTOrchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AutoRTOrchestrator: {e}")
            raise RuntimeError(f"AutoRT initialization failed: {e}")

    def _cleanup_expired_tasks(self):
        """Remove expired tasks from cache."""
        now = datetime.now()
        expiration_threshold = timedelta(seconds=TASK_EXPIRATION_SECONDS)

        with self._task_lock:
            expired_ids = [
                task_id
                for task_id, (_, timestamp) in self._pending_tasks.items()
                if now - timestamp > expiration_threshold
            ]

            for task_id in expired_ids:
                del self._pending_tasks[task_id]
                logger.debug(f"Removed expired task: {task_id}")

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired tasks")

    def _cache_task(self, task) -> str:
        """
        Cache a generated task.

        Args:
            task: ProposedTask object from AutoRTOrchestrator

        Returns:
            Task ID (uses existing task.task_id)
        """
        task_id = task.task_id
        timestamp = datetime.now()

        with self._task_lock:
            # Enforce cache size limit
            if len(self._pending_tasks) >= TASK_CACHE_SIZE:
                # Remove oldest task
                oldest_id = min(
                    self._pending_tasks.keys(), key=lambda k: self._pending_tasks[k][1]
                )
                del self._pending_tasks[oldest_id]
                logger.debug(f"Cache full, removed oldest task: {oldest_id}")

            self._pending_tasks[task_id] = (task, timestamp)

        logger.debug(f"Cached task: {task_id}")
        return task_id

    def _serialize_task(self, task) -> dict:
        """
        Convert ProposedTask to Unity-compatible format.

        Args:
            task: ProposedTask object

        Returns:
            Serialized task dict
        """
        # Convert Operation objects to dicts
        operations_list = []
        for op in task.operations:
            operations_list.append(
                {
                    "type": op.type,
                    "robot_id": op.robot_id,
                    "parameters": op.parameters,
                }
            )

        return {
            "task_id": task.task_id,
            "description": task.description,
            "operations": operations_list,
            "required_robots": task.required_robots,
            "estimated_complexity": task.estimated_complexity,
            "reasoning": task.reasoning,
        }

    def generate_tasks(
        self,
        num_tasks: Optional[int] = None,
        robot_ids: Optional[List[str]] = None,
        strategy: str = "balanced",
    ) -> dict:
        """
        Generate new tasks without executing them.

        Args:
            num_tasks: Number of tasks to generate (default: config value)
            robot_ids: Robot IDs to use (default: config value)
            strategy: Selection strategy ("balanced", "explore", "exploit", "random")

        Returns:
            Response dict: {success, tasks[], error}
        """
        try:
            self._initialize_orchestrator()
            self._cleanup_expired_tasks()

            # Type guard: ensure orchestrator is initialized
            if self._orchestrator is None:
                raise RuntimeError("Orchestrator initialization failed")

            num_tasks = num_tasks or MAX_TASK_CANDIDATES
            robot_ids = robot_ids or DEFAULT_ROBOTS

            logger.info(
                f"Generating {num_tasks} tasks for robots {robot_ids} with strategy '{strategy}'"
            )

            # Capture scene state
            scene_state = self._orchestrator._capture_scene()

            # Generate task candidates using TaskGenerator
            candidates = self._orchestrator.task_generator.generate_tasks(
                scene_state,
                robot_ids=robot_ids,
                num_tasks=num_tasks,
                include_collaborative=(
                    len(robot_ids) > 1 and ENABLE_COLLABORATIVE_TASKS
                ),
            )

            if not candidates:
                logger.warning("No task candidates generated")
                return {
                    "success": True,
                    "tasks": [],
                    "loop_running": self._loop_running,
                    "error": None,
                }

            # Filter through constitution (safety validation) - skip if disabled
            if ENABLE_SAFETY_VALIDATION:
                validated_tasks = []
                for candidate in candidates:
                    verdict = self._orchestrator.constitution.evaluate_task(
                        candidate, scene_state
                    )
                    if verdict.approved:
                        validated_tasks.append(candidate)
                        if verdict.warnings:
                            logger.debug(
                                f"Task '{candidate.task_id}' approved with warnings: {verdict.warnings}"
                            )
                    else:
                        logger.debug(
                            f"Task '{candidate.task_id}' rejected: {verdict.rejection_reason}"
                        )

                if not validated_tasks:
                    logger.warning("All tasks rejected by constitution")
                    return {
                        "success": True,
                        "tasks": [],
                        "loop_running": self._loop_running,
                        "error": "All tasks rejected by safety filters",
                    }
            else:
                # Skip safety validation
                logger.warning("Safety validation DISABLED - accepting all tasks")
                validated_tasks = candidates

            # Select tasks using TaskSelector — while loop ensures we fill the
            # requested count even when the selector filters some candidates.
            selected_tasks = []
            while len(selected_tasks) < num_tasks and validated_tasks:
                selected = self._orchestrator.task_selector.select_task(
                    validated_tasks, strategy=strategy
                )
                if selected:
                    selected_tasks.append(selected)
                    validated_tasks = [t for t in validated_tasks if t != selected]
                else:
                    break

            # Cache tasks and serialize for Unity
            serialized_tasks = []
            for task in selected_tasks:
                self._cache_task(task)  # Cache using task.task_id
                serialized = self._serialize_task(task)
                serialized_tasks.append(serialized)

            logger.info(f"Generated {len(serialized_tasks)} valid tasks")

            return {
                "success": True,
                "tasks": serialized_tasks,
                "loop_running": self._loop_running,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Task generation failed: {e}", exc_info=True)
            return {
                "success": False,
                "tasks": [],
                "loop_running": self._loop_running,
                "error": str(e),
            }

    def start_loop(
        self,
        loop_delay: Optional[float] = None,
        robot_ids: Optional[List[str]] = None,
        strategy: str = "balanced",
    ) -> dict:
        """
        Start continuous task generation loop in background thread.

        Args:
            loop_delay: Seconds between generations (default: config value)
            robot_ids: Robot IDs to use (default: config value)
            strategy: Selection strategy

        Returns:
            Response dict: {success, loop_running}
        """
        if self._loop_running:
            logger.warning("Loop already running")
            return {
                "success": True,
                "loop_running": True,
                "error": "Loop already running",
            }

        try:
            self._loop_delay = loop_delay or LOOP_DELAY_SECONDS
            self._loop_robot_ids = robot_ids or DEFAULT_ROBOTS
            self._loop_strategy = strategy

            self._loop_stop_event.clear()
            self._loop_thread = threading.Thread(
                target=self._loop_worker,
                name="AutoRT-Loop",
                daemon=True,
            )
            self._loop_running = True
            self._loop_thread.start()

            logger.info(
                f"Started AutoRT loop: delay={self._loop_delay}s, "
                f"robots={self._loop_robot_ids}, strategy={self._loop_strategy}"
            )

            return {
                "success": True,
                "loop_running": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Failed to start loop: {e}", exc_info=True)
            self._loop_running = False
            return {
                "success": False,
                "loop_running": False,
                "error": str(e),
            }

    def stop_loop(self) -> dict:
        """
        Stop continuous task generation loop.

        Returns:
            Response dict: {success, loop_running}
        """
        if not self._loop_running:
            logger.info("Loop not running")
            return {
                "success": True,
                "loop_running": False,
                "error": None,
            }

        try:
            logger.info("Stopping AutoRT loop...")
            self._loop_stop_event.set()
            self._loop_running = False

            # Wait for thread to finish (max 5 seconds)
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=5.0)

            logger.info("AutoRT loop stopped")

            return {
                "success": True,
                "loop_running": False,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Failed to stop loop: {e}", exc_info=True)
            return {
                "success": False,
                "loop_running": self._loop_running,
                "error": str(e),
            }

    def execute_task(self, task_id: str) -> dict:
        """
        Execute a previously generated task (approved by Unity).

        Returns immediately with acknowledgment. Task executes asynchronously.

        Args:
            task_id: Task ID from cache

        Returns:
            Response dict: {success, result, error, status}
        """
        try:
            # Retrieve task from cache
            with self._task_lock:
                if task_id not in self._pending_tasks:
                    logger.warning(f"Task not found in cache: {task_id}")
                    return {
                        "success": False,
                        "result": None,
                        "error": f"Task {task_id} not found (may have expired)",
                        "status": "not_found",
                    }

                task, _ = self._pending_tasks[task_id]
                # Remove from cache after retrieval
                del self._pending_tasks[task_id]

            logger.info(f"Starting execution of approved task: {task_id}")

            # Execute task asynchronously in background thread
            # This allows immediate response to Unity
            def execute_async():
                try:
                    self._initialize_orchestrator()

                    # Type guard: ensure orchestrator is initialized
                    if self._orchestrator is None:
                        logger.error("Orchestrator initialization failed")
                        return

                    result = self._orchestrator._execute_task(task)
                    logger.info(
                        f"Task {task_id} execution completed: {result.get('success')}"
                    )

                except Exception as e:
                    logger.error(f"Async task execution failed: {e}", exc_info=True)

            # Submit to bounded pool (max_workers=2) to cap concurrency
            self._exec_pool.submit(execute_async)

            # Return immediately with acknowledgment
            logger.info(
                f"Task {task_id} submitted to executor pool, returning immediate response"
            )
            return {
                "success": True,
                "result": {"task_id": task_id, "status": "executing"},
                "error": None,
                "status": "started",
            }

        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "status": "error",
            }

    def get_status(self) -> dict:
        """
        Get current AutoRT status.

        Returns:
            Status dict: {loop_running, pending_tasks_count, loop_config}
        """
        with self._task_lock:
            pending_count = len(self._pending_tasks)

        return {
            "success": True,
            "loop_running": self._loop_running,
            "pending_tasks_count": pending_count,
            "loop_config": {
                "delay": self._loop_delay,
                "robot_ids": self._loop_robot_ids,
                "strategy": self._loop_strategy,
            },
            "error": None,
        }

    def get_pending_tasks(self) -> dict:
        """Return all cached pending tasks serialized for HTTP responses."""
        with self._task_lock:
            serialized = []
            for task_id, (task, timestamp) in self._pending_tasks.items():
                task_dict = self._serialize_task(task)
                task_dict["cached_at"] = timestamp.isoformat()
                serialized.append(task_dict)
        return {
            "success": True,
            "tasks": serialized,
            "loop_running": self._loop_running,
            "pending_tasks_count": len(serialized),
        }

    def _loop_worker(self):
        """Background thread worker for continuous task generation."""
        logger.info("AutoRT loop worker started")

        while not self._loop_stop_event.is_set():
            try:
                # Generate tasks
                response = self.generate_tasks(
                    num_tasks=MAX_TASK_CANDIDATES,
                    robot_ids=self._loop_robot_ids,
                    strategy=self._loop_strategy,
                )

                # Send to Unity via callback (if registered)
                if self._task_callback and response.get("tasks"):
                    self._task_callback(response, request_id=0)
                    logger.debug(
                        f"Sent {len(response['tasks'])} tasks to Unity via callback"
                    )

                # Push to web dashboard via WebSocket broadcast (if registered)
                if self._web_broadcast_callback and response.get("tasks"):
                    try:
                        self._web_broadcast_callback(
                            {
                                "type": "autort_tasks",
                                "tasks": response["tasks"],
                                "loop_running": self._loop_running,
                            }
                        )
                    except Exception as cb_err:
                        logger.error(f"Web broadcast callback failed: {cb_err}")

                # Wait with interruptible sleep
                self._loop_stop_event.wait(timeout=self._loop_delay)

            except Exception as e:
                logger.error(f"Loop iteration error: {e}", exc_info=True)
                # Continue loop despite errors
                self._loop_stop_event.wait(timeout=self._loop_delay)

        logger.info("AutoRT loop worker stopped")
