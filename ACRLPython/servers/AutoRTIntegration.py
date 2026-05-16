#!/usr/bin/env python3
"""Unity ↔ Python AutoRT handler. Tasks generated but NOT executed until Unity approves."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from datetime import datetime, timedelta

try:
    from core.LoggingSetup import setup_logging

    setup_logging(__name__)
except ImportError:
    from ..core.LoggingSetup import setup_logging

    setup_logging(__name__)

logger = logging.getLogger(__name__)

try:
    from config.AutoRT import (
        MAX_TASK_CANDIDATES,
        LOOP_DELAY_SECONDS,
        DEFAULT_ROBOTS,
        ENABLE_COLLABORATIVE_TASKS,
        ENABLE_SAFETY_VALIDATION,
        TASK_CACHE_SIZE,
        TASK_EXPIRATION_SECONDS,
        UNITY_INTEGRATION_ENABLED,
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
        UNITY_INTEGRATION_ENABLED,
    )


class AutoRTHandler:
    """Singleton handler for AutoRT task generation. Tasks aren't executed until Unity approves."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
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
        """Register callback for pushing generated tasks to Unity."""
        self._task_callback = callback
        logger.info("Task callback registered")

    def set_web_broadcast_callback(self, callback):
        """Register callback for pushing tasks to WebSocket clients."""
        self._web_broadcast_callback = callback
        logger.info("Web broadcast callback registered")

    def _initialize_orchestrator(self):
        if self._orchestrator is not None:
            return

        try:
            from autort.AutoRTLoop import AutoRTOrchestrator

            self._orchestrator = AutoRTOrchestrator()
            logger.info("AutoRTOrchestrator initialized successfully")
            if not UNITY_INTEGRATION_ENABLED:
                logger.warning(
                    "AUTORT_UNITY_INTEGRATION=false — tasks will be generated "
                    "but not pushed to Unity"
                )
        except Exception as e:
            logger.error(f"Failed to initialize AutoRTOrchestrator: {e}")
            raise RuntimeError(f"AutoRT initialization failed: {e}")

    def _cleanup_expired_tasks(self):
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
        try:
            self._initialize_orchestrator()
            self._cleanup_expired_tasks()

            if self._orchestrator is None:
                raise RuntimeError("Orchestrator initialization failed")

            num_tasks = num_tasks or MAX_TASK_CANDIDATES
            robot_ids = robot_ids or DEFAULT_ROBOTS

            logger.info(
                f"Generating {num_tasks} tasks for robots {robot_ids} with strategy '{strategy}'"
            )

            scene_state = self._orchestrator._capture_scene()

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
                            f"[AutoRT constitution] REJECTED '{candidate.task_id}': {verdict.rejection_reason}"
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
        """Run approved task async; returns immediately."""
        try:
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
                del self._pending_tasks[task_id]

            logger.info(f"Starting execution of approved task: {task_id}")

            def execute_async():
                try:
                    self._initialize_orchestrator()

                    if self._orchestrator is None:
                        logger.error("Orchestrator initialization failed")
                        return

                    result = self._orchestrator._execute_task(task)
                    logger.info(
                        f"Task {task_id} execution completed: {result.get('success')}"
                    )

                except Exception as e:
                    logger.error(f"Async task execution failed: {e}", exc_info=True)

            self._exec_pool.submit(execute_async)

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
        with self._task_lock:
            serialized = []
            for _task_id, (task, timestamp) in self._pending_tasks.items():
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
        logger.info("AutoRT loop worker started")

        while not self._loop_stop_event.is_set():
            try:
                # Skip generation if we already have pending tasks waiting for approval.
                with self._task_lock:
                    pending_count = len(self._pending_tasks)
                if pending_count > 0:
                    logger.debug(
                        f"Loop skipping generation — {pending_count} tasks already pending"
                    )
                    self._loop_stop_event.wait(timeout=self._loop_delay)
                    continue

                response = self.generate_tasks(
                    num_tasks=MAX_TASK_CANDIDATES,
                    robot_ids=self._loop_robot_ids,
                    strategy=self._loop_strategy,
                )

                if (
                    UNITY_INTEGRATION_ENABLED
                    and self._task_callback
                    and response.get("tasks")
                ):
                    self._task_callback(response, request_id=0)
                    logger.debug(
                        f"Sent {len(response['tasks'])} tasks to Unity via callback"
                    )

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

                self._loop_stop_event.wait(timeout=self._loop_delay)

            except Exception as e:
                logger.error(f"Loop iteration error: {e}", exc_info=True)
                self._loop_stop_event.wait(timeout=self._loop_delay)

        logger.info("AutoRT loop worker stopped")
