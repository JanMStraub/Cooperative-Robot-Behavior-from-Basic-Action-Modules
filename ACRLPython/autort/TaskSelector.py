#!/usr/bin/env python3
"""
AutoRT Task Selector

Task selection with exploration/exploitation balance.
"""

import logging
import random
import threading
import time
from typing import List, Optional, Dict, Any
from collections import defaultdict

from autort.DataModels import ProposedTask

logger = logging.getLogger(__name__)


class TaskSelector:
    """
    Selects best task from approved candidates.

    Tracks task execution history to balance exploration (trying new tasks)
    vs exploitation (repeating successful tasks).
    """

    def __init__(self):
        # History: task description hash → list of outcomes
        self.history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # Lock protecting all reads and writes to self.history across threads
        self._history_lock = threading.Lock()

    def select_task(
        self, candidates: List[ProposedTask], strategy: str = "balanced"
    ) -> Optional[ProposedTask]:
        """
        Select a task from approved candidates.

        Args:
            candidates: List of approved tasks
            strategy: Selection strategy
                - "balanced": Mix novel and practiced tasks
                - "explore": Prioritize untried tasks
                - "exploit": Prioritize high-success tasks
                - "random": Random selection

        Returns:
            Selected task or None if no candidates
        """
        if not candidates:
            return None

        if strategy == "random":
            return random.choice(candidates)

        elif strategy == "explore":
            return self._select_explore(candidates)

        elif strategy == "exploit":
            return self._select_exploit(candidates)

        else:  # balanced
            return self._select_balanced(candidates)

    def _select_explore(self, candidates: List[ProposedTask]) -> ProposedTask:
        """Prioritize tasks with fewer past attempts"""
        scored = []
        with self._history_lock:
            for task in candidates:
                key = self._task_key(task)
                attempt_count = len(self.history[key])
                # Lower count = higher priority (less explored)
                scored.append((task, -attempt_count))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _select_exploit(self, candidates: List[ProposedTask]) -> ProposedTask:
        """Prioritize tasks with highest success rate"""
        scored = []
        with self._history_lock:
            for task in candidates:
                key = self._task_key(task)
                outcomes = self.history[key]
                if not outcomes:
                    # Unknown tasks get neutral score
                    scored.append((task, 0.5))
                else:
                    success_rate = sum(1 for o in outcomes if o.get("success")) / len(
                        outcomes
                    )
                    scored.append((task, success_rate))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _select_balanced(self, candidates: List[ProposedTask]) -> ProposedTask:
        """
        Balance exploration and exploitation.

        Score = success_rate * 0.6 + novelty * 0.4
        Novelty = 1.0 for untried tasks, decays with attempts.
        """
        scored = []
        with self._history_lock:
            for task in candidates:
                key = self._task_key(task)
                outcomes = self.history[key]

                if not outcomes:
                    # Untried tasks get high novelty bonus
                    score = 0.5 * 0.6 + 1.0 * 0.4  # 0.7
                else:
                    success_rate = sum(1 for o in outcomes if o.get("success")) / len(
                        outcomes
                    )
                    novelty = 1.0 / (1.0 + len(outcomes))  # Decays with attempts
                    score = success_rate * 0.6 + novelty * 0.4

                scored.append((task, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def update_history(self, task: ProposedTask, result: Dict[str, Any]):
        """
        Record task outcome for future selection.

        Args:
            task: The executed task
            result: Execution result with at least 'success' key
        """
        key = self._task_key(task)
        with self._history_lock:
            self.history[key].append(
                {
                    "success": result.get("success", False),
                    "timestamp": time.time(),
                    "task_id": task.task_id,
                }
            )

    def _task_key(self, task: ProposedTask) -> str:
        """
        Generate a key for task type grouping.

        Groups by operation sequence pattern (ignoring specific coordinates),
        so "pick red cube" and "pick blue cube" are different but
        "pick red cube at (0.1, 0.2, 0.3)" and "pick red cube at (0.4, 0.5, 0.6)"
        are the same task type.
        """
        op_types = tuple(op.type for op in task.operations)
        return str(op_types)
