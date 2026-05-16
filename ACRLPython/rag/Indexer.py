#!/usr/bin/env python3
"""Build searchable RAG index from the operations registry."""

from typing import Optional, List, Dict, Any

# Lazy imports to avoid circular dependency
# from operations.Registry import OperationRegistry, get_global_registry
# from operations.WorkflowPatterns import WorkflowPatternRegistry, get_global_workflow_registry

from .Embeddings import EmbeddingGenerator
from .vector_store import VectorStore

try:
    from config.Rag import RAG_AUTO_SAVE_INDEX
except ImportError:
    from ..config.Rag import RAG_AUTO_SAVE_INDEX

from core.LoggingSetup import get_logger

logger = get_logger(__name__)


class OperationIndexer:
    """Build searchable embeddings index from the operations and workflow registries."""

    def __init__(
        self,
        registry: Optional[Any] = None,  # Changed to Any to avoid circular import
        workflow_registry: Optional[
            Any
        ] = None,  # Changed to Any to avoid circular import
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        # Lazy imports
        from core.Imports import get_global_registry
        from operations.WorkflowPatterns import get_global_workflow_registry

        self.registry = registry or get_global_registry()
        self.workflow_registry = workflow_registry or get_global_workflow_registry()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def _get_multi_robot_context_documents(self) -> List[Dict[str, Any]]:
        """Generate context documents for workspace layout, coordination patterns, and safety constraints."""
        context_docs = []

        workspace_doc = """
        MULTI-ROBOT WORKSPACE LAYOUT

        Available Robots:
        - Robot1: Left workspace (approximate position x≈-0.4, y≈0.0, z varies)
        - Robot2: Right workspace (approximate position x≈0.4, y≈0.0, z varies)

        Workspace Regions:
        - Left region (x < -0.15): Primary workspace for Robot1
        - Right region (x > 0.15): Primary workspace for Robot2
        - Center region (-0.15 ≤ x ≤ 0.15): Shared handoff zone, requires coordination

        Workspace Dimensions:
        - X range: -0.5 to 0.5 meters (left to right)
        - Y range: 0.0 to 0.6 meters (front to back, from robot base forward)
        - Z range: -0.45 to 0.45 meters (vertical, centered at robot base height)

        Coordinate System:
        - Origin (0, 0, 0) is at the center of the shared workspace at robot base height
        - Negative X is towards Robot1's side (left)
        - Positive X is towards Robot2's side (right)
        - Positive Y is forward from robot base
        - Positive Z is upward from robot base level

        Safety Guidelines:
        - Maintain minimum 0.2m separation between robots
        - Unity's CollaborativeStrategy automatically handles collision checking
        - Use center region (handoff zone) for object transfers between robots
        """

        context_docs.append(
            {
                "text": workspace_doc,
                "metadata": {
                    "operation_id": "context_workspace_layout",
                    "name": "workspace_layout",
                    "category": "multi_robot_context",
                    "complexity": "informational",
                    "description": "Multi-robot workspace layout and coordinate system",
                    "type": "context",
                },
            }
        )

        coordination_doc = """
        MULTI-ROBOT COORDINATION PATTERNS

        The LLM should plan coordination using atomic operations + sync primitives.
        Here are typical coordination patterns (NOT operations, but planning examples):

        Pattern 1: Parallel Movement
        When robots move to different areas without collision:
        - Use parallel_group to execute movements simultaneously
        - Example: Robot1 moves to (0.3, 0.1, 0.2), Robot2 moves to (-0.3, -0.1, 0.2)
        - Unity automatically checks for collisions

        Pattern 2: Sequential Handoff
        When transferring object from Robot1 to Robot2:
        1. Robot1: detect_object → move_to_coordinate → control_gripper (close)
        2. Robot1: signal("object_gripped")
        3. Robot2: wait_for_signal("object_gripped")
        4. Both: move_to_coordinate (handoff zone, e.g., x=0.0, y=0.0, z=0.3)
        5. Robot1: signal("robot1_ready"), Robot2: signal("robot2_ready")
        6. Robot1: wait_for_signal("robot2_ready"), Robot2: wait_for_signal("robot1_ready")
        7. Robot2: control_gripper (close)
        8. wait(500) - Allow gripper to stabilize
        9. Robot1: control_gripper (open)

        Pattern 3: One Robot Waits While Other Works
        When Robot2 depends on Robot1 completing a task:
        1. Robot1: Execute task operations
        2. Robot1: signal("task_complete")
        3. Robot2: wait_for_signal("task_complete") at the start
        4. Robot2: Execute dependent operations

        Pattern 4: Synchronized Parallel Operations
        When both robots need to act at the same time:
        - Use parallel_group to group simultaneous operations
        - Use signals to synchronize before/after parallel actions
        - Example: Both robots grip at handoff - use parallel_group

        Key Principles:
        - Break complex tasks into atomic operations per robot
        - Use signal/wait_for_signal for synchronization points
        - Use parallel_group for true simultaneous execution
        - Consider workspace regions when planning movements
        - Use the center handoff zone (x≈0.0) for object transfers
        """

        context_docs.append(
            {
                "text": coordination_doc,
                "metadata": {
                    "operation_id": "context_coordination_patterns",
                    "name": "coordination_patterns",
                    "category": "multi_robot_context",
                    "complexity": "informational",
                    "description": "Typical multi-robot coordination patterns using atomic operations",
                    "type": "context",
                },
            }
        )

        safety_doc = """
        MULTI-ROBOT SAFETY CONSTRAINTS

        Automatic Safety (Handled by Unity):
        - Collision detection between robots (minimum 0.2m separation)
        - Workspace conflict detection
        - Path collision prediction during movements
        - Automatic serialization if movements would collide

        LLM Planning Considerations:
        - When both robots move to nearby positions, consider using sequential execution
        - Use parallel_group only when movements are clearly separated
        - In the center handoff zone (-0.15 ≤ x ≤ 0.15), coordinate carefully
        - Signal when entering/exiting shared workspace regions
        - Use wait_for_signal to ensure sequential access to shared regions

        Gripper Safety:
        - Close gripper only when object is within reach
        - Use wait(500) after closing gripper to allow stabilization
        - During handoff, ensure receiving robot grips before releasing robot opens
        - Verify object position before attempting grip

        Timeout Safety:
        - Use appropriate timeout_ms for wait_for_signal (default 30000ms = 30 seconds)
        - For quick operations, use shorter timeouts (e.g., 5000ms)
        - If timeout occurs, check if signaling robot completed its operation
        - Provide recovery suggestions in error messages

        Error Recovery:
        - If operation fails, check_robot_status to diagnose
        - If handoff fails, source robot should return object to original position
        - Use signal/wait error codes to detect coordination failures
        - Retry individual steps rather than entire sequence
        """

        context_docs.append(
            {
                "text": safety_doc,
                "metadata": {
                    "operation_id": "context_safety_constraints",
                    "name": "safety_constraints",
                    "category": "multi_robot_context",
                    "complexity": "informational",
                    "description": "Safety constraints and guidelines for multi-robot operations",
                    "type": "context",
                },
            }
        )

        parallel_doc = """
        PARALLEL EXECUTION WITH PARALLEL_GROUP

        The parallel_group field enables true simultaneous execution of operations.
        Operations with the same parallel_group number execute concurrently.

        How It Works:
        - Operations are grouped by their parallel_group number
        - Group 0 executes first, then group 1, then group 2, etc.
        - All operations in a group run simultaneously (in threads)
        - Next group waits for all operations in current group to complete

        Example: Two-Robot Handoff
        parallel_group=0: Robot1 detect_object, Robot2 idle
        parallel_group=1: Robot1 move_to_coordinate, Robot2 wait_for_signal
        parallel_group=2: Robot1 control_gripper (close), Robot1 signal
        parallel_group=3: Robot1 move (handoff), Robot2 move (handoff)
        parallel_group=4: Robot1 signal, Robot2 signal
        parallel_group=5: Robot1 wait_for_signal, Robot2 wait_for_signal
        parallel_group=6: Robot2 control_gripper (close), wait(500)
        parallel_group=7: Robot1 control_gripper (open)

        When to Use Parallel Groups:
        - Moving both robots to different regions simultaneously
        - One robot working while another waits for signal
        - Both robots signaling at the same time
        - Synchronized actions (e.g., both grip during handoff)

        When NOT to Use (Sequential Instead):
        - Operations that depend on previous operation results
        - When unsure about collision safety
        - Single robot performing multiple steps
        - Operations requiring strict ordering

        Mixing Sequential and Parallel:
        - Same parallel_group = concurrent
        - Different parallel_group = sequential between groups
        - Omit parallel_group field = fully sequential execution
        """

        context_docs.append(
            {
                "text": parallel_doc,
                "metadata": {
                    "operation_id": "context_parallel_execution",
                    "name": "parallel_execution_guide",
                    "category": "multi_robot_context",
                    "complexity": "informational",
                    "description": "Guide for using parallel_group to execute operations concurrently",
                    "type": "context",
                },
            }
        )

        return context_docs

    def build_index(self, save: bool = True) -> VectorStore:
        """Build and return a populated VectorStore from all operations and workflows."""
        operations = self.registry.get_all_operations()
        workflows = self.workflow_registry.get_all_patterns()
        context_docs = self._get_multi_robot_context_documents()

        if not operations and not workflows:
            logger.warning("No operations or workflows found in registries")
            return VectorStore()

        logger.debug(
            f"Building index for {len(operations)} operations, {len(workflows)} workflows, and {len(context_docs)} context documents..."
        )

        store = VectorStore()
        texts_to_embed = []
        operation_data = []

        for op in operations:
            rag_text = op.to_rag_document()
            texts_to_embed.append(rag_text)
            operation_data.append(
                {
                    "operation_id": op.operation_id,
                    "metadata": {
                        "name": op.name,
                        "category": op.category.value,
                        "complexity": op.complexity.value,
                        "description": op.description,
                        "average_duration_ms": op.average_duration_ms,
                        "success_rate": op.success_rate,
                        "parameters": [p.name for p in op.parameters],
                        "type": "operation",
                    },
                }
            )

        for workflow in workflows:
            rag_text = workflow.to_rag_document()
            texts_to_embed.append(rag_text)
            operation_data.append(
                {
                    "operation_id": workflow.pattern_id,
                    "metadata": {
                        "name": workflow.name,
                        "category": workflow.category.value,
                        "complexity": "workflow",
                        "description": workflow.description,
                        "average_duration_ms": 0,  # Workflows don't have duration
                        "success_rate": 1.0,
                        "parameters": [],
                        "type": "workflow",
                        "step_count": len(workflow.steps),
                    },
                }
            )

        for context_doc in context_docs:
            texts_to_embed.append(context_doc["text"])
            operation_data.append(
                {
                    "operation_id": context_doc["metadata"]["operation_id"],
                    "metadata": context_doc["metadata"],
                }
            )

        embeddings = self.embedding_generator.generate_embeddings(texts_to_embed)

        for data, embedding in zip(operation_data, embeddings):
            store.add_operation(
                operation_id=data["operation_id"],
                embedding=embedding,
                metadata=data["metadata"],
            )

        logger.info(
            f"Index built with {len(operations)} operations, {len(workflows)} workflows, {len(context_docs)} context docs"
        )

        if save and RAG_AUTO_SAVE_INDEX:
            store.save()

        return store

    def refresh_index(
        self, existing_store: VectorStore, save: bool = True
    ) -> VectorStore:
        """Incrementally add only new operations/workflows to an existing index, preserving accumulated metadata."""
        operations = self.registry.get_all_operations()
        workflows = self.workflow_registry.get_all_patterns()
        context_docs = self._get_multi_robot_context_documents()

        existing_ids = set(existing_store.operation_ids)

        # Collect texts and metadata for documents not yet indexed
        texts_to_embed = []
        operation_data = []

        for op in operations:
            if op.operation_id in existing_ids:
                continue
            texts_to_embed.append(op.to_rag_document())
            operation_data.append(
                {
                    "operation_id": op.operation_id,
                    "metadata": {
                        "name": op.name,
                        "category": op.category.value,
                        "complexity": op.complexity.value,
                        "description": op.description,
                        "average_duration_ms": op.average_duration_ms,
                        "success_rate": op.success_rate,
                        "parameters": [p.name for p in op.parameters],
                        "type": "operation",
                    },
                }
            )

        for workflow in workflows:
            if workflow.pattern_id in existing_ids:
                continue
            texts_to_embed.append(workflow.to_rag_document())
            operation_data.append(
                {
                    "operation_id": workflow.pattern_id,
                    "metadata": {
                        "name": workflow.name,
                        "category": workflow.category.value,
                        "complexity": "workflow",
                        "description": workflow.description,
                        "average_duration_ms": 0,
                        "success_rate": 1.0,
                        "parameters": [],
                        "type": "workflow",
                        "step_count": len(workflow.steps),
                    },
                }
            )

        for ctx in context_docs:
            if ctx["metadata"]["operation_id"] in existing_ids:
                continue
            texts_to_embed.append(ctx["text"])
            operation_data.append(
                {
                    "operation_id": ctx["metadata"]["operation_id"],
                    "metadata": ctx["metadata"],
                }
            )

        total_checked = len(operations) + len(workflows) + len(context_docs)
        logger.info(
            f"Refreshing index: checking {total_checked} documents "
            f"({len(existing_ids)} already indexed, {len(texts_to_embed)} new)..."
        )

        if not texts_to_embed:
            logger.info("Index already up-to-date; nothing new to add")
            return existing_store

        embeddings = self.embedding_generator.generate_embeddings(texts_to_embed)

        for data, embedding in zip(operation_data, embeddings):
            existing_store.add_operation(
                operation_id=data["operation_id"],
                embedding=embedding,
                metadata=data["metadata"],
            )

        logger.info(
            f"Added {len(texts_to_embed)} new document(s); "
            f"{len(existing_ids)} documents already indexed"
        )

        if save and RAG_AUTO_SAVE_INDEX:
            existing_store.save()

        return existing_store

    def rebuild_index(self) -> VectorStore:
        """Rebuild index from scratch."""
        return self.build_index(save=True)

    def update_index(self, existing_store: VectorStore) -> VectorStore:
        """Update existing index (currently rebuilds entire index)."""
        return self.build_index(save=True)

    def get_indexer_stats(self) -> dict:
        """Return statistics about the indexer and embedding generator."""
        operations = self.registry.get_all_operations()

        return {
            "num_operations": len(operations),
            "embedding_generator": repr(self.embedding_generator),
            "embedding_dimension": self.embedding_generator.get_embedding_dimension(),
            "using_lm_studio": self.embedding_generator.is_using_lm_studio(),
        }


def build_index_from_registry(
    registry: Optional[Any] = None,  # Changed to Any to avoid circular import
    save_path: Optional[str] = None,
) -> VectorStore:
    """Convenience function to build index from the global operation registry."""
    indexer = OperationIndexer(registry=registry)
    store = indexer.build_index(save=True)

    if save_path:
        store.save(save_path)

    return store
