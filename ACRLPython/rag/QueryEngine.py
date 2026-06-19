#!/usr/bin/env python3
"""Semantic search and retrieval over robot operations."""

from typing import List, Dict, Any, Optional

from .Embeddings import EmbeddingGenerator
from .VectorStore import VectorStore

try:
    from config.Rag import RAG_DEFAULT_TOP_K, RAG_CONFIDENCE_TIERS
except ImportError:
    from ..config.Rag import RAG_DEFAULT_TOP_K, RAG_CONFIDENCE_TIERS

from core.LoggingSetup import get_logger

logger = get_logger(__name__)


class QueryEngine:
    """Semantic search over indexed robot operations."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        registry: Optional[Any] = None,  # Changed to Any to avoid circular import
    ):
        # Lazy import to avoid circular dependency
        from core.Imports import get_global_registry

        self.vector_store = vector_store
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.registry = registry or get_global_registry()
        self._world_state = None  # Optional WorldState for context-aware search

    def set_world_state(self, world_state):
        """Inject WorldState so search results can be re-ranked by reachability and grasp state."""
        self._world_state = world_state
        logger.debug("WorldState injected into QueryEngine for context-aware search")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        category_filter: Optional[str] = None,
        complexity_filter: Optional[str] = None,
        include_full_operation: bool = False,
        robot_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        logger.debug(f"Searching for: '{query}'")
        query_embedding = self.embedding_generator.generate_embedding(query)

        k = top_k if top_k is not None else RAG_DEFAULT_TOP_K
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            min_score=min_score,
            category_filter=category_filter,
            complexity_filter=complexity_filter,
            query_text=query,
        )

        if self._world_state and robot_id:
            results = self._apply_world_constraints(results, robot_id, query)

        # Deduplicate by name - operations indexed as both operation and workflow
        # can appear twice; keep the higher-ranked (first) occurrence.
        seen_names: set = set()
        deduped = []
        for r in results:
            name = r.get("metadata", {}).get("name", r["operation_id"])
            if name not in seen_names:
                seen_names.add(name)
                deduped.append(r)
        results = deduped

        if include_full_operation:
            for result in results:
                op_id = result["operation_id"]
                operation = self.registry.get_operation(op_id)
                if operation:
                    result["operation"] = operation

        logger.debug(f"Found {len(results)} results for query")
        return results

    def _apply_world_constraints(
        self, results: List[Dict[str, Any]], robot_id: str, query: str
    ) -> List[Dict[str, Any]]:
        assert self._world_state is not None
        if not results:
            return results

        reachable_objs = self._world_state.get_reachable_objects(robot_id)
        reachable_obj_ids = {obj.object_id for obj in reachable_objs}

        robot_state = self._world_state.get_robot_state(robot_id)
        all_objects = self._world_state.get_all_objects()

        adjusted_results = []
        for result in results:
            score = result["score"]
            op_id = result["operation_id"]
            metadata = result.get("metadata", {})
            op_name = metadata.get("name", op_id)

            query_lower = query.lower()
            affected_by_constraints = False

            for obj in all_objects:
                obj_name_lower = obj.object_id.lower()
                if obj_name_lower in query_lower:
                    affected_by_constraints = True

                    # Downrank if object is stale (cap score at medium confidence tier)
                    if obj.stale:
                        score *= RAG_CONFIDENCE_TIERS["medium"]
                        logger.debug(
                            f"Downranked {op_name} (target object {obj.object_id} is stale)"
                        )

                    # Downrank if object grasped by another robot (cap at low confidence tier)
                    if (
                        obj.grasped_by is not None
                        and obj.grasped_by != robot_id
                        and "grasp" in op_name.lower()
                    ):
                        score *= RAG_CONFIDENCE_TIERS["low"]
                        logger.debug(
                            f"Downranked {op_name} (object {obj.object_id} grasped by {obj.grasped_by})"
                        )

                    # Boost if object is reachable
                    if obj.object_id in reachable_obj_ids:
                        score *= 1.15
                        logger.debug(
                            f"Boosted {op_name} (object {obj.object_id} is reachable)"
                        )

            from config.Robot import WORKSPACE_REGIONS

            for region in WORKSPACE_REGIONS.keys():
                if region.lower().replace("_", " ") in query_lower:
                    affected_by_constraints = True
                    region_owner = self._world_state.get_workspace_owner(region)

                    # Downrank if region allocated to another robot (cap at low confidence tier)
                    if region_owner is not None and region_owner != robot_id:
                        score *= RAG_CONFIDENCE_TIERS["low"]
                        logger.debug(
                            f"Downranked {op_name} (region {region} allocated to {region_owner})"
                        )

                    # Boost if in robot's own workspace
                    if robot_state and robot_state.position:
                        robot_region = self._world_state.get_region_for_position(
                            robot_state.position
                        )
                        if robot_region == region:
                            score *= 1.1
                            logger.debug(
                                f"Boosted {op_name} (robot in target region {region})"
                            )

            result["score"] = score
            if affected_by_constraints:
                result["world_state_adjusted"] = True

            adjusted_results.append(result)

        adjusted_results.sort(key=lambda x: x["score"], reverse=True)

        return adjusted_results

    def get_operation_context(
        self, query: str, top_k: int = 3, robot_id: Optional[str] = None
    ) -> Dict[str, Any]:
        results = self.search(
            query, top_k=top_k, include_full_operation=True, robot_id=robot_id
        )

        operations_context = []
        for result in results:
            operation = result.get("operation")
            if operation:
                context = {
                    "operation_id": operation.operation_id,
                    "name": operation.name,
                    "category": operation.category.value,
                    "complexity": operation.complexity.value,
                    "description": operation.description,
                    "parameters": [
                        {
                            "name": p.name,
                            "type": p.type,
                            "description": p.description,
                            "required": p.required,
                            "default": p.default,
                            "valid_range": p.valid_range,
                        }
                        for p in operation.parameters
                    ],
                    "usage_examples": operation.usage_examples,
                    "preconditions": operation.preconditions,
                    "postconditions": operation.postconditions,
                    "failure_modes": operation.failure_modes,
                    "similarity_score": result["score"],
                    "confidence": result.get("confidence", {}),
                }

                if operation.relationships:
                    rel = operation.relationships
                    context["relationships"] = {
                        "required_operations": [
                            {
                                "operation_id": op_id,
                                "reason": rel.required_reasons.get(
                                    op_id, "Dependency required"
                                ),
                            }
                            for op_id in rel.required_operations
                        ],
                        "commonly_paired_with": [
                            {
                                "operation_id": op_id,
                                "reason": rel.pairing_reasons.get(
                                    op_id, "Often used together"
                                ),
                            }
                            for op_id in rel.commonly_paired_with
                        ],
                        "parameter_flows": [
                            {
                                "from": f"{pf.source_operation}.{pf.source_output_key}",
                                "to": f"{pf.target_operation}.{pf.target_input_param}",
                                "description": pf.description,
                            }
                            for pf in rel.parameter_flows
                        ],
                        "typical_before": rel.typical_before,
                        "typical_after": rel.typical_after,
                        "coordination_requirements": rel.coordination_requirements,
                    }

                operations_context.append(context)

        result = {
            "query": query,
            "num_results": len(operations_context),
            "summary": f"Found {len(operations_context)} relevant operations for: {query}",
            "operations": operations_context,
        }

        if self._world_state and robot_id:
            result["world_state"] = self._world_state.get_world_context_string(robot_id)
            logger.debug(f"Injected world state context for {robot_id}")

        return result

    def search_by_category(
        self, category: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self.search(
            query=f"{category} operations",
            top_k=top_k or 50,  # Large number to get all in category
            category_filter=category,
        )

    def find_similar_operations(
        self, operation_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        op_data = self.vector_store.get_operation(operation_id)
        if not op_data:
            logger.warning(f"Operation '{operation_id}' not found in vector store")
            return []

        results = self.vector_store.search(
            query_embedding=op_data["embedding"], top_k=top_k + 1  # +1 to exclude self
        )

        return [r for r in results if r["operation_id"] != operation_id][:top_k]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store_stats": self.vector_store.get_stats(),
            "embedding_generator": repr(self.embedding_generator),
            "using_lm_studio": self.embedding_generator.is_using_lm_studio(),
        }
