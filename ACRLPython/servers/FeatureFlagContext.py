"""Apply BenchmarkFeatureFlags to server process; restores on exit."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator

from benchmarks.FeatureFlags import BenchmarkFeatureFlags

logger = logging.getLogger(__name__)


@contextmanager
def FeatureFlagContext(flags: BenchmarkFeatureFlags) -> Generator[None, None, None]:
    """Apply feature flag overrides; restored in finally block."""
    restores: Dict[str, Any] = {}
    try:
        _apply(flags, restores)
        yield
    finally:
        _restore(restores)


def _apply(flags: BenchmarkFeatureFlags, restores: Dict[str, Any]) -> None:
    import config.Servers as srv
    import config.ROS as ros
    import config.Negotiation as neg
    import orchestrators.SequenceExecutor as seq_mod

    if flags.use_vgn is not None:
        restores["srv.VGN_ENABLED"] = srv.VGN_ENABLED
        srv.VGN_ENABLED = flags.use_vgn
        logger.debug(f"[FeatureFlagContext] VGN_ENABLED -> {flags.use_vgn}")

    if flags.use_reflection is not None:
        restores["srv.REFLECTION_ENABLED"] = srv.REFLECTION_ENABLED
        restores["seq.REFLECTION_ENABLED"] = seq_mod.REFLECTION_ENABLED
        srv.REFLECTION_ENABLED = flags.use_reflection
        seq_mod.REFLECTION_ENABLED = flags.use_reflection
        logger.debug(
            f"[FeatureFlagContext] REFLECTION_ENABLED -> {flags.use_reflection}"
        )

    if flags.use_ros is not None:
        restores["ros.ROS_ENABLED"] = ros.ROS_ENABLED
        restores["ros.DEFAULT_CONTROL_MODE"] = ros.DEFAULT_CONTROL_MODE
        ros.ROS_ENABLED = flags.use_ros
        ros.DEFAULT_CONTROL_MODE = "ros" if flags.use_ros else "unity"
        logger.debug(f"[FeatureFlagContext] ROS_ENABLED -> {flags.use_ros}")

    if flags.use_negotiation is not None:
        restores["neg.NEGOTIATION_ENABLED"] = neg.NEGOTIATION_ENABLED
        neg.NEGOTIATION_ENABLED = flags.use_negotiation
        logger.debug(
            f"[FeatureFlagContext] NEGOTIATION_ENABLED -> {flags.use_negotiation}"
        )

    if flags.use_rag is not None:
        _apply_rag(flags.use_rag, restores)


def _apply_rag(use_rag: bool, restores: Dict[str, Any]) -> None:
    try:
        from core.Imports import get_command_parser
        from orchestrators.CommandParser import _PromptBuilder

        parser = get_command_parser()
        if parser is None:
            return

        restores["parser.rag"] = parser.rag
        restores["parser.prompt_builder"] = parser._prompt_builder

        if use_rag:
            if parser.rag is None:
                try:
                    from rag import RAGSystem

                    rag = RAGSystem()
                    rag.index_operations(rebuild=False)
                    parser.rag = rag
                    parser._prompt_builder = _PromptBuilder(
                        parser.registry, parser.workflow_registry, parser.rag
                    )
                except Exception as e:
                    logger.warning(f"[FeatureFlagContext] Could not enable RAG: {e}")
        else:
            parser.rag = None
            parser._prompt_builder = _PromptBuilder(
                parser.registry, parser.workflow_registry, None
            )
        logger.debug(f"[FeatureFlagContext] RAG -> {use_rag}")
    except Exception as e:
        logger.warning(f"[FeatureFlagContext] RAG toggle failed: {e}")


def _restore(restores: Dict[str, Any]) -> None:
    try:
        import config.Servers as srv
        import config.ROS as ros
        import config.Negotiation as neg
        import orchestrators.SequenceExecutor as seq_mod

        if "srv.VGN_ENABLED" in restores:
            srv.VGN_ENABLED = restores["srv.VGN_ENABLED"]
        if "srv.REFLECTION_ENABLED" in restores:
            srv.REFLECTION_ENABLED = restores["srv.REFLECTION_ENABLED"]
        if "seq.REFLECTION_ENABLED" in restores:
            seq_mod.REFLECTION_ENABLED = restores["seq.REFLECTION_ENABLED"]
        if "ros.ROS_ENABLED" in restores:
            ros.ROS_ENABLED = restores["ros.ROS_ENABLED"]
        if "ros.DEFAULT_CONTROL_MODE" in restores:
            ros.DEFAULT_CONTROL_MODE = restores["ros.DEFAULT_CONTROL_MODE"]
        if "neg.NEGOTIATION_ENABLED" in restores:
            neg.NEGOTIATION_ENABLED = restores["neg.NEGOTIATION_ENABLED"]

        if "parser.rag" in restores or "parser.prompt_builder" in restores:
            try:
                from core.Imports import get_command_parser

                parser = get_command_parser()
                if parser is not None:
                    if "parser.rag" in restores:
                        parser.rag = restores["parser.rag"]
                    if "parser.prompt_builder" in restores:
                        parser._prompt_builder = restores["parser.prompt_builder"]
            except Exception as e:
                logger.error(f"[FeatureFlagContext] Parser restore failed: {e}")
    except Exception as e:
        logger.error(f"[FeatureFlagContext] Restore failed: {e}")
