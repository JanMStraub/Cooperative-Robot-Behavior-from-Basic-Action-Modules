"""
Main Entry Point Orchestrators

This package contains orchestrator scripts that coordinate multiple
servers and processing modules for complete pipelines.

Orchestrators:
- RunSequenceServer: Main sequence execution pipeline (StreamingServer + ResultsServer + StatusServer + SequenceServer)
"""

# Note: Orchestrators are typically run as scripts, not imported as modules
# They can be executed directly: python -m orchestrators.RunSequenceServer

__all__ = []
