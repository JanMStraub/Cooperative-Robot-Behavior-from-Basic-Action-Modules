"""
Main Entry Point Orchestrators

This package contains orchestrator scripts that coordinate multiple
servers and processing modules for complete pipelines.

Orchestrators:
- RunAnalyzer: LLM vision pipeline (StreamingServer + ResultsServer + Ollama)
- RunDetector: Single-camera object detection pipeline
- RunStereoDetector: Stereo detection pipeline with 3D depth
"""

# Note: Orchestrators are typically run as scripts, not imported as modules
# They can be executed directly: python -m orchestrators.RunAnalyzer

__all__ = []
