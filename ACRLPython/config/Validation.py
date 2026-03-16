#!/usr/bin/env python3
"""
Configuration Validation
=========================

Validates configuration values to catch common errors early.
"""

import logging
import warnings
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _validate_range(name: str, value: float, min_val: float, max_val: float) -> bool:
    """Validate that a value is within a range."""
    if not (min_val <= value <= max_val):
        warnings.warn(
            f"{name}={value} is outside valid range [{min_val}, {max_val}]. "
            f"This may cause unexpected behavior."
        )
        return False
    return True


def _validate_port(name: str, port: int) -> bool:
    """Validate that a port number is valid."""
    if not (1024 <= port <= 65535):
        warnings.warn(
            f"{name}={port} is outside valid port range [1024, 65535]. "
            f"Using system ports (<1024) requires root privileges."
        )
        return False
    return True


def _validate_positive(name: str, value: float) -> bool:
    """Validate that a value is positive."""
    if value <= 0:
        warnings.warn(f"{name}={value} must be positive.")
        return False
    return True


def validate_config(config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, list]:
    """
    Validate configuration values.

    Args:
        config_dict: Dictionary of config values to validate (default: current module)

    Returns:
        Dict with 'errors' and 'warnings' lists
    """
    errors = []
    warnings_list = []

    # If no config_dict provided, import from parent modules
    if config_dict is None:
        try:
            from . import Servers, Vision, Rag, Robot

            config_dict = {
                **vars(Servers),
                **vars(Vision),
                **vars(Rag),
                **vars(Robot),
            }
        except ImportError:
            errors.append("Failed to import config modules")
            return {"errors": errors, "warnings": warnings_list}

    # ========================================================================
    # Port Validations
    # ========================================================================
    port_configs = [
        "STREAMING_SERVER_PORT",
        "STEREO_DETECTION_PORT",
        "LLM_RESULTS_PORT",
        "DEPTH_RESULTS_PORT",
        "RAG_SERVER_PORT",
        "STATUS_SERVER_PORT",
        "SEQUENCE_SERVER_PORT",
    ]

    for port_name in port_configs:
        if port_name in config_dict:
            if not _validate_port(port_name, config_dict[port_name]):
                warnings_list.append(f"Invalid port: {port_name}")

    # Check for port conflicts
    ports = [config_dict.get(name) for name in port_configs if name in config_dict]
    if len(ports) != len(set(ports)):
        errors.append("Port conflict detected: Multiple servers assigned the same port")

    # ========================================================================
    # Threshold Validations (0.0-1.0)
    # ========================================================================
    threshold_configs = {
        "YOLO_CONFIDENCE_THRESHOLD": (0.0, 1.0),
        "YOLO_IOU_THRESHOLD": (0.0, 1.0),
        "MIN_CONFIDENCE": (0.0, 1.0),
        "RAG_MIN_SIMILARITY_SCORE": (0.0, 1.0),
        "STEREO_MIN_IOU": (0.0, 1.0),
        "TRACKING_MIN_IOU": (0.0, 1.0),
    }

    for name, (min_val, max_val) in threshold_configs.items():
        if name in config_dict:
            if not _validate_range(name, config_dict[name], min_val, max_val):
                warnings_list.append(f"Threshold out of range: {name}")

    # ========================================================================
    # Positive Value Validations
    # ========================================================================
    positive_configs = [
        "MIN_IMAGE_AGE",
        "MAX_IMAGE_AGE",
        "IMAGE_CHECK_INTERVAL",
        "DEFAULT_STEREO_BASELINE",
        "DEFAULT_STEREO_FOV",
        "MIN_CUBE_AREA_PX",
        "MAX_CUBE_AREA_PX",
        "VISION_STREAM_FPS",
        "COLLISION_SAFETY_MARGIN",
        "MIN_ROBOT_SEPARATION",
        "MAX_ROBOT_REACH",
    ]

    for name in positive_configs:
        if name in config_dict and not _validate_positive(name, config_dict[name]):
            warnings_list.append(f"Value must be positive: {name}")

    # ========================================================================
    # Logical Consistency Checks
    # ========================================================================

    # MIN_IMAGE_AGE < MAX_IMAGE_AGE
    if "MIN_IMAGE_AGE" in config_dict and "MAX_IMAGE_AGE" in config_dict:
        if config_dict["MIN_IMAGE_AGE"] >= config_dict["MAX_IMAGE_AGE"]:
            errors.append("MIN_IMAGE_AGE must be less than MAX_IMAGE_AGE")

    # MIN_CUBE_AREA_PX < MAX_CUBE_AREA_PX
    if "MIN_CUBE_AREA_PX" in config_dict and "MAX_CUBE_AREA_PX" in config_dict:
        if config_dict["MIN_CUBE_AREA_PX"] >= config_dict["MAX_CUBE_AREA_PX"]:
            errors.append("MIN_CUBE_AREA_PX must be less than MAX_CUBE_AREA_PX")

    # MIN_ASPECT_RATIO < MAX_ASPECT_RATIO
    if "MIN_ASPECT_RATIO" in config_dict and "MAX_ASPECT_RATIO" in config_dict:
        if config_dict["MIN_ASPECT_RATIO"] >= config_dict["MAX_ASPECT_RATIO"]:
            errors.append("MIN_ASPECT_RATIO must be less than MAX_ASPECT_RATIO")

    # Duplicate debug directories warning (from original analysis)
    if "DEBUG_IMAGES_DIR" in config_dict and "DEBUG_DISPARITY_DIR" in config_dict:
        if config_dict["DEBUG_IMAGES_DIR"] == config_dict["DEBUG_DISPARITY_DIR"]:
            warnings_list.append(
                "DEBUG_IMAGES_DIR and DEBUG_DISPARITY_DIR point to the same path. "
                "Consider using separate directories for better organization."
            )

    # ========================================================================
    # Report Results
    # ========================================================================
    if errors:
        logger.error(f"Configuration validation found {len(errors)} errors:")
        for error in errors:
            logger.error(f"  - {error}")

    if warnings_list:
        logger.warning(f"Configuration validation found {len(warnings_list)} warnings:")
        for warning in warnings_list:
            logger.warning(f"  - {warning}")

    if not errors and not warnings_list:
        logger.info("✓ Configuration validation passed")

    return {"errors": errors, "warnings": warnings_list}
