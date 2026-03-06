#!/usr/bin/env python3
"""
LoggingSetup.py - Centralized logging configuration for all servers and modules

This module provides a single function to configure Python's logging system
with both console and optional file output, including log rotation.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Import config
try:
    from config.Servers import (
        LOG_LEVEL,
        LOG_FORMAT,
        LOG_DIR,
        ENABLE_FILE_LOGGING,
        LOG_FILE_MAX_BYTES,
        LOG_FILE_BACKUP_COUNT,
    )
except ImportError:
    from ..config.Servers import (
        LOG_LEVEL,
        LOG_FORMAT,
        LOG_DIR,
        ENABLE_FILE_LOGGING,
        LOG_FILE_MAX_BYTES,
        LOG_FILE_BACKUP_COUNT,
    )

# Global flag to track if logging has been configured
_logging_configured = False


def setup_logging(module_name: Optional[str] = None) -> logging.Logger:
    """
    Configure centralized logging with console and optional file output.

    This function should be called once at the start of each server/orchestrator.
    It configures the root logger with handlers for both console and file output
    (if enabled in config), ensuring all subsequent logging calls use the same
    configuration.

    Args:
        module_name: Optional module name for creating a per-module logger.
                    If None, returns the root logger.

    Returns:
        logging.Logger: Configured logger instance (module-specific or root)

    Example:
        # In server initialization:
        logger = setup_logging(__name__)
        logger.info("Server starting...")
    """
    global _logging_configured

    # Only configure root logger once to avoid duplicate handlers
    if not _logging_configured:
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, LOG_LEVEL))

        # Clear any existing handlers to avoid duplicates
        root_logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, LOG_LEVEL))
        console_formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Create file handler if enabled
        if ENABLE_FILE_LOGGING:
            try:
                # Create log directory if it doesn't exist
                log_dir = Path(LOG_DIR)
                log_dir.mkdir(parents=True, exist_ok=True)

                # Use a fixed filename so RotatingFileHandler manages rotation
                # via the standard .1, .2 ... .N suffix mechanism.
                log_file_path = log_dir / "server_logs.txt"

                # Create rotating file handler
                file_handler = RotatingFileHandler(
                    filename=str(log_file_path),
                    maxBytes=LOG_FILE_MAX_BYTES,
                    backupCount=LOG_FILE_BACKUP_COUNT,
                    encoding="utf-8",
                )
                file_handler.setLevel(getattr(logging, LOG_LEVEL))
                file_formatter = logging.Formatter(LOG_FORMAT)
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)

                # Log to console that file logging is enabled
                root_logger.info(f"File logging enabled: {log_file_path}")
                root_logger.info(
                    f"Log rotation: {LOG_FILE_MAX_BYTES / (1024*1024):.1f}MB max, {LOG_FILE_BACKUP_COUNT} backups"
                )

            except Exception as e:
                # If file logging fails, continue with console only
                root_logger.error(f"Failed to setup file logging: {e}")
                root_logger.warning("Continuing with console logging only")

        _logging_configured = True

    # Return module-specific logger or root logger
    if module_name:
        return logging.getLogger(module_name)
    else:
        return logging.getLogger()


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    This is a convenience function for getting a module-specific logger
    after setup_logging() has been called. If setup_logging() hasn't been
    called yet, this will call it automatically.

    Args:
        module_name: Module name (typically __name__)

    Returns:
        logging.Logger: Module-specific logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing request...")
    """
    if not _logging_configured:
        setup_logging()

    return logging.getLogger(module_name)

class WebSocketLogHandler(logging.Handler):
    """
    A custom logging handler that broadcasts log records to a generic callback.
    Used by WebUIServer to stream live logs to the frontend UI.
    """
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname.lower()
            if level == 'warning': level = 'warning'
            elif level in ['error', 'critical']: level = 'error'
            else: level = 'info'
            
            # Non-blocking callback
            self.callback(msg, level)
        except Exception:
            self.handleError(record)

def add_websocket_handler(callback):
    """Adds the websocket broadcast handler to the root logger."""
    root_logger = logging.getLogger()
    handler = WebSocketLogHandler(callback)
    # Only send INFO and above to the UI to avoid flooding
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    return handler
