#!/usr/bin/env python3
"""Centralized logging configuration: console output + optional file rotation."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from config.Servers import (
        LOG_LEVEL,
        LOG_FORMAT,
        LOG_DIR,
        ENABLE_FILE_LOGGING,
        LOG_FILE_BACKUP_COUNT,
    )
except ImportError:
    from ..config.Servers import (
        LOG_LEVEL,
        LOG_FORMAT,
        LOG_DIR,
        ENABLE_FILE_LOGGING,
        LOG_FILE_BACKUP_COUNT,
    )

# Global flag to track if logging has been configured
_logging_configured = False


def setup_logging(module_name: Optional[str] = None) -> logging.Logger:
    """
    Configure centralized logging (idempotent). Returns a module-specific or root logger.

    Call once per server/orchestrator; subsequent calls are no-ops on the root config.

    Example:
        logger = setup_logging(__name__)
        logger.info("Server starting...")
    """
    global _logging_configured

    if not _logging_configured:
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, LOG_LEVEL))
        root_logger.handlers.clear()  # Avoid duplicate handlers on re-import.

        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, LOG_LEVEL))
        console_formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        _logging_configured = True

    if module_name:
        return logging.getLogger(module_name)
    else:
        return logging.getLogger()


def get_logger(module_name: str) -> logging.Logger:
    """Get a module-specific logger, calling setup_logging() if not yet configured."""
    if not _logging_configured:
        setup_logging()

    return logging.getLogger(module_name)


def enable_file_logging() -> None:
    """
    Attach a file handler to the root logger.

    Call this only after all servers have successfully started to avoid
    creating a log file for runs that exit during startup. Has no effect
    if ENABLE_FILE_LOGGING is False or if a file handler is already attached.
    """
    if not ENABLE_FILE_LOGGING:
        return

    root_logger = logging.getLogger()

    if any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        return

    try:
        log_dir = Path(LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        existing_logs = sorted(
            log_dir.glob("server_logs_*.txt"), key=lambda p: p.stat().st_mtime
        )
        for old_log in existing_logs[
            : max(0, len(existing_logs) - LOG_FILE_BACKUP_COUNT + 1)
        ]:
            old_log.unlink()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_dir / f"server_logs_{timestamp}.txt"

        file_handler = logging.FileHandler(
            filename=str(log_file_path), encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, LOG_LEVEL))
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(file_handler)

        root_logger.info(f"File logging enabled: {log_file_path}")

    except Exception as e:
        root_logger = logging.getLogger()
        root_logger.error(f"Failed to setup file logging: {e}")
        root_logger.warning("Continuing with console logging only")


_patched_handlers: set = set()  # Track which handlers have been patched


class SafeStreamHandler(logging.StreamHandler):
    """
    StreamHandler that silently ignores I/O errors from closed streams.

    Prevents spurious errors when pytest closes handlers before background threads finish.
    """

    def emit(self, record):
        try:
            super().emit(record)
        except (ValueError, OSError):
            pass

    def handleError(self, record):
        import sys

        if sys.exc_info()[0] in (ValueError, OSError):
            pass
        else:
            super().handleError(record)


def _safe_log(log_func, message: str, *args, **kwargs):
    """
    Safely log a message, patching any new handlers added dynamically (e.g. by pytest).

    Lazily patches handlers to suppress I/O errors from closed streams.
    """
    import logging as _logging

    root_logger = _logging.getLogger()
    calling_logger = log_func.__self__ if hasattr(log_func, "__self__") else root_logger
    all_handlers = root_logger.handlers + getattr(calling_logger, "handlers", [])
    for handler in all_handlers:
        if id(handler) not in _patched_handlers:
            _make_handler_safe(handler)
            _patched_handlers.add(id(handler))
    try:
        log_func(message, *args, **kwargs)
    except (ValueError, OSError):
        pass


def _make_handler_safe(handler):
    if not hasattr(handler, "_original_emit"):
        handler._original_emit = handler.__class__.emit
        handler._original_handleError = handler.__class__.handleError

    def safe_emit(record):
        try:
            handler._original_emit(handler, record)
        except (ValueError, OSError, RuntimeError):
            pass

    def safe_handleError(record):
        import sys

        exc_type = sys.exc_info()[0]
        if exc_type in (ValueError, OSError, RuntimeError):
            pass
        else:
            handler._original_handleError(handler, record)

    handler.emit = safe_emit
    handler.handleError = safe_handleError
    return handler


class WebSocketLogHandler(logging.Handler):
    """Broadcasts log records via callback; used by WebUIServer to stream logs to the frontend."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname.lower()
            if level == "warning":
                level = "warning"
            elif level in ["error", "critical"]:
                level = "error"
            else:
                level = "info"

            self.callback(msg, level)
        except Exception:
            self.handleError(record)


def add_websocket_handler(callback):
    """Add the WebSocket broadcast handler to the root logger."""
    root_logger = logging.getLogger()
    handler = WebSocketLogHandler(callback)
    handler.setLevel(logging.INFO)  # INFO+ only to avoid flooding the UI.
    root_logger.addHandler(handler)
    return handler
