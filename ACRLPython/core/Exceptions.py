#!/usr/bin/env python3
"""
Exceptions - ACRL exception hierarchy.

Provides typed exceptions to replace bare ``except Exception:`` blocks
throughout the codebase. Callers should catch the most specific subclass
relevant to their context, and let ``ACRLError`` bubble up for generic
ACRL-layer failures.

Hierarchy::

    ACRLError
    ├── CommunicationError   TCP/protocol-level errors (recoverable)
    ├── OperationError       Operation execution failures
    ├── ROSError             ROS bridge / MoveIt errors
    └── ConfigurationError  Invalid or missing configuration
"""


class ACRLError(Exception):
    """Base class for all ACRL exceptions."""


class CommunicationError(ACRLError):
    """
    TCP/protocol-level error.

    Raised when a network connection fails, a message cannot be sent, or
    a protocol framing error is encountered. These errors are typically
    recoverable — the server can log a warning and continue serving other
    clients.
    """


class OperationError(ACRLError):
    """
    Operation execution failure.

    Raised when a robot operation (move, grasp, gripper, etc.) fails
    due to bad parameters, unreachable targets, or Unity reporting an
    error result.
    """


class ROSError(ACRLError):
    """
    ROS bridge / MoveIt error.

    Raised when the ROS bridge connection fails or MoveIt planning/execution
    returns an error. In hybrid mode the caller should fall back to TCP.
    """


class ConfigurationError(ACRLError):
    """
    Configuration error.

    Raised when a required configuration value is missing, has an invalid
    type, or is out of its legal range.
    """
