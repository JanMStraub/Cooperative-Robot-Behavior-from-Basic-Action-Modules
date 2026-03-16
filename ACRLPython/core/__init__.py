#!/usr/bin/env python3
"""
Core infrastructure for Unity-Python communication.

This package provides base classes and protocol definitions for TCP communication
between Unity and Python servers.
"""

from .TCPServerBase import TCPServerBase, ServerConfig
from .UnityProtocol import UnityProtocol

__all__ = ["TCPServerBase", "ServerConfig", "UnityProtocol"]
