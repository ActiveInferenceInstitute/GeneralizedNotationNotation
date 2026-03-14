#!/usr/bin/env python3
"""
Backward-compatibility shim — real implementation moved to module_info.py.
"""
from .module_info import get_module_info, get_audio_generation_options, register_tools

__all__ = ["get_module_info", "get_audio_generation_options", "register_tools"]
