#!/usr/bin/env python3
"""SAPF module information facade."""

from .module_info import get_audio_generation_options, get_module_info, register_tools

__all__ = ["get_module_info", "get_audio_generation_options", "register_tools"]
