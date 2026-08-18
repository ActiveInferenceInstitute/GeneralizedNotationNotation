#!/usr/bin/env python3
"""Shared helpers for RxInfer strategy code-generation modules."""

from __future__ import annotations

from datetime import datetime


def now() -> str:
    """Return a timestamp string for generated-script headers."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
