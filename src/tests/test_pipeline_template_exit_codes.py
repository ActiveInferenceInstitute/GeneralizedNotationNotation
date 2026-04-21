#!/usr/bin/env python3
"""Tests for the widened exit-code contract in utils.pipeline_template.

Contract under test (src/utils/pipeline_template.py::_coerce_exit_code):
  bool  -> True→0, False→1
  int   -> passthrough (0, 1, 2, ...); 2 emits a warning log line
  other -> coerced via bool()
"""

import logging
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable (mirrors conftest.py pattern)
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.pipeline_template import _coerce_exit_code  # noqa: E402


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test_pipeline_template_exit_codes")


def test_bool_true_maps_to_zero(logger):
    assert _coerce_exit_code(True, "test_step", logger) == 0


def test_bool_false_maps_to_one(logger):
    assert _coerce_exit_code(False, "test_step", logger) == 1


def test_int_zero_passes_through(logger):
    assert _coerce_exit_code(0, "test_step", logger) == 0


def test_int_one_passes_through(logger):
    assert _coerce_exit_code(1, "test_step", logger) == 1


def test_int_two_passes_through_and_warns(caplog, logger):
    caplog.set_level(logging.WARNING)
    assert _coerce_exit_code(2, "test_step", logger) == 2
    # The exact wording is not contracted, but a warning MUST be emitted
    # so operators can see the "completed with warnings" signal.
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_int_three_passes_through(logger):
    # Unusual but valid — the contract says "int passthrough".
    assert _coerce_exit_code(3, "test_step", logger) == 3


def test_none_coerces_to_one(logger):
    # None is falsy, so it maps to exit-code 1 for backward compatibility.
    assert _coerce_exit_code(None, "test_step", logger) == 1


def test_truthy_string_coerces_to_zero(logger):
    assert _coerce_exit_code("ok", "test_step", logger) == 0


def test_empty_string_coerces_to_one(logger):
    assert _coerce_exit_code("", "test_step", logger) == 1
