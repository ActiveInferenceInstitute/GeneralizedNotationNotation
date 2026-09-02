#!/usr/bin/env python3
"""Edge-case tests for utils.path_conversion.

Covers the honest utils path-conversion helpers: string->Path coercion,
None handling for critical path arguments, and the config validation entry
point. This file pins real edge behaviour that had thin dedicated coverage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.path_conversion import (
    convert_path_arguments,
    validate_and_convert_paths,
)
from utils.pipeline_arguments import PipelineArguments


class TestConvertPathArguments:
    """Edge cases for convert_path_arguments (string path -> Path)."""

    def test_converts_named_path_attributes(self) -> None:
        ns = argparse.Namespace(
            target_dir="input/gnn_files",
            output_dir="out",
            verbose=False,
            model_file="model.gnn",
        )
        convert_path_arguments(ns)
        assert isinstance(ns.target_dir, Path)
        assert isinstance(ns.output_dir, Path)
        assert isinstance(ns.model_file, Path)
        # Non-path-ish attrs are left untouched.
        assert ns.verbose is False

    def test_leaves_non_strings_alone(self) -> None:
        p = Path("x")
        ns = argparse.Namespace(target_dir=p, verbose=True)
        convert_path_arguments(ns)
        assert ns.target_dir is p  # unchanged object identity

    def test_does_not_touch_underscore_attrs(self) -> None:
        """Attributes starting with underscore are skipped (may hold private state)."""
        ns = argparse.Namespace(_hidden="/definitely/a/path", output_dir="o")
        convert_path_arguments(ns)
        assert not isinstance(ns._hidden, Path)
        assert isinstance(ns.output_dir, Path)


class TestValidateAndConvertPaths:
    """Edge cases for validate_and_convert_paths (critical path args)."""

    def _new_args(self, **overrides: Any) -> PipelineArguments:
        args = PipelineArguments()
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_converts_string_to_path(self) -> None:
        args = self._new_args(target_dir="input/raw", output_dir="custom_out")
        validate_and_convert_paths(args, logging.getLogger("test"))
        assert isinstance(args.target_dir, Path)
        assert isinstance(args.output_dir, Path)
        assert args.target_dir == Path("input/raw")

    def test_none_output_dir_raises(self) -> None:
        """A critical path arg that is None after parsing is a hard error."""
        args = self._new_args(output_dir=None)
        with pytest.raises(ValueError):
            validate_and_convert_paths(args, logging.getLogger("test"))

    def test_none_target_dir_raises(self) -> None:
        args = self._new_args(target_dir=None)
        with pytest.raises(ValueError):
            validate_and_convert_paths(args, logging.getLogger("test"))

    def test_non_path_missing_optional_arg_is_tolerated(self, caplog: Any) -> None:
        """Non-critical optional path args (e.g. ontology_terms_file=None)
        are skipped without raising; their absence is only a debug note."""
        args = self._new_args(ontology_terms_file=None, pipeline_summary_file=None)
        with caplog.at_level(logging.WARNING, logger="utils.argument_utils"):
            validate_and_convert_paths(args, logging.getLogger("test"))
        # No critical path arg was None, so no exception.
        assert args.ontology_terms_file is None

    def test_already_path_is_unchanged(self) -> None:
        p = Path("existing")
        args = self._new_args(target_dir=p, output_dir=Path("out"))
        validate_and_convert_paths(args, logging.getLogger("test"))
        assert args.target_dir is p
