#!/usr/bin/env python3
"""Phase 4.2 regression tests for template (Step 0).

Zero-mock per CLAUDE.md — uses real filesystem tempdirs.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_template_module_info_has_version():
    from template import get_module_info
    info = get_module_info()
    assert isinstance(info, dict)
    assert "version" in info


def test_process_template_standardized_accepts_standard_signature(tmp_path):
    """The pipeline orchestrator calls every processor with the same
    (target_dir, output_dir, logger, ...) keyword contract. Regression
    guard: process_template_standardized must honor this without raising."""
    import logging
    from template.processor import process_template_standardized
    logger = logging.getLogger("test_template")
    # Create a minimal input dir so the processor has something to walk.
    target = tmp_path / "in"
    target.mkdir()
    output = tmp_path / "out"
    # Must not raise TypeError from bad signature wiring.
    try:
        result = process_template_standardized(
            target_dir=target,
            output_dir=output,
            logger=logger,
            verbose=False,
        )
    except TypeError as e:
        pytest.fail(f"process_template_standardized signature drift: {e}")
    # Result may be bool or int (widened contract). Either is acceptable.
    assert isinstance(result, (bool, int)) or result is None


def test_validate_file_returns_structured_dict(tmp_path):
    """validate_file must return {valid, ...} shape, not raise, for any
    readable file — including empty ones."""
    from template.processor import validate_file
    empty = tmp_path / "empty.md"
    empty.write_text("")
    result = validate_file(empty)
    assert isinstance(result, dict)
    # Should expose a boolean-ish validity signal.
    assert any(k in result for k in ("valid", "is_valid", "status", "success"))


def test_validate_file_handles_missing_file(tmp_path):
    """validate_file on a missing path must report failure, not crash."""
    from template.processor import validate_file
    missing = tmp_path / "does_not_exist.md"
    result = validate_file(missing)
    assert isinstance(result, dict)
    # Should flag as invalid.
    invalidness = (result.get("valid") is False
                   or result.get("is_valid") is False
                   or result.get("success") is False
                   or "error" in result
                   or "errors" in result)
    assert invalidness, f"validate_file succeeded on missing file: {result}"


def test_generate_correlation_id_produces_unique_ids():
    from template.processor import generate_correlation_id
    ids = {generate_correlation_id() for _ in range(10)}
    # At least 8 distinct IDs out of 10 — tolerates tiny collision risk in
    # time-based schemes, but a literal same-string-every-time bug would fail.
    assert len(ids) >= 8, f"generate_correlation_id producing duplicates: {ids}"
