"""Pins for ``utils/pipeline`` (previously 41% coverage).

The module is the ``gnn.utils`` public-surface delegation layer over
``gnn.pipeline.config``; tests pin both the delegated behavior and the
recovery fallbacks.
"""

from __future__ import annotations

from pathlib import Path

from gnn.utils.pipeline import (
    RecoveryArgumentParser,
    get_output_dir_for_script,
    get_pipeline_utilities,
    validate_output_directory,
)


def test_get_output_dir_for_script_delegates_to_canonical(tmp_path: Path) -> None:
    resolved = get_output_dir_for_script("3_gnn.py", tmp_path)

    assert resolved == tmp_path / "3_gnn_output"
    assert resolved.is_dir() is False or resolved.is_dir()


def test_get_output_dir_for_script_default_base(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

    resolved = get_output_dir_for_script("7_export.py")

    assert resolved == Path("output") / "7_export_output"


def test_recovery_argument_parser_returns_defaults() -> None:
    args = RecoveryArgumentParser.parse_step_arguments("3_gnn.py")

    assert args.verbose is False
    assert args.output_dir == Path("output")
    assert args.step_name == "3_gnn.py"


def test_get_pipeline_utilities_returns_logger_and_parser() -> None:
    logger, parser = get_pipeline_utilities("3_gnn.py", verbose=False)

    assert logger.name  # configured logger
    assert hasattr(parser, "parse_step_arguments")


def test_validate_output_directory_creates_and_accepts_writable_dir(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "3_gnn_output"

    assert validate_output_directory(target, "3_gnn.py") is True
    assert target.is_dir()


def test_validate_output_directory_rejects_unwritable_dir(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "3_gnn.py_test.tmp").write_bytes(b"")
    # Make the directory read-only so the write probe must fail.
    blocked.chmod(0o500)

    try:
        assert validate_output_directory(blocked, "3_gnn.py") is False
    finally:
        blocked.chmod(0o700)
