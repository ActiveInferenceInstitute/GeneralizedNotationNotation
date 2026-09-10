"""MAJ-05 regression tests: deprecated ``validate_gnn*`` aliases.

Each deprecated name must (a) emit a ``DeprecationWarning`` when called,
(b) return exactly what its canonical replacement returns for the same
input, and (c) remain importable from its historical module. Canonical
behavior itself is pinned by the module test suites; this file pins the
alias contract only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import gnn
from gnn.execute.pymdp.pymdp_utils import (
    check_gnn_pomdp_spec,
    validate_gnn_pomdp_structure,
)
from gnn.mcp.processors import (
    check_cross_format_consistency,
    validate_gnn_cross_format_consistency,
)
from gnn.parsers.basic import (
    validate_gnn,
    validate_gnn_syntax,
    validate_gnn_syntax_formal,
)
from gnn.pipeline.config import get_output_dir_for_script
from gnn.processing.processor import (
    check_gnn_file_structure,
    validate_gnn_structure,
)
from gnn.schema_validator import (
    validate_gnn_file as schema_validate_gnn_file,
)
from gnn.schema_validator import (
    validate_gnn_file_comprehensive,
)
from gnn.validation.simple import (
    check_gnn_directory_basic,
    check_gnn_file_basic,
    validate_gnn_directory,
    validate_gnn_file,
)


def test_parsers_syntax_formal_alias_matches_canonical() -> None:
    content = "not a gnn file"
    with pytest.warns(DeprecationWarning):
        old = validate_gnn_syntax_formal(content)
    assert old == validate_gnn_syntax(content)


def test_parsers_validate_gnn_alias_matches_canonical() -> None:
    content = "not a gnn file"
    with pytest.warns(DeprecationWarning):
        old = validate_gnn(content)
    assert old == validate_gnn_syntax(content)


def test_check_gnn_file_structure_alias_matches_canonical(tmp_path: Path) -> None:
    model = tmp_path / "m.md"
    model.write_text(
        "## ModelName\nM\n\n## StateSpaceBlock\nS_f: [3]\n\n## Connections\n",
        encoding="utf-8",
    )
    with pytest.warns(DeprecationWarning):
        old = validate_gnn_structure(model)
    new = check_gnn_file_structure(model)
    old.pop("validation_timestamp")
    new.pop("validation_timestamp")
    assert old == new

    with pytest.warns(DeprecationWarning):
        old_content = validate_gnn_structure("## ModelName\nM")
    new_content = check_gnn_file_structure("## ModelName\nM")
    old_content.pop("validation_timestamp")
    new_content.pop("validation_timestamp")
    assert old_content == new_content


def test_package_root_lazy_export_still_deprecated() -> None:
    with pytest.warns(DeprecationWarning):
        result = gnn.validate_gnn_structure("## ModelName\nM")
    assert isinstance(result, dict)


def test_check_gnn_pomdp_spec_alias_matches_canonical() -> None:
    spec = {
        "model_name": "t",
        "initial_parameterization": {"A": {"matrix": [[1.0]]}},
    }
    with pytest.warns(DeprecationWarning):
        old = validate_gnn_pomdp_structure(spec)
    assert old == check_gnn_pomdp_spec(spec)


def test_simple_file_alias_matches_canonical(tmp_path: Path) -> None:
    model = tmp_path / "a.md"
    model.write_text("## ModelName\nA\n", encoding="utf-8")
    with pytest.warns(DeprecationWarning):
        old = validate_gnn_file(model)
    assert old == check_gnn_file_basic(model)


def test_simple_directory_alias_matches_canonical(tmp_path: Path) -> None:
    with pytest.warns(DeprecationWarning):
        old = validate_gnn_directory(tmp_path)
    assert old == check_gnn_directory_basic(tmp_path)


def test_schema_validator_alias_matches_canonical(tmp_path: Path) -> None:
    model = tmp_path / "s.md"
    model.write_text("## ModelName\nS\n", encoding="utf-8")
    with pytest.warns(DeprecationWarning):
        old = schema_validate_gnn_file(model)
    new = validate_gnn_file_comprehensive(model)
    assert old.is_valid == new.is_valid
    assert old.errors == new.errors


def test_cross_format_alias_matches_canonical(tmp_path: Path) -> None:
    target = tmp_path / "in"
    target.mkdir()
    out_old = tmp_path / "out_old"
    out_old.mkdir()
    out_new = tmp_path / "out_new"
    out_new.mkdir()
    with pytest.warns(DeprecationWarning):
        old = validate_gnn_cross_format_consistency(target, out_old)
    new = check_cross_format_consistency(target, out_new)
    assert old == new is True


def test_llm_module_alias_forwards_to_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn.llm.llm_operations as llo

    calls: list[str] = []
    monkeypatch.setattr(
        llo, "validate_gnn_with_llm", lambda content: calls.append(content) or "ok"
    )
    with pytest.warns(DeprecationWarning):
        result = llo.validate_gnn("content")
    assert result == "ok"
    assert calls == ["content"]


def test_aliases_match_canonical_on_invalid_input_and_warn_at_caller(
    tmp_path: Path,
) -> None:
    """Error-path parity: each deprecated alias returns exactly what its
    canonical replacement returns for the same invalid input, and the
    DeprecationWarning attributes to the caller (``stacklevel=2``), not to
    the alias module or the package-root lazy export."""
    missing_file = tmp_path / "does-not-exist.md"
    missing_dir = tmp_path / "does-not-exist-dir"
    invalid_spec: dict[str, Any] = {"unexpected_key": True}

    def _normalize_structure(result: dict[str, Any]) -> dict[str, Any]:
        result.pop("validation_timestamp", None)
        return result

    cases: list[
        tuple[str, Callable[[], Any], Callable[[], Any], Callable[[Any], Any]]
    ] = [
        (
            "parsers.validate_gnn_syntax_formal",
            lambda: validate_gnn_syntax_formal("not a gnn file"),
            lambda: validate_gnn_syntax("not a gnn file"),
            lambda r: r,
        ),
        (
            "parsers.validate_gnn",
            lambda: validate_gnn("not a gnn file"),
            lambda: validate_gnn_syntax("not a gnn file"),
            lambda r: r,
        ),
        (
            "package.validate_gnn_file",
            lambda: gnn.validate_gnn_file("not a gnn file"),
            lambda: gnn.validate_gnn_source("not a gnn file"),
            lambda r: r,
        ),
        (
            "processor.validate_gnn_structure",
            lambda: validate_gnn_structure(missing_file),
            lambda: check_gnn_file_structure(missing_file),
            _normalize_structure,
        ),
        (
            "pomdp.validate_gnn_pomdp_structure",
            lambda: validate_gnn_pomdp_structure(invalid_spec),
            lambda: check_gnn_pomdp_spec(invalid_spec),
            lambda r: r,
        ),
        (
            "simple.validate_gnn_file",
            lambda: validate_gnn_file(missing_file),
            lambda: check_gnn_file_basic(missing_file),
            lambda r: r,
        ),
        (
            "simple.validate_gnn_directory",
            lambda: validate_gnn_directory(missing_dir),
            lambda: check_gnn_directory_basic(missing_dir),
            lambda r: r,
        ),
        (
            "schema.validate_gnn_file",
            lambda: schema_validate_gnn_file(missing_file),
            lambda: validate_gnn_file_comprehensive(missing_file),
            lambda r: (r.is_valid, r.errors),
        ),
        (
            "mcp.validate_gnn_cross_format_consistency",
            lambda: validate_gnn_cross_format_consistency(
                missing_dir, tmp_path / "out-x"
            ),
            lambda: check_cross_format_consistency(missing_dir, tmp_path / "out-y"),
            lambda r: r,
        ),
    ]
    for name, alias_call, canonical_call, normalize in cases:
        with pytest.warns(DeprecationWarning) as caught:
            old = alias_call()
        new = normalize(canonical_call())
        assert normalize(old) == new, name
        assert caught[0].filename == __file__, name


def test_package_root_syntax_formal_lazy_export_still_deprecated() -> None:
    with pytest.warns(DeprecationWarning) as caught:
        result = gnn.validate_gnn_syntax_formal("not a gnn file")
    assert result == validate_gnn_syntax("not a gnn file")
    # The lazy re-export resolves to the parsers.basic alias wrapper whose
    # ``stacklevel=2`` must attribute the warning to this caller.
    assert caught[0].filename == __file__


def test_pipeline_template_output_dir_reexport_warns(tmp_path: Path) -> None:
    """The legacy ``gnn.utils.pipeline_template`` re-export of the canonical
    ``gnn.pipeline.config.get_output_dir_for_script`` must warn and forward.
    """
    import gnn.utils.pipeline_template as template

    with pytest.warns(DeprecationWarning):
        legacy = template.get_output_dir_for_script  # noqa: B018
    assert callable(legacy)
    assert legacy("3_gnn.py", tmp_path) == get_output_dir_for_script("3_gnn.py", tmp_path)
