"""Public and formal validation must preserve explicit source identity."""

from pathlib import Path

import pytest

from gnn import validate_gnn_source
from gnn.parsers.basic import validate_gnn_syntax
from gnn.processing.discovery import is_model_source_path

ROOT = Path(__file__).resolve().parents[2]
MODELS = sorted(
    path
    for path in (ROOT / "input/gnn_files").rglob("*.md")
    if is_model_source_path(path)
)
SAMPLE = ROOT / "input/gnn_files/discrete/actinf_pomdp_agent.md"


@pytest.mark.parametrize(
    "path", MODELS, ids=lambda path: str(path.relative_to(ROOT / "input/gnn_files"))
)
def test_all_tracked_models_preserve_file_and_content_verdicts(path):
    before = path.read_bytes()
    content = before.decode("utf-8")
    file_result = validate_gnn_source(path)
    assert file_result["is_valid"], file_result["errors"]
    assert file_result == validate_gnn_source(content, is_content=True)
    valid, errors = validate_gnn_syntax(content, is_content=True)
    assert file_result == {"is_valid": valid, "errors": errors}
    assert path.read_bytes() == before


@pytest.mark.parametrize("suffix", ("界" * 10000, "\x00", "\n"))
def test_implicit_long_or_nonpath_text_reaches_same_formal_gate(suffix):
    content = SAMPLE.read_text() + suffix
    explicit = validate_gnn_source(content, is_content=True)
    assert validate_gnn_source(content) == explicit
    valid, errors = validate_gnn_syntax(content, is_content=True)
    assert explicit == {"is_valid": valid, "errors": errors}


@pytest.mark.parametrize("value", ("", None, 42, "not a GNN document"))
def test_empty_or_nonpath_source_returns_formal_invalidity(value):
    result = validate_gnn_source(value)
    assert result["is_valid"] is False and result["errors"]


@pytest.mark.parametrize("name", ("literal-name.md", "境界-source.md"))
def test_explicit_literal_filename_does_not_read_existing_file(
    name, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / name
    source.write_bytes(SAMPLE.read_bytes())
    before = source.read_bytes()
    assert validate_gnn_source(name)["is_valid"] is True
    explicit = validate_gnn_source(name, is_content=True)
    assert explicit["is_valid"] is False and explicit["errors"]
    valid, errors = validate_gnn_syntax(name, is_content=True)
    assert explicit == {"is_valid": valid, "errors": errors}
    assert source.read_bytes() == before


def test_path_objects_can_be_forced_to_literal_text(tmp_path):
    source = tmp_path / "valid.md"
    source.write_bytes(SAMPLE.read_bytes())
    assert validate_gnn_source(source)["is_valid"] is True
    explicit = validate_gnn_source(source, is_content=True)
    assert explicit["is_valid"] is False
    valid, errors = validate_gnn_syntax(source, is_content=True)
    assert explicit == {"is_valid": valid, "errors": errors}


def test_file_contents_cannot_redirect_to_another_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "target.md"
    target.write_bytes(SAMPLE.read_bytes())
    indirection = tmp_path / "indirection.md"
    indirection.write_text(target.name)
    assert validate_gnn_source(target)["is_valid"] is True
    result = validate_gnn_source(indirection)
    assert result["is_valid"] is False and result["errors"]
    assert indirection.read_text() == target.name


def test_directory_and_undecodable_file_report_invalidity(tmp_path):
    assert validate_gnn_source(tmp_path)["is_valid"] is False
    source = tmp_path / "invalid-utf8.md"
    source.write_bytes(b"\xff\xfe")
    assert validate_gnn_source(source)["is_valid"] is False
    assert source.read_bytes() == b"\xff\xfe"
