#!/usr/bin/env python3
"""
Exporter determinism contract tests.

Contract: exporting the same parsed model twice must produce byte-identical
artifacts in every pipeline format (json, xml, graphml, gexf, pickle). The only
permitted cross-run difference is documented run metadata recorded by the
parser — the ``canonical_parse.parse_timestamp`` field, which
``export_single_gnn_file`` re-derives on every call and which the json and
pickle writers embed verbatim. Covered surfaces:

1. ``export_model`` over two distinct parsed-model fixtures: every artifact is
   byte-identical across two runs (each fixture is parsed once, so even the
   timestamp-bearing formats must round-trip identically).
2. ``export_single_gnn_file`` over one inline spec: xml/graphml/gexf are
   byte-identical across runs; json/pickle may differ only by
   ``canonical_parse.parse_timestamp``.
3. The receipt script end-to-end over a two-file corpus: exit 0, receipt JSON
   well-formed, ``result == 'deterministic'`` with at least one artifact
   compared.
4. NEGATIVE: the receipt's artifact comparator detects a tampered artifact
   (and reports a deleted artifact), and the result decision maps a mismatch
   to ``nondeterministic``.
"""

from __future__ import annotations

import importlib.util
import json
import pickle  # nosec B403
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.export import export_model  # noqa: E402
from gnn.export.processor import export_single_gnn_file, parse_gnn_content  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "run_exporter_determinism_receipt.py"
FORMATS = ["json", "xml", "graphml", "gexf", "pickle"]

# Minimal GNN markdown specs. Both must parse with typed variables and
# connections so the XML/graph exporters exercise real structure.
SPEC_A = """\
# GNN Example: Determinism Fixture A
# GNN Version: 1.0

## ModelName
Determinism Fixture A

## Variables
s: discrete
o: discrete

## StateSpaceBlock
A[2,2,type=float]   # observation likelihood

## Connections
s -> o
s -> s
"""

SPEC_B = """\
# GNN Example: Determinism Fixture B
# GNN Version: 1.0

## ModelName
Determinism Fixture B

## Variables
x: continuous
y: continuous
z: discrete

## StateSpaceBlock
B[3,3,type=float]   # transition support

## Connections
x -> y
y -> z
z -> x
"""


def _artifact_path(result: dict[str, Any], fmt: str) -> Path:
    """Resolve the on-disk artifact for ``fmt`` from an ``export_model`` result."""
    info = result["exports"][fmt]
    assert isinstance(info, dict), f"{fmt}: missing export record: {info}"
    assert info.get("success") is True, f"{fmt}: export did not succeed: {info}"
    file_path = info.get("file")
    assert isinstance(file_path, str) and file_path, f"{fmt}: missing file: {info}"
    return Path(file_path)


def _load_artifact(fmt: str, raw: bytes) -> Any:
    """Deserialize a json/pickle artifact for documented-metadata stripping."""
    if fmt == "json":
        return json.loads(raw.decode("utf-8"))
    return pickle.loads(raw)  # nosec B301


def _strip_documented_run_metadata(data: Any) -> Any:
    """Remove the single documented per-parse field from a loaded artifact."""
    assert isinstance(data, dict), "expected parsed-content dict artifact"
    canonical = data.get("canonical_parse")
    assert isinstance(canonical, dict), "missing canonical_parse section"
    assert "parse_timestamp" in canonical, "missing documented parse_timestamp"
    stripped = dict(data)
    stripped["canonical_parse"] = {
        k: v for k, v in canonical.items() if k != "parse_timestamp"
    }
    return stripped


def test_export_model_two_fixtures_byte_identical(tmp_path: Path) -> None:
    """export_model twice per fixture: every artifact byte-identical across runs."""
    for name, spec in (("a", SPEC_A), ("b", SPEC_B)):
        model = parse_gnn_content(spec)
        assert "sections" in model, f"{name}: fixture failed to parse: {model}"
        out_a = tmp_path / f"{name}_run_a"
        out_b = tmp_path / f"{name}_run_b"
        result_a = export_model(model, out_a, formats=FORMATS)
        result_b = export_model(model, out_b, formats=FORMATS)
        assert result_a["success"] is True, result_a
        assert result_b["success"] is True, result_b
        for fmt in FORMATS:
            file_a = _artifact_path(result_a, fmt)
            file_b = _artifact_path(result_b, fmt)
            assert file_a.is_file(), f"{name}/{fmt}: run-a artifact missing"
            assert file_b.is_file(), f"{name}/{fmt}: run-b artifact missing"
            assert file_a.read_bytes() == file_b.read_bytes(), (
                f"{name}/{fmt}: artifact bytes differ between runs"
            )


def test_export_single_gnn_file_only_documented_timestamp_differs(
    tmp_path: Path,
) -> None:
    """export_single_gnn_file twice: xml/graphml/gexf identical; json/pickle may
    differ only by canonical_parse.parse_timestamp."""
    spec_path = tmp_path / "spec_a.md"
    spec_path.write_text(SPEC_A, encoding="utf-8")
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    # export_single_gnn_file expects the exports directory to already exist.
    dir_a.mkdir()
    dir_b.mkdir()
    result_a = export_single_gnn_file(spec_path, dir_a)
    result_b = export_single_gnn_file(spec_path, dir_b)
    assert result_a["success"] is True, result_a
    assert result_b["success"] is True, result_b

    extensions = {
        "json": "json",
        "xml": "xml",
        "graphml": "graphml",
        "gexf": "gexf",
        "pickle": "pkl",
    }
    for fmt, ext in extensions.items():
        file_a = dir_a / f"spec_a.{ext}"
        file_b = dir_b / f"spec_a.{ext}"
        assert file_a.is_file(), f"{fmt}: run-a artifact missing"
        assert file_b.is_file(), f"{fmt}: run-b artifact missing"
        bytes_a = file_a.read_bytes()
        bytes_b = file_b.read_bytes()
        if fmt in ("json", "pickle"):
            stripped_a = _strip_documented_run_metadata(_load_artifact(fmt, bytes_a))
            stripped_b = _strip_documented_run_metadata(_load_artifact(fmt, bytes_b))
            assert stripped_a == stripped_b, (
                f"{fmt}: difference beyond canonical_parse.parse_timestamp"
            )
        else:
            assert bytes_a == bytes_b, f"{fmt}: artifact bytes differ between runs"


def test_receipt_script_end_to_end_deterministic(tmp_path: Path) -> None:
    """The receipt script runs over a small corpus and reports deterministic."""
    target = tmp_path / "target"
    target.mkdir()
    (target / "spec_a.md").write_text(SPEC_A, encoding="utf-8")
    (target / "spec_b.md").write_text(SPEC_B, encoding="utf-8")
    receipts_dir = tmp_path / "receipts"

    proc = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--target-dir",
            str(target),
            "--receipts-dir",
            str(receipts_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"

    receipt_path = receipts_dir / "exporter_determinism_receipt.json"
    assert receipt_path.is_file(), "receipt JSON not written"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["receipt_version"] == "gnn_exporter_determinism_receipt_v1"
    assert receipt["result"] == "deterministic"
    assert receipt["mismatches"] == []
    assert receipt["artifacts_compared"] > 0
    assert len(receipt["corpus"]) == 2
    assert all(entry["parse_ok"] for entry in receipt["corpus"])


def _load_receipt_module() -> Any:
    """Import the receipt script as a module without going through sys.path."""
    spec = importlib.util.spec_from_file_location(
        "run_exporter_determinism_receipt", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None, SCRIPT_PATH
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_negative_tampered_artifact_is_nondeterministic(tmp_path: Path) -> None:
    """NEGATIVE: the comparator fires on a tampered artifact and a deleted
    artifact, and the decision maps mismatches to nondeterministic."""
    module = _load_receipt_module()
    model = parse_gnn_content(SPEC_A)
    assert "sections" in model
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    result_a = export_model(model, dir_a, formats=FORMATS)
    result_b = export_model(model, dir_b, formats=FORMATS)
    assert result_a["success"] is True
    assert result_b["success"] is True

    assert module.compare_artifacts(dir_a, dir_b) == [], "clean baseline flagged"

    target_file = dir_a / "model.json"
    target_file.write_bytes(target_file.read_bytes() + b"\n<!-- tampered -->")
    diffs = module.compare_artifacts(dir_a, dir_b)
    assert len(diffs) == 1, diffs
    assert diffs[0]["path"] == "model.json"
    assert diffs[0]["run_a_sha256"] != diffs[0]["run_b_sha256"]
    assert (
        module.decide_result(artifacts_compared=1, mismatches=diffs, parseable_files=1)
        == "nondeterministic"
    )

    (dir_b / "model.xml").unlink()
    diffs_missing = module.compare_artifacts(dir_a, dir_b)
    assert any(
        d["path"] == "model.xml" and d["run_b_sha256"] is None for d in diffs_missing
    ), diffs_missing

    # A clean comparison still maps to deterministic (decision sanity).
    assert (
        module.decide_result(artifacts_compared=1, mismatches=[], parseable_files=1)
        == "deterministic"
    )


def test_negative_vacuous_zero_artifact_run_fails_closed(tmp_path: Path) -> None:
    """NEGATIVE: a parseable corpus that exports zero comparable artifacts must
    not pass as deterministic. An unsupported format list exercises this
    end-to-end (exit 1, result 'error', explanatory note in the receipt)."""
    module = _load_receipt_module()
    assert (
        module.decide_result(artifacts_compared=0, mismatches=[], parseable_files=1)
        == "error"
    ), "vacuous run must not map to deterministic"

    target = tmp_path / "target"
    target.mkdir()
    (target / "spec_a.md").write_text(SPEC_A, encoding="utf-8")
    receipts_dir = tmp_path / "receipts"
    proc = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--target-dir",
            str(target),
            "--receipts-dir",
            str(receipts_dir),
            "--formats",
            "not_a_real_format",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 1, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    receipt = json.loads(
        (receipts_dir / "exporter_determinism_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["result"] == "error"
    assert receipt["artifacts_compared"] == 0
    assert "no comparable artifacts" in receipt["error"]
