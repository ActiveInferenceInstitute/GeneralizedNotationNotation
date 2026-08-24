#!/usr/bin/env python3
"""Direct coverage for the documented GNN-aware export writers.

``src/export/format_exporters.py`` is the module the export documentation
(user-facing README and doc/gnn/integration/gnn_export.md) names as the GNN
export surface, yet its format writers had zero direct coverage — only the
higher-level ``formatters``/``processor`` path was exercised. These tests pin
the documented public writers (``export_to_json_gnn``, ``export_to_xml_gnn``,
``export_to_python_pickle``, ``export_to_gexf``, ``export_to_graphml``,
``export_to_json_adjacency_list``, ``export_to_plaintext_summary``,
``export_to_plaintext_dsl``) against a real parsed GNN model, verifying they
write well-formed output and fail closed on invalid input.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPO_ROOT = SRC.parent
SAMPLE = REPO_ROOT / "input" / "gnn_files" / "basics" / "static_perception.md"

from export.core import export_gnn_files  # noqa: E402
from export.format_exporters import (  # noqa: E402
    export_to_gexf,
    export_to_graphml,
    export_to_json_adjacency_list,
    export_to_json_gnn,
    export_to_plaintext_dsl,
    export_to_plaintext_summary,
    export_to_python_pickle,
    export_to_xml_gnn,
)


def _sample_model() -> dict[str, Any]:
    """Return a real parsed GNN model as a dict with the fields the exporters use."""
    from gnn import parse_gnn_file

    spec = parse_gnn_file(SAMPLE)
    model = spec.to_dict() if hasattr(spec, "to_dict") else spec
    # The adjacency/graph exporters consume "statespaceblock" + "connections".
    if "statespaceblock" not in model:
        model["statespaceblock"] = [
            {"id": "s_agent1", "type": "state"},
            {"id": "o_agent1", "type": "observation"},
        ]
    if "connections" not in model:
        model["connections"] = [
            {"sources": ["s_agent1"], "targets": ["o_agent1"], "operator": "-"}
        ]
    return model


def test_export_to_json_gnn_writes_valid_json(tmp_path: Path) -> None:
    model = _sample_model()
    out = tmp_path / "model.json"
    ok, message = export_to_json_gnn(model, out)
    assert ok, message
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload.get("name") == model.get("name")


def test_export_to_xml_gnn_writes_xml(tmp_path: Path) -> None:
    model = _sample_model()
    out = tmp_path / "model.xml"
    ok, message = export_to_xml_gnn(model, out)
    assert ok, message
    text = out.read_text(encoding="utf-8")
    assert text.lstrip().startswith("<?xml") or "<" in text


def test_export_to_python_pickle_roundtrips(tmp_path: Path) -> None:
    import pickle

    model = _sample_model()
    out = tmp_path / "model.pkl"
    ok, message = export_to_python_pickle(model, out)
    assert ok, message
    with open(out, "rb") as f:
        loaded = pickle.load(f)
    assert loaded.get("name") == model.get("name")


def test_export_to_plaintext_summary_writes(tmp_path: Path) -> None:
    model = _sample_model()
    out = tmp_path / "summary.txt"
    ok, message = export_to_plaintext_summary(model, str(out))
    assert ok, message
    text = out.read_text(encoding="utf-8")
    assert "GNN Model Summary" in text


def test_export_to_plaintext_dsl_uses_raw_sections(tmp_path: Path) -> None:
    model = _sample_model()
    model["raw_sections"] = {"ModelName": "Demo Model", "Footer": "v1"}
    out = tmp_path / "model.gnn"
    ok, message = export_to_plaintext_dsl(model, str(out))
    assert ok, message
    text = out.read_text(encoding="utf-8")
    assert "## ModelName" in text
    assert "Demo Model" in text


def test_graph_exports_write_when_networkx_available(tmp_path: Path) -> None:
    # NetworkX is a hard project dependency (pyproject.toml), so the graph
    # exporters must work unconditionally (repo zero-skip contract).
    model = _sample_model()
    gexf = tmp_path / "model.gexf"
    ok, message = export_to_gexf(model, str(gexf))
    assert ok, message
    assert gexf.exists()

    graphml = tmp_path / "model.graphml"
    ok, message = export_to_graphml(model, str(graphml))
    assert ok, message
    assert graphml.exists()

    adj = tmp_path / "adjacency.json"
    ok, message = export_to_json_adjacency_list(model, str(adj))
    assert ok, message
    data = json.loads(adj.read_text(encoding="utf-8"))
    # networkx adjacency_data returns a dict with an "adjacency" key.
    assert isinstance(data, dict)
    assert "adjacency" in data


def test_graph_exports_fail_closed_without_networkx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the HAS_NETWORKX flag to False to pin the documented fail-closed path.
    import export.format_exporters as fe

    monkeypatch.setattr(fe, "HAS_NETWORKX", False)
    model = _sample_model()
    ok, message = export_to_gexf(model, str(tmp_path / "x.gexf"))
    assert ok is False
    assert "NetworkX not available" in message
    ok, message = export_to_graphml(model, str(tmp_path / "x.graphml"))
    assert ok is False
    ok, message = export_to_json_adjacency_list(model, str(tmp_path / "x.json"))
    assert ok is False



def test_export_gnn_files_writes_all_formats(tmp_path: Path) -> None:
    """The documented batch export orchestrator writes every format for a real
    GNN file into the per-step output boundary, returning True when all succeed.
    """
    import logging

    target = tmp_path / "input"
    target.mkdir()
    model_file = SAMPLE
    # copy the sample into the target dir
    dest = target / model_file.name
    dest.write_bytes(model_file.read_bytes())

    logger = logging.getLogger("test_export_core")
    ok = export_gnn_files(target, tmp_path / "output", logger)
    assert ok is True

    stem = dest.stem
    out_dir = list((tmp_path / "output").glob("*/" + stem))[0]
    written = {p.name for p in out_dir.iterdir()}
    assert f"{stem}.json" in written
    assert f"{stem}.xml" in written
    assert f"{stem}_summary.txt" in written
    assert f"{stem}_dsl.txt" in written
    # Graph formats are written only when NetworkX is available.
    from export.core import HAS_NETWORKX

    if HAS_NETWORKX:
        assert f"{stem}.graphml" in written
        assert f"{stem}.gexf" in written


def test_export_gnn_files_no_files_returns_true(tmp_path: Path) -> None:
    import logging

    empty = tmp_path / "empty"
    empty.mkdir()
    ok = export_gnn_files(empty, tmp_path / "output", logging.getLogger())
    assert ok is True


def test_export_gnn_files_fails_closed_without_exporters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import logging

    import export.core as core_mod

    monkeypatch.setattr(core_mod, "FORMAT_EXPORTERS_LOADED", False)
    target = tmp_path / "input"
    target.mkdir()
    (target / "model.md").write_text("## ModelName\nDemo\n", encoding="utf-8")
    ok = core_mod.export_gnn_files(target, tmp_path / "output", logging.getLogger())
    assert ok is False
