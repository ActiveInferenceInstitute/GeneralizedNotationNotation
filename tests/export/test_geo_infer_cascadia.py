"""Cascadia river homing: a real H3 spatial model through the strict exporter.

The fixtures under ``tests/export/cascadia/`` are byte copies of the
GEO-INFER-ACT Cascadia example (``GEO-INFER-ACT/examples/cascadia/data/`` in
the GEO_INFER checkout): the GNN v1 source GEO-INFER rendered from measured
NHDPlus HR lower Smith River geometry on H3 resolution 8, its state-ID order,
and the ``gnn-geo-infer/1`` artifact this exporter produced from them.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import h3
import pytest

from gnn.export.geo_infer import build_geo_infer_artifact
from gnn.export.notation_metadata import derive_geo_metadata
from gnn.extract.pomdp_extractor import extract_pomdp_from_content

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "cascadia"
SOURCE = FIXTURES / "smith_river_homing.gnn.md"
STATE_IDS = FIXTURES / "smith_river_homing.state_ids.json"
GOLDEN = FIXTURES / "smith_river_homing.geo-infer.json"
STEP_SECONDS = 21600


def _export(text: str | None = None) -> dict:
    return build_geo_infer_artifact(
        SOURCE.read_text(encoding="utf-8") if text is None else text,
        step_seconds=STEP_SECONDS,
        state_ids=json.loads(STATE_IDS.read_text(encoding="utf-8")),
        space_kind="h3",
    )


def test_export_reproduces_the_golden_artifact() -> None:
    artifact = _export()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert artifact == golden
    assert artifact["dimensions"] == {"states": 26, "observations": 12, "actions": 4}
    assert (
        artifact["provenance"]["source_sha256"]
        == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    )


def test_cli_h3_export_matches_the_golden_artifact(tmp_path: Path) -> None:
    output = tmp_path / "cascadia.geo-infer.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gnn.export.geo_infer",
            str(SOURCE),
            str(output),
            "--step-seconds",
            str(STEP_SECONDS),
            "--space-kind",
            "h3",
            "--state-ids",
            str(STATE_IDS),
        ],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        check=True,
        timeout=120,
    )
    assert output.read_bytes() == GOLDEN.read_bytes()


def test_b_axis_order_is_detected_from_the_data() -> None:
    model = extract_pomdp_from_content(
        SOURCE.read_text(encoding="utf-8"),
        strict_validation=True,
        on_error="raise",
        insert_default_c=False,
    )
    origin = model.matrix_provenance["B"]
    assert origin["detected_order"] == ["next_state", "previous_state", "action"]
    assert origin["contradiction"] is False
    assert origin["derived"] is False


def test_transposed_transition_literal_is_rejected() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["matrices"]["B"]
    n = len(golden)
    swapped = [[golden[prev][nxt] for prev in range(n)] for nxt in range(n)]
    planes = ",\n".join(
        "  (\n"
        + ",\n".join("    (" + ", ".join(repr(v) for v in cell) + ")" for cell in plane)
        + "\n  )"
        for plane in swapped
    )
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("B={")
    end = text.index("\n}\n", start) + 3
    mutated = text[:start] + "B={\n" + planes + "\n}\n" + text[end:]
    with pytest.raises(ValueError):
        _export(mutated)


def test_every_probability_literal_is_an_exact_small_dyadic() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    for name in ("A", "B", "D", "E"):
        start = text.index(f"\n{name}={{") + 1
        block = text[start : text.index("}", start)]
        for literal in re.findall(r"\d+\.\d+", block):
            denominator = Fraction(Decimal(literal)).denominator
            assert denominator & (denominator - 1) == 0 and denominator <= 64, (
                name,
                literal,
            )


def test_transitions_only_stay_or_move_to_h3_neighbors() -> None:
    artifact = _export()
    ids = artifact["space"]["state_ids"]
    B = artifact["matrices"]["B"]
    for nxt, plane in enumerate(B):
        for prev, cell in enumerate(plane):
            if any(value > 0 for value in cell) and nxt != prev:
                assert h3.are_neighbor_cells(ids[prev], ids[nxt])
    assert {h3.get_resolution(cell) for cell in ids} == {8}


def test_notation_declares_its_own_time_step_and_space_kind() -> None:
    derived = derive_geo_metadata(SOURCE.read_text(encoding="utf-8"))
    assert derived["options"] == {
        "step_seconds": float(STEP_SECONDS),
        "space_kind": "h3",
    }
    record = derived["metadata_derivation"]
    assert record["step_seconds"]["derived"] is True
    assert record["space_kind"]["derived"] is True
    assert record["time_index"]["variable"] == "t"


def test_noncanonical_h3_identifier_is_rejected() -> None:
    ids = json.loads(STATE_IDS.read_text(encoding="utf-8"))
    ids[0] = ids[0].upper()
    with pytest.raises(ValueError, match="canonical"):
        build_geo_infer_artifact(
            SOURCE.read_text(encoding="utf-8"),
            step_seconds=STEP_SECONDS,
            state_ids=ids,
            space_kind="h3",
        )
