"""Real producer reproduction and corruption controls for the worked example."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).with_name("qualify_export.py")
FIXTURES = ROOT / "tests/export/cascadia"


def invoke(folder: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(folder), "--step-seconds", "21600", *flags],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_reference_export_and_create_only_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    result = invoke(FIXTURES, "--output", str(receipt))
    assert result.returncode == 0, result.stderr
    record = json.loads(receipt.read_bytes())
    assert record["export_reproduced"] and record["native_claim_ready"] is False
    owners = dict(record["producer_source_sha256"])
    assert "src/gnn/export/geo_infer.py" in owners
    assert invoke(FIXTURES, "--check-receipt", str(receipt)).returncode == 0
    before = receipt.read_bytes()
    assert invoke(FIXTURES, "--output", str(receipt)).returncode != 0
    assert receipt.read_bytes() == before


@pytest.mark.parametrize(
    "mutation", ["habit", "source", "order", "clock", "duplicate", "boolean"]
)
def test_reproduction_rejects_corrupted_inputs(tmp_path: Path, mutation: str) -> None:
    shutil.copytree(FIXTURES, tmp_path / "condition")
    folder = tmp_path / "condition"
    artifact = folder / "smith_river_homing.geo-infer.json"
    payload = json.loads(artifact.read_bytes())
    if mutation == "habit":
        payload["matrices"]["E"] = [0.25] * 4
    elif mutation == "source":
        payload["provenance"]["source_sha256"] = "0" * 64
    elif mutation == "order":
        payload["space"]["state_ids"].reverse()
    elif mutation == "clock":
        payload["time"]["step_seconds"] = 1
    elif mutation == "boolean":
        payload["matrices"]["B"][0][0][1] = True
    else:
        text = artifact.read_text(encoding="utf-8")
        artifact.write_text(
            text.replace("{", '{"schema_version":"ignored",', 1), encoding="utf-8"
        )
        result = invoke(folder)
        assert result.returncode != 0 and "duplicate JSON key" in result.stderr
        return
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    result = invoke(folder)
    assert result.returncode != 0 and "differs" in result.stderr


def test_receipt_recheck_rejects_changed_producer_digest(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    created = invoke(FIXTURES, "--output", str(receipt))
    assert created.returncode == 0, created.stderr
    data = json.loads(receipt.read_bytes())
    entry = next(
        row
        for row in data["producer_source_sha256"]
        if row[0] == "src/gnn/export/geo_infer.py"
    )
    entry[1] = "0" * 64
    receipt.write_text(json.dumps(data), encoding="utf-8")
    before = receipt.read_bytes()
    rejected = invoke(FIXTURES, "--check-receipt", str(receipt))
    assert rejected.returncode != 0 and "receipt differs" in rejected.stderr
    assert receipt.read_bytes() == before
