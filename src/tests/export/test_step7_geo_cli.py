"""Drive the real numbered parser/export commands with explicit GEO metadata."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "content",
    [
        '{"a.md":{},"a.md":{}}',
        '{"a.md":{"step_seconds":1,"step_seconds":2}}',
    ],
)
def test_duplicate_physical_metadata_is_rejected(tmp_path: Path, content: str) -> None:
    from export.options import process_export_cli

    source = tmp_path / "options.json"
    source.write_text(content)
    with pytest.raises(ValueError, match="Duplicate GEO metadata key"):
        process_export_cli(tmp_path, tmp_path / "out", geo_infer_options_file=source)


@pytest.mark.parametrize("include_metadata", [True, False])
def test_numbered_step7_geo_export(tmp_path: Path, include_metadata: bool) -> None:
    source = tmp_path / "input"
    source.mkdir()
    model = source / "gaussian.md"
    model.write_bytes(Path(__file__).with_name("gaussian_rectangular.md").read_bytes())
    output = tmp_path / "output"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    common = ["--target-dir", str(source), "--output-dir", str(output)]
    parsed = subprocess.run(
        [sys.executable, str(ROOT / "src/3_gnn.py"), *common],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert parsed.returncode == 0, parsed.stdout + parsed.stderr
    command = [
        sys.executable,
        str(ROOT / "src/7_export.py"),
        *common,
        "--formats",
        "geo_infer",
    ]
    if include_metadata:
        metadata = tmp_path / "geo-options.json"
        metadata.write_text(
            json.dumps(
                {
                    "gaussian.md": {
                        "model_type": "linear_gaussian",
                        "step_seconds": 2,
                        "units": {
                            "states": ["m", "m/s", "K"],
                            "observations": ["m", "m/s"],
                            "controls": ["N"],
                        },
                    }
                }
            )
        )
        command.extend(["--geo-infer-options-file", str(metadata)])
    exported = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60
    )
    assert exported.returncode == (0 if include_metadata else 1), (
        exported.stdout + exported.stderr
    )
    receipt = json.loads((output / "7_export_output/export_results.json").read_text())
    assert receipt["summary"]["successful_exports"] == int(include_metadata)
    if include_metadata:
        item = receipt["files_exported"][0]["exports"]["geo_infer"]
        artifact = json.loads(Path(item["export_file"]).read_text())
        assert (
            artifact["provenance"]["source_sha256"]
            == hashlib.sha256(model.read_bytes()).hexdigest()
        )
        assert artifact["dimensions"] == {"states": 3, "observations": 2, "controls": 1}
