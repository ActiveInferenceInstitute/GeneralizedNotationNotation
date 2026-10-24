"""Portable interactive bundles preserve dependencies and producer collisions."""

import json
import os
from pathlib import Path

from gnn.website.collection import _collect_visualizations


def test_interactive_sidecars_keep_relative_layout_without_collision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    roots = [
        tmp_path / "8_visualization_output/visualization_results",
        tmp_path / "9_advanced_viz_output",
    ]
    for index, root in enumerate(roots):
        (root / "models").mkdir(parents=True)
        (root / "data").mkdir()
        (root / "models/view.html").write_text(
            '<script src="../data/scene.js"></script>'
        )
        (root / "data/scene.js").write_text(f"const producer = {index};")
    assets = tmp_path / "20_website_output/assets"
    assets.mkdir(parents=True)
    entries, warnings = _collect_visualizations(roots, assets)
    assert not warnings
    assert len(entries) == 2
    assert entries[0]["path"] != entries[1]["path"]
    for index, entry in enumerate(entries):
        html = assets / entry["path"]
        assert html.read_text() == '<script src="../data/scene.js"></script>'
        assert (
            html.parent / "../data/scene.js"
        ).read_text() == f"const producer = {index};"

    from gnn.pipeline.run_context import build_run_context
    from gnn.website.collection import collect_website_data

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    context = build_run_context(input_dir, tmp_path, "portable-site", [20], {})
    context_file = context.write(tmp_path / "context.json")
    monkeypatch.setenv("GNN_RUN_CONTEXT_FILE", str(context_file))
    monkeypatch.setenv("GNN_RUN_ID", context.run_id)
    summary = tmp_path / "00_pipeline_summary/current_summary.json"
    summary.parent.mkdir(exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "run_id": context.run_id,
                "steps": [
                    {
                        "artifacts": [
                            {"path": path.relative_to(tmp_path).as_posix()}
                            for root in roots
                            for path in root.rglob("*")
                            if path.is_file()
                        ]
                    }
                ],
            }
        )
    )
    # A custom site destination must not be mistaken for the producer root.
    relative_root = Path(os.path.relpath(tmp_path))
    custom_site = relative_root / "public/deep/site"
    custom_assets = custom_site / "assets"
    custom_assets.mkdir(parents=True)
    data = collect_website_data(
        relative_root, input_dir, custom_assets, output_dir=custom_site
    )
    assert not data["warnings"]
    assert len(data["visualizations"]) == 2
    for entry in data["visualizations"]:
        html = custom_assets / entry["path"]
        assert html.is_file()
        assert (html.parent / "../data/scene.js").is_file()
