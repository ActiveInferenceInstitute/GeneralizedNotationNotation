#!/usr/bin/env python3
"""
Pre-flight functionality tests for GNN pipeline core capabilities.

These tests exercise core modules directly in isolated temp directories,
without assuming prior pipeline steps have produced global artifacts.
"""

import pytest
import json
import os
from pathlib import Path
from typing import Dict, Any


class TestPipelineFunctionality:
    """Pre-flight checks that run each core module in isolation."""

    def test_visualization_generates_images(self, isolated_temp_dir):
        from visualization.processor import process_visualization
        # Create a minimal markdown GNN file (processor expects .md)
        gnn_dir = isolated_temp_dir / "gnn"
        gnn_dir.mkdir(parents=True, exist_ok=True)
        md = gnn_dir / "sample.md"
        md.write_text(
            """
# Test Model

## Variables
s: state
o: observation

## Connections
s -> o

## Matrices
A = [0.5, 0.5; 0.3, 0.7]
            """.strip()
        )
        out_root = isolated_temp_dir / "viz_out"
        out_root.mkdir(parents=True, exist_ok=True)

        ok = process_visualization(gnn_dir, out_root, verbose=False)
        assert ok is True

        results_dir = out_root / "visualization_results"
        assert results_dir.exists()

        # ensure at least one image exists in any model subdir
        png_files = list(results_dir.glob("**/*.png"))
        assert len(png_files) > 0
        for png in png_files:
            assert png.stat().st_size > 100

    def test_gnn_processing_generates_results(self, sample_gnn_files, isolated_temp_dir):
        from gnn.core_processor import process_gnn_directory

        gnn_dir = list(sample_gnn_files.values())[0].parent
        out_root = isolated_temp_dir / "gnn_proc"
        out_root.mkdir(parents=True, exist_ok=True)

        result = process_gnn_directory(gnn_dir, output_dir=out_root, recursive=True)
        assert result["status"] in ("SUCCESS", "FAILED")

        results_file = out_root / "gnn_core_results.json"
        assert results_file.exists()
        data = json.loads(results_file.read_text())
        assert isinstance(data, dict)

    def test_multi_format_export_generates_files(self, isolated_temp_dir):
        from export.processor import generate_exports
        # Create a minimal markdown GNN file for export
        gnn_dir = isolated_temp_dir / "gnn"
        gnn_dir.mkdir(parents=True, exist_ok=True)
        (gnn_dir / "sample.md").write_text(
            """
# Test Model

## Variables
x: latent
y: observed

## Connections
x -> y
            """.strip()
        )
        out_root = isolated_temp_dir / "exports"
        out_root.mkdir(parents=True, exist_ok=True)

        ok = generate_exports(gnn_dir, out_root, verbose=False)
        assert ok is True

        exports_dir = out_root / "exports"
        assert exports_dir.exists()
        produced = list(exports_dir.glob("*.*"))
        # Expect at least JSON to be present
        assert any(p.suffix == ".json" for p in produced)

    def test_results_json_are_valid(self, sample_gnn_files, isolated_temp_dir):
        # Produce viz and export results on local temp files, then validate JSON summaries
        from visualization.processor import process_visualization
        from export.processor import generate_exports
        gnn_dir = isolated_temp_dir / "gnn"
        gnn_dir.mkdir(parents=True, exist_ok=True)
        (gnn_dir / "sample.md").write_text("# M\n\n## Variables\na: v\n\n## Connections\na -> a\n")
        out_root = isolated_temp_dir / "preflight"
        out_root.mkdir(parents=True, exist_ok=True)

        process_visualization(gnn_dir, out_root)
        generate_exports(gnn_dir, out_root)

        viz_summary = out_root / "visualization_results" / "visualization_summary.json"
        exp_summary = out_root / "exports" / "export_results.json"
        for fpath in [viz_summary, exp_summary]:
            assert fpath.exists(), f"Missing summary: {fpath}"
            data = json.loads(fpath.read_text())
            assert isinstance(data, dict)
            assert len(data) > 0

    def test_preflight_metrics(self, sample_gnn_files, isolated_temp_dir):
        # Sanity metrics based on generated artifacts
        from visualization.processor import process_visualization
        gnn_dir = list(sample_gnn_files.values())[0].parent
        out_root = isolated_temp_dir / "metrics"
        out_root.mkdir(parents=True, exist_ok=True)

        process_visualization(gnn_dir, out_root)
        png_count = len(list((out_root / "visualization_results").glob("**/*.png")))
        assert png_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 