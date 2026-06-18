from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visualization.matrix import MatrixVisualizer


def test_threejs_tensor_explorer_writes_html_and_json(tmp_path: Path) -> None:
    tensor = np.zeros((2, 2, 2), dtype=float)
    tensor[:, :, 0] = [[0.9, 0.1], [0.2, 0.8]]
    output = tmp_path / "tensor.html"
    visualizer = MatrixVisualizer()
    assert visualizer.generate_threejs_tensor_explorer("B", tensor, output) is True
    html = output.read_text(encoding="utf-8")
    fallback = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert "three@" in html
    assert "<canvas" in html
    assert 'data-threejs-status="loading"' in html
    assert "JSON fallback data" in html
    assert fallback["shape"] == [2, 2, 2]


def test_threejs_tensor_explorer_browser_canvas_smoke(tmp_path: Path) -> None:
    tensor = np.zeros((2, 2, 2), dtype=float)
    tensor[:, :, 0] = [[0.9, 0.1], [0.2, 0.8]]
    output = tmp_path / "tensor.html"
    assert MatrixVisualizer().generate_threejs_tensor_explorer("B", tensor, output)
    html = output.read_text(encoding="utf-8")
    assert 'canvas id="scene"' in html
    assert "WebGLRenderer" in html
    assert "function animate()" in html
    assert "document.body.dataset.threejsStatus = 'ready'" in html


def test_threejs_tensor_explorer_reports_fallback_when_cdn_blocked(
    tmp_path: Path,
) -> None:
    tensor = np.zeros((2, 2, 2), dtype=float)
    tensor[:, :, 0] = [[0.9, 0.1], [0.2, 0.8]]
    output = tmp_path / "tensor.html"
    assert MatrixVisualizer().generate_threejs_tensor_explorer("B", tensor, output)
    html = output.read_text(encoding="utf-8")
    fallback = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert "markFallback" in html
    assert "document.body.dataset.threejsStatus = 'fallback'" in html
    assert 'id="fallback"' in html
    assert fallback["name"] == "B"
