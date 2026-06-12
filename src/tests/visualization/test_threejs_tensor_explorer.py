from __future__ import annotations

import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

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
    playwright_api = pytest.importorskip("playwright.sync_api")
    tensor = np.zeros((2, 2, 2), dtype=float)
    tensor[:, :, 0] = [[0.9, 0.1], [0.2, 0.8]]
    output = tmp_path / "tensor.html"
    assert MatrixVisualizer().generate_threejs_tensor_explorer("B", tensor, output)
    server = _serve_directory(tmp_path)
    try:
        with playwright_api.sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page()
                console_errors: list[str] = []
                page.on(
                    "console",
                    lambda msg: (
                        console_errors.append(msg.text)
                        if msg.type == "error" and "favicon" not in msg.text.lower()
                        else None
                    ),
                )
                page.goto(f"http://127.0.0.1:{server.server_port}/tensor.html")
                page.wait_for_selector("canvas#scene")
                page.wait_for_function(
                    """
                    () => {
                      const canvas = document.querySelector('canvas#scene');
                      return canvas && canvas.width > 0 && canvas.height > 0;
                    }
                    """
                )
                non_background_pixels = page.evaluate(
                    """
                    () => {
                      const canvas = document.querySelector('canvas#scene');
                      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                      const pixels = new Uint8Array(canvas.width * canvas.height * 4);
                      gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
                      let count = 0;
                      for (let i = 0; i < pixels.length; i += 4) {
                        if (!(pixels[i] === 16 && pixels[i + 1] === 24 && pixels[i + 2] === 32)) {
                          count += 1;
                        }
                      }
                      return count;
                    }
                    """
                )
                assert non_background_pixels > 0
                assert console_errors == []
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


def test_threejs_tensor_explorer_reports_fallback_when_cdn_blocked(
    tmp_path: Path,
) -> None:
    playwright_api = pytest.importorskip("playwright.sync_api")
    tensor = np.zeros((2, 2, 2), dtype=float)
    tensor[:, :, 0] = [[0.9, 0.1], [0.2, 0.8]]
    output = tmp_path / "tensor.html"
    assert MatrixVisualizer().generate_threejs_tensor_explorer("B", tensor, output)
    server = _serve_directory(tmp_path)
    try:
        with playwright_api.sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page()
                page.route("**/three.module.js", lambda route: route.abort())
                page.goto(f"http://127.0.0.1:{server.server_port}/tensor.html")
                page.wait_for_function(
                    "() => document.body.dataset.threejsStatus === 'fallback'"
                )
                assert page.locator("#fallback").is_visible()
                assert "JSON fallback" in page.locator("#hud").inner_text()
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


def _serve_directory(path: Path) -> ThreadingHTTPServer:
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
