"""Tests for the RxInfer GIF dashboard generator (roadmap A5).

Generates the dashboard against a tmp directory of fixture manifest.json files
plus tiny real GIF files (written via PIL), then asserts the HTML carries the
category filter, the bucketed state-space-size filter, the compare-mode
controls, and every model card. Deterministic; no skips.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from analysis.rxinfer.dashboard import (
    SIZE_BUCKET_LABELS,
    UNKNOWN_SIZE_BUCKET,
    _size_bucket,
    generate_dashboard,
)

# (model stem, num_states, expected size bucket, expected category)
FIXTURE_MODELS: tuple[tuple[str, int, str, str], ...] = (
    ("simple_mdp", 3, SIZE_BUCKET_LABELS[0], "Discrete"),
    ("hierarchical_agent", 8, SIZE_BUCKET_LABELS[1], "Hierarchical"),
    ("multiagent_swarm", 64, SIZE_BUCKET_LABELS[2], "Multi-Agent"),
    ("continuous_navigation", 256, SIZE_BUCKET_LABELS[3], "Continuous"),
)


def _write_gif(path: Path) -> None:
    """Write a tiny real GIF (valid GIF87a/89a bytes) via PIL."""
    Image.new("P", (4, 4), color=0).save(path, format="GIF")


def _build_gif_dir(tmp_path: Path) -> Path:
    """Populate a tmp dir with fixture GIFs and manifest sidecars."""
    gif_dir = tmp_path / "gifs"
    gif_dir.mkdir()
    for stem, num_states, _bucket, _category in FIXTURE_MODELS:
        gif = gif_dir / f"{stem}_100steps.gif"
        _write_gif(gif)
        manifest = {
            "num_states": num_states,
            "timesteps": 100,
            "belief_accuracy": 0.9,
            "inference_converged": True,
            "model_kind": "flat",
            "seed": 42,
            "generator": "gif_animator.py",
        }
        gif.with_suffix(".manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return gif_dir


def _generate_html(tmp_path: Path) -> str:
    gif_dir = _build_gif_dir(tmp_path)
    output = tmp_path / "dashboard.html"
    path = generate_dashboard(gif_dir, output)
    assert path == str(output)
    assert output.exists()
    return output.read_text(encoding="utf-8")


def test_fixture_gifs_are_real_gif_bytes(tmp_path: Path) -> None:
    """The minimal GIFs written for the dashboard are genuine GIF files."""
    gif_dir = _build_gif_dir(tmp_path)
    gifs = sorted(gif_dir.glob("*_100steps.gif"))
    assert len(gifs) == len(FIXTURE_MODELS)
    for gif in gifs:
        header = gif.read_bytes()[:6]
        assert header in (b"GIF87a", b"GIF89a"), f"{gif.name}: {header!r}"


def test_size_bucket_boundaries() -> None:
    """_size_bucket buckets exactly at <=4 / 5-16 / 17-64 / 65+."""
    assert _size_bucket(1) == SIZE_BUCKET_LABELS[0]
    assert _size_bucket(4) == SIZE_BUCKET_LABELS[0]
    assert _size_bucket(5) == SIZE_BUCKET_LABELS[1]
    assert _size_bucket(16) == SIZE_BUCKET_LABELS[1]
    assert _size_bucket(17) == SIZE_BUCKET_LABELS[2]
    assert _size_bucket(64) == SIZE_BUCKET_LABELS[2]
    assert _size_bucket(65) == SIZE_BUCKET_LABELS[3]
    assert _size_bucket(729) == SIZE_BUCKET_LABELS[3]


def test_size_bucket_rejects_unusable_values() -> None:
    """Missing / malformed num_states lands in the unknown bucket."""
    bad_values: tuple[object, ...] = (None, "3", -1, 0, True, [3], {})
    for bad in bad_values:
        assert _size_bucket(bad) == UNKNOWN_SIZE_BUCKET


def test_dashboard_contains_category_filter(tmp_path: Path) -> None:
    """The category filter select and every present category option exist."""
    html = _generate_html(tmp_path)
    assert 'select id="filter"' in html
    for _stem, _num_states, _bucket, category in FIXTURE_MODELS:
        assert f'<option value="{category}">{category}</option>' in html


def test_dashboard_contains_size_filter_with_correct_bucketing(
    tmp_path: Path,
) -> None:
    """The size filter select exists, offers all present buckets, and each
    card carries the bucket derived from its manifest num_states."""
    html = _generate_html(tmp_path)
    assert 'select id="size-filter"' in html
    for stem, _num_states, bucket, _category in FIXTURE_MODELS:
        assert f'<option value="{bucket}">{bucket}</option>' in html
        assert f'data-model="{stem}" data-size-bucket="{bucket}"' in html
    # No unknown bucket is offered when every manifest carries num_states.
    assert UNKNOWN_SIZE_BUCKET not in html


def test_dashboard_contains_compare_mode_controls(tmp_path: Path) -> None:
    """Two compare dropdowns + the compare panel exist, listing every model."""
    html = _generate_html(tmp_path)
    assert 'select id="compare-a"' in html
    assert 'select id="compare-b"' in html
    assert 'id="compare-panel"' in html
    for stem, _num_states, _bucket, _category in FIXTURE_MODELS:
        # Each model is selectable in both compare dropdowns.
        assert html.count(f'<option value="{stem}">{stem}</option>') == 2
    # Manifest stats are embedded for the compare panels.
    assert "const MODELS =" in html
    assert '"num_states": 256' in html
    assert '"belief_accuracy": 0.9' in html


def test_dashboard_contains_every_model_card(tmp_path: Path) -> None:
    """Each model gets a card with its GIF, title, and manifest meta line."""
    html = _generate_html(tmp_path)
    for stem, num_states, _bucket, category in FIXTURE_MODELS:
        assert f'<div class="card-title">{stem}</div>' in html
        assert f'src="{stem}_100steps.gif"' in html
        assert f'data-category="{category}"' in html
        assert f"{num_states} states, 100 steps, acc=0.9" in html


def test_dashboard_is_self_contained_vanilla_js(tmp_path: Path) -> None:
    """No external assets: no http(s) URLs, and the filter/compare JS is inline."""
    html = _generate_html(tmp_path)
    assert "http://" not in html
    assert "https://" not in html
    assert "function applyFilter()" in html
    assert "function updateCompare()" in html


def test_dashboard_keeps_neutral_dark_house_style(tmp_path: Path) -> None:
    """Neutral dark-gray/black background with visible focus states."""
    html = _generate_html(tmp_path)
    assert "background: #121212" in html
    assert "focus-visible" in html


def test_missing_manifest_lands_in_unknown_bucket(tmp_path: Path) -> None:
    """A GIF without a manifest sidecar still gets a card, bucketed unknown."""
    gif_dir = tmp_path / "gifs"
    gif_dir.mkdir()
    _write_gif(gif_dir / "orphan_model_100steps.gif")
    output = tmp_path / "dashboard.html"
    path = generate_dashboard(gif_dir, output)
    html = Path(path).read_text(encoding="utf-8")
    assert f'data-model="orphan_model" data-size-bucket="{UNKNOWN_SIZE_BUCKET}"' in html
    assert (
        f'<option value="{UNKNOWN_SIZE_BUCKET}">{UNKNOWN_SIZE_BUCKET}</option>' in html
    )


def test_empty_directory_returns_empty_string(tmp_path: Path) -> None:
    """No GIFs means no dashboard file and an empty-string return."""
    gif_dir = tmp_path / "empty"
    gif_dir.mkdir()
    output = tmp_path / "dashboard.html"
    result: Any = generate_dashboard(gif_dir, output)
    assert result == ""
    assert not output.exists()
