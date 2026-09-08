"""Pins for ``analysis/pymdp/visualizer`` (previously 10% coverage)."""

from __future__ import annotations

from pathlib import Path

from gnn.analysis.pymdp.visualizer import (
    PyMDPVisualizer,
    create_visualizer,
    save_all_visualizations,
)


def test_constructor_normalizes_positional_output_dir(tmp_path: Path) -> None:
    visualizer = PyMDPVisualizer(tmp_path)

    assert visualizer.save_dir == Path(tmp_path)
    assert visualizer.show_plots is False or visualizer.show_plots is True


def test_constructor_prefers_explicit_save_dir(tmp_path: Path) -> None:
    visualizer = PyMDPVisualizer(save_dir=tmp_path / "custom", show_plots=False)

    assert visualizer.save_dir == Path(tmp_path / "custom")


def test_factory_applies_config(tmp_path: Path) -> None:
    visualizer = create_visualizer(
        {"grid_size": 4, "save_dir": str(tmp_path), "figsize": (6, 6)}
    )

    assert visualizer.grid_size == 4
    assert visualizer.save_dir == Path(str(tmp_path))
    assert visualizer.figsize == (6, 6)


def test_save_all_visualizations_renders_synthetic_results(tmp_path: Path) -> None:
    results = {
        "states": [0, 1, 2, 1, 0],
        "num_states": 3,
        "beliefs": [[0.8, 0.2], [0.3, 0.7]],
        "observations": [0, 1, 1],
        "metrics": {
            "vfe_history": [1.2, 0.9, 0.7],
            "efe_history": [2.0, 1.8, 1.5],
        },
    }

    saved = save_all_visualizations(results, tmp_path)

    assert saved, "expected rendered visualizations for a full results payload"
    for path in saved.values():
        assert path.exists()
