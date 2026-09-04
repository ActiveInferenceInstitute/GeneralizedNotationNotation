"""Package-root public API surface tests for visualization.

Pins that every documented entry point (module README "Public API" table)
is importable from the ``visualization`` package root, and that the
pipeline's injected-logger contract is honored.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPackageRootAPI:
    def test_readme_table_symbols_resolve(self) -> None:
        import visualization

        documented = [
            "process_visualization",
            "discover_visualization_files",
            "load_visualization_model",
            "parse_gnn_content",
            "GNNParser",
            "MatrixVisualizer",
            "generate_network_visualizations",
            "generate_variable_parameter_bipartite",
            "generate_combined_analysis",
            "generate_combined_visualizations",
            "GNNVisualizer",
            "generate_graph_visualization",
            "generate_matrix_visualization",
            "parse_matrix_data",
            "backend_status",
        ]
        for name in documented:
            assert hasattr(visualization, name), f"missing package-root export: {name}"

    def test_new_pure_helpers_exported(self) -> None:
        import visualization

        for name in (
            "sample_parsed_data",
            "collect_visualization_matrices",
            "compute_connection_statistics",
        ):
            assert callable(getattr(visualization, name)), name

    def test_all_exports_resolve(self) -> None:
        import visualization

        for name in visualization.__all__:
            assert hasattr(visualization, name), f"__all__ entry missing: {name}"


class TestLoggerInjection:
    def test_process_visualization_uses_injected_logger(self, tmp_path: Path) -> None:
        from visualization import process_visualization

        injected = logging.getLogger("viz-di-test")
        empty_target = tmp_path / "no-such-target"
        out_dir = tmp_path / "out"
        result = process_visualization(
            empty_target, out_dir, logger=injected, verbose=True
        )
        # No inputs -> warning-code 2 contract preserved.
        assert result == 2
        assert (out_dir / "visualization_summary.json").exists()
