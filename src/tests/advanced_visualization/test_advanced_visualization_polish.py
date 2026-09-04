"""Polish-pass seams: d2 parsed_json_dir, network-graph indices, extractor DI, theme parity.

Complements the composability/public-api files; each test defends an
observable contract of the polish pass.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestProcessGnnFileWithD2Param:
    def test_parsed_json_dir_is_optional_keyword(self) -> None:
        """The lookup path is parameterized; None falls back to the
        historical cwd-relative ``output/3_gnn_output`` lookup."""
        import inspect

        from advanced_visualization.d2_visualizer import process_gnn_file_with_d2

        sig = inspect.signature(process_gnn_file_with_d2)
        assert "parsed_json_dir" in sig.parameters
        assert sig.parameters["parsed_json_dir"].default is None

    def test_parsed_json_dir_supplies_step3_artifacts(self, tmp_path: Any) -> None:
        """A parsed_json_dir containing real Step 3 JSON is consumed without
        touching the cwd fallback; results stay well-formed either way."""
        import json as jsonlib

        from advanced_visualization.d2_visualizer import process_gnn_file_with_d2

        model_dir = tmp_path / "parsed" / "t"
        model_dir.mkdir(parents=True)
        model_data = {"variables": [{"name": "a"}, {"name": "b"}], "connections": []}
        (model_dir / "t_parsed.json").write_text(jsonlib.dumps(model_data))
        gnn_file = tmp_path / "t.md"
        gnn_file.write_text("# t\n\na -> b\n")

        results = process_gnn_file_with_d2(
            gnn_file, tmp_path / "out", parsed_json_dir=tmp_path / "parsed"
        )
        assert isinstance(results, list)
        assert all(isinstance(r.success, bool) for r in results)


class TestNetworkGraphIndices:
    def test_draws_only_resolvable_pairs(self, tmp_path: Any) -> None:
        """Edges resolve via the name→index map; unresolvable names are
        silently skipped and the artifact still renders."""
        from advanced_visualization.visualizer import AdvancedVisualizer

        payload = {
            "success": True,
            "blocks": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
            "connections": [
                {"source_variables": ["a"], "target_variables": ["b"]},
                {"source_variables": ["a"], "target_variables": ["ghost"]},
            ],
            "parameters": [],
        }

        class Stub:
            def extract_from_content(self, content: str) -> dict[str, Any]:
                return payload

        AdvancedVisualizer(extractor=Stub()).generate_visualizations(
            content="x", model_name="idx", output_dir=tmp_path
        )
        assert (tmp_path / "idx" / "idx_network.png").exists()

    def test_legacy_scalar_connections_resolve(self, tmp_path: Any) -> None:
        """Legacy ``source``/``target`` scalar dicts are accepted as
        singletons (same contract as _shared.normalize_connection_format)."""
        from advanced_visualization.visualizer import AdvancedVisualizer

        payload = {
            "success": True,
            "blocks": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
            "connections": [{"source": "a", "target": "c"}],
            "parameters": [],
        }

        class Stub:
            def extract_from_content(self, content: str) -> dict[str, Any]:
                return payload

        AdvancedVisualizer(extractor=Stub()).generate_visualizations(
            content="x", model_name="leg", output_dir=tmp_path
        )
        assert (tmp_path / "leg" / "leg_network.png").exists()


class TestExtractorInjection:
    def test_stub_extractor_is_used(self, tmp_path: Any) -> None:
        """The constructor seam accepts any extract_from_content provider;
        no module-global monkeypatching required, and a failing extraction
        routes to the recovery path."""
        from advanced_visualization.visualizer import AdvancedVisualizer

        calls: list[str] = []

        class Stub:
            def extract_from_content(self, content: str) -> dict[str, Any]:
                calls.append(content)
                return {"success": False, "errors": ["stub"]}

        viz = AdvancedVisualizer(extractor=Stub())
        viz.generate_visualizations(
            content="probe-content", model_name="di", output_dir=tmp_path
        )
        assert calls == ["probe-content"]  # stub consumed, real extractor never built
        assert (tmp_path / "di" / "di_fallback_summary.html").exists()

    def test_default_construction_still_builds_extractor(self) -> None:
        from advanced_visualization.visualizer import AdvancedVisualizer

        assert AdvancedVisualizer()._get_extractor() is not None


class TestThemeParity:
    def test_shared_constants_exist_and_are_pure_data(self) -> None:
        from advanced_visualization import _theme

        for name in _theme.__all__:
            value = getattr(_theme, name)
            assert isinstance(value, str) and value

    def test_dashboard_renders_shared_rules_with_single_braces(
        self, tmp_path: Any
    ) -> None:
        """Theme-refactor parity guard: the dashboard HTML renders the shared
        rules exactly once, with single braces (no f-string escape
        regressions) and identical reset/font/gradient values."""
        from advanced_visualization._theme import BASE_CSS, BODY_GRADIENT, FONT_STACK
        from advanced_visualization.dashboard import generate_dashboard

        content = """# Theme Parity Model

## StateSpaceBlock
hidden_states[10, type=float]
observations[5, type=float]

## Connections
hidden_states -> observations

## Parameters
learning_rate = 0.01
"""
        result = generate_dashboard(content, "theme_parity", tmp_path)
        assert result is not None
        html = result.read_text()

        assert BASE_CSS.replace("{{", "{") in html
        assert "{{" not in html
        assert FONT_STACK in html
        assert BODY_GRADIENT in html
        assert "function showTab(tabName) {" in html
        assert "Generated on 20" in html
