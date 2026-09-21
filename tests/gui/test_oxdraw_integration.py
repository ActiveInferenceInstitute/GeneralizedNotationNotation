"""
Integration tests for oxdraw module

Tests end-to-end workflows:
- GNN file discovery and conversion
- Mermaid generation with metadata
- oxdraw integration (headless mode)
- Round-trip conversion validation
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from gnn.gui.oxdraw.mermaid_converter import (
    convert_gnn_file_to_mermaid,
    generate_mermaid_metadata,
    gnn_to_mermaid,
)
from gnn.gui.oxdraw.mermaid_parser import (
    convert_mermaid_file_to_gnn,
    extract_gnn_metadata,
    mermaid_to_gnn,
)
from gnn.gui.oxdraw.processor import (
    check_oxdraw_installed,
    get_module_info,
    process_oxdraw,
)

# Sample GNN content for testing
SAMPLE_GNN_CONTENT = """# GNN Example: Simple Active Inference Agent
# GNN Version: 1.0

## ModelName
Simple Active Inference Agent

## StateSpaceBlock
A[3,3,type=float]   # Likelihood matrix
B[3,3,3,type=float] # Transition matrix
C[3,type=float]     # Preference vector
D[3,type=float]     # Prior vector
s[3,1,type=float]   # Hidden state
o[3,1,type=int]     # Observation
u[1,type=int]       # Action

## Connections
D>s
s-A
A-o
s-B
u>B

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
s=HiddenState
o=Observation
u=Action

## ModelParameters
num_states: 3
num_obs: 3
num_actions: 3
"""


@pytest.fixture
def temp_dir() -> Any:
    """Create temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_gnn_model() -> Any:
    """Create sample parsed GNN model."""
    return {
        "model_name": "Simple Active Inference Agent",
        "version": "1.0",
        "variables": {
            "A": {
                "dimensions": [3, 3],
                "data_type": "float",
                "ontology_mapping": "LikelihoodMatrix",
                "description": "Likelihood matrix",
            },
            "B": {
                "dimensions": [3, 3, 3],
                "data_type": "float",
                "ontology_mapping": "TransitionMatrix",
                "description": "Transition matrix",
            },
            "C": {
                "dimensions": [3],
                "data_type": "float",
                "ontology_mapping": "LogPreferenceVector",
                "description": "Preference vector",
            },
            "D": {
                "dimensions": [3],
                "data_type": "float",
                "ontology_mapping": "PriorOverHiddenStates",
                "description": "Prior vector",
            },
            "s": {
                "dimensions": [3, 1],
                "data_type": "float",
                "ontology_mapping": "HiddenState",
                "description": "Hidden state",
            },
            "o": {
                "dimensions": [3, 1],
                "data_type": "int",
                "ontology_mapping": "Observation",
                "description": "Observation",
            },
            "u": {
                "dimensions": [1],
                "data_type": "int",
                "ontology_mapping": "Action",
                "description": "Action",
            },
        },
        "connections": [
            {
                "source": "D",
                "target": "s",
                "symbol": ">",
                "connection_type": "generative",
            },
            {
                "source": "s",
                "target": "A",
                "symbol": "-",
                "connection_type": "inference",
            },
            {
                "source": "A",
                "target": "o",
                "symbol": "-",
                "connection_type": "inference",
            },
            {
                "source": "s",
                "target": "B",
                "symbol": "-",
                "connection_type": "inference",
            },
            {
                "source": "u",
                "target": "B",
                "symbol": ">",
                "connection_type": "generative",
            },
        ],
        "parameters": {"num_states": 3, "num_obs": 3, "num_actions": 3},
        "ontology_mappings": [
            {"variable": "A", "ontology_term": "LikelihoodMatrix"},
            {"variable": "B", "ontology_term": "TransitionMatrix"},
            {"variable": "C", "ontology_term": "LogPreferenceVector"},
            {"variable": "D", "ontology_term": "PriorOverHiddenStates"},
            {"variable": "s", "ontology_term": "HiddenState"},
            {"variable": "o", "ontology_term": "Observation"},
            {"variable": "u", "ontology_term": "Action"},
        ],
    }


class TestModuleInfo:
    """Test module information and capabilities."""

    def test_get_module_info(self) -> Any:
        """Test module info retrieval."""
        info = get_module_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert info["name"] == "oxdraw"
        assert "version" in info
        assert "capabilities" in info
        assert isinstance(info["capabilities"], list)
        assert len(info["capabilities"]) > 0

    def test_check_oxdraw_installed(self) -> Any:
        """Test oxdraw CLI availability check."""
        result = check_oxdraw_installed()
        assert isinstance(result, bool)
        # Test should work regardless of whether oxdraw is installed

    def test_direct_mcp_registration_executes_keyword_schemas(
        self, temp_dir: Path
    ) -> None:
        """The nested oxdraw registry must work outside the parent GUI adapter."""
        from gnn.gui.oxdraw.mcp import register_tools
        from gnn.mcp.mcp import MCP

        instance = MCP(enable_caching=False, enable_rate_limiting=False)
        try:
            register_tools(instance)
            assert len(instance.tools) == 5
            assert all(
                tool.module == "gnn.gui.oxdraw" for tool in instance.tools.values()
            )
            assert instance.execute_tool("oxdraw.get_info", {})["success"] is True
            result = instance.execute_tool(
                "oxdraw.convert_to_mermaid",
                {"gnn_file_path": str(temp_dir / "missing.md")},
            )
            assert result["success"] is False
            assert "not found" in result["error"].lower()
        finally:
            instance.shutdown()


class TestGNNToMermaidConversion:
    """Test GNN to Mermaid conversion functionality."""

    def test_gnn_to_mermaid_basic(self, sample_gnn_model: Any) -> Any:
        """Test basic GNN to Mermaid conversion."""
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)

        assert isinstance(mermaid_content, str)
        assert "flowchart TD" in mermaid_content
        assert "Simple Active Inference Agent" in mermaid_content

        # Check for variables
        for var_name in ["A", "B", "C", "D", "s", "o", "u"]:
            assert var_name in mermaid_content

        # Check for connections
        assert "D ==>" in mermaid_content  # Generative
        assert "s -.->" in mermaid_content  # Inference

    def test_gnn_to_mermaid_with_metadata(self, sample_gnn_model: Any) -> Any:
        """Test metadata embedding in Mermaid."""
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)

        assert "GNN_METADATA_START" in mermaid_content
        assert "GNN_METADATA_END" in mermaid_content

        # Extract and parse metadata
        metadata = extract_gnn_metadata(mermaid_content)
        assert isinstance(metadata, dict)
        assert metadata["model_name"] == "Simple Active Inference Agent"
        assert "variables" in metadata
        assert "connections" in metadata

    def test_gnn_to_mermaid_without_metadata(self, sample_gnn_model: Any) -> Any:
        """Test conversion without metadata."""
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=False)

        assert "flowchart TD" in mermaid_content
        assert "GNN_METADATA" not in mermaid_content

    def test_convert_gnn_file_to_mermaid(
        self, sample_gnn_file: Any, temp_dir: Any
    ) -> Any:
        """Test file-based GNN to Mermaid conversion."""
        output_file = temp_dir / "test_model.mmd"

        mermaid_content = convert_gnn_file_to_mermaid(sample_gnn_file, output_file)

        assert output_file.exists()
        assert isinstance(mermaid_content, str)
        assert len(mermaid_content) > 0

        # Verify file content
        file_content = output_file.read_text()
        assert file_content == mermaid_content


class TestMermaidToGNNConversion:
    """Test Mermaid to GNN conversion functionality."""

    def test_mermaid_to_gnn_basic(self, sample_gnn_model: Any) -> Any:
        """Test basic Mermaid to GNN conversion."""
        # First convert to Mermaid
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)

        # Then convert back
        gnn_model = mermaid_to_gnn(mermaid_content, validate_ontology=False)

        assert isinstance(gnn_model, dict)
        assert "model_name" in gnn_model
        assert "variables" in gnn_model
        assert "connections" in gnn_model

        # Check variables preserved
        assert len(gnn_model["variables"]) == len(sample_gnn_model["variables"])

        # Check connections preserved
        assert len(gnn_model["connections"]) == len(sample_gnn_model["connections"])

    def test_extract_gnn_metadata(self, sample_gnn_model: Any) -> Any:
        """Test metadata extraction from Mermaid."""
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)
        metadata = extract_gnn_metadata(mermaid_content)

        assert isinstance(metadata, dict)
        assert metadata["model_name"] == sample_gnn_model["model_name"]
        assert "variables" in metadata
        assert len(metadata["variables"]) == len(sample_gnn_model["variables"])

    def test_convert_mermaid_file_to_gnn(
        self, sample_gnn_file: Any, temp_dir: Any
    ) -> Any:
        """Test file-based Mermaid to GNN conversion."""
        # First create Mermaid file
        mermaid_file = temp_dir / "test_model.mmd"
        convert_gnn_file_to_mermaid(sample_gnn_file, mermaid_file)

        # Then convert back
        output_gnn = temp_dir / "test_model_from_mermaid.md"
        gnn_model = convert_mermaid_file_to_gnn(mermaid_file, output_gnn)

        assert output_gnn.exists()
        assert isinstance(gnn_model, dict)
        assert "variables" in gnn_model
        assert len(gnn_model["variables"]) > 0


class TestRoundTripConversion:
    """Test round-trip conversion: GNN → Mermaid → GNN."""

    def test_round_trip_preserves_structure(self, sample_gnn_model: Any) -> Any:
        """Test that round-trip conversion preserves model structure."""
        # GNN → Mermaid
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)

        # Mermaid → GNN
        recovered_model = mermaid_to_gnn(mermaid_content, validate_ontology=False)

        # Compare structures
        assert len(recovered_model["variables"]) == len(sample_gnn_model["variables"])
        assert len(recovered_model["connections"]) == len(
            sample_gnn_model["connections"]
        )

        # Check variable names preserved
        original_vars = set(sample_gnn_model["variables"].keys())
        recovered_vars = set(recovered_model["variables"].keys())
        assert original_vars == recovered_vars

    def test_round_trip_preserves_ontology(self, sample_gnn_model: Any) -> Any:
        """Test that ontology mappings are preserved."""
        # GNN → Mermaid
        mermaid_content = gnn_to_mermaid(sample_gnn_model, include_metadata=True)

        # Mermaid → GNN
        recovered_model = mermaid_to_gnn(mermaid_content, validate_ontology=False)

        # Check ontology mappings
        assert "ontology_mappings" in recovered_model
        original_ontology = {
            m["variable"]: m["ontology_term"]
            for m in sample_gnn_model["ontology_mappings"]
        }
        recovered_ontology = {
            m["variable"]: m["ontology_term"]
            for m in recovered_model["ontology_mappings"]
        }

        for var_name in original_ontology:
            assert var_name in recovered_ontology
            assert original_ontology[var_name] == recovered_ontology[var_name]


class TestProcessOxdraw:
    """Test main processing function."""

    def test_process_oxdraw_headless(
        self, sample_gnn_file: Any, temp_dir: Any, capsys: Any
    ) -> Any:
        """Test headless processing mode."""
        import logging

        logger = logging.getLogger(__name__)

        output_dir = temp_dir / "output"

        success = process_oxdraw(
            target_dir=sample_gnn_file.parent,
            output_dir=output_dir,
            logger=logger,
            mode="headless",
            auto_convert=True,
            validate_on_save=False,
            launch_editor=False,
        )

        assert success, "oxdraw processing should succeed"
        assert output_dir.exists()

        # Check for results file
        results_file = output_dir / "oxdraw_processing_results.json"
        assert results_file.exists()

        # Parse results
        with open(results_file) as f:
            results = json.load(f)

        assert "gnn_to_mermaid_conversions" in results
        assert len(results["gnn_to_mermaid_conversions"]) > 0

        assert results["websocket_bridge"]["message_contract_available"] is True
        assert results["websocket_bridge"]["server_running"] is False
        assert results["websocket_bridge"]["status"] == "message_contract_only"
        messages = results["websocket_bridge"]["messages"]
        assert len(messages) == 1
        load_message = messages[0]
        assert load_message["type"] == "model.load"
        assert load_message["payload"]["model_id"] == sample_gnn_file.stem
        assert load_message["payload"]["format"] == "mermaid"
        assert load_message["payload"]["mermaid_file"].endswith(
            f"{sample_gnn_file.stem}.mmd"
        )
        assert "mermaid" not in load_message["payload"]
        assert Path(load_message["payload"]["mermaid_file"]).exists()

    def test_process_oxdraw_no_files(self, temp_dir: Any, capsys: Any) -> Any:
        """Test processing with no GNN files."""
        import logging

        logger = logging.getLogger(__name__)

        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        output_dir = temp_dir / "output"

        success = process_oxdraw(
            target_dir=empty_dir, output_dir=output_dir, logger=logger, mode="headless"
        )

        assert not success  # Should fail with no files


class TestProcessOxdrawTruth:
    """Truth contract for process_oxdraw results, summary, and phase gating."""

    def _run(
        self, sample_gnn_file: Any, temp_dir: Any, **kwargs: Any
    ) -> tuple[bool, dict[str, Any]]:
        """Run process_oxdraw headless and reload the results JSON."""
        import logging

        logger = logging.getLogger(__name__)
        output_dir = temp_dir / "output"

        success = process_oxdraw(
            target_dir=sample_gnn_file.parent,
            output_dir=output_dir,
            logger=logger,
            mode="headless",
            launch_editor=False,
            **kwargs,
        )

        results_file = output_dir / "oxdraw_processing_results.json"
        with open(results_file) as f:
            results = json.load(f)
        return success, results

    def test_phase3_converts_only_this_run_mermaid(
        self, sample_gnn_file: Any, temp_dir: Any
    ) -> Any:
        """Phase 3 converts only the mermaid files this run produced."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        stale = output_dir / "stale_prior_run.mmd"
        stale.write_text("flowchart TD\n    stale_node[stale]\n")

        success, results = self._run(
            sample_gnn_file, temp_dir, auto_convert=True, validate_on_save=True
        )

        assert len(results["mermaid_to_gnn_conversions"]) == 1
        entry = results["mermaid_to_gnn_conversions"][0]
        assert Path(entry["mermaid_file"]).name != "stale_prior_run.mmd"
        assert not (output_dir / "stale_prior_run_from_mermaid.md").exists()
        assert (output_dir / f"{sample_gnn_file.stem}_from_mermaid.md").exists()
        assert success is True

    def test_stale_mmd_never_resurrected_without_production(
        self, sample_gnn_file: Any, temp_dir: Any
    ) -> Any:
        """Without auto_convert no stale .mmd is back-converted."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        stale = output_dir / "stale_prior_run.mmd"
        stale.write_text("flowchart TD\n    stale_node[stale]\n")

        success, results = self._run(
            sample_gnn_file, temp_dir, auto_convert=False, validate_on_save=True
        )

        assert results["mermaid_to_gnn_conversions"] == []
        assert list(output_dir.glob("*_from_mermaid.md")) == []
        assert success is False
        assert results["summary"]["unexpected_failures"] == 0

    def test_validation_failure_surfaces_in_errors(
        self, sample_gnn_file: Any, temp_dir: Any, monkeypatch: Any
    ) -> Any:
        """A mermaid file failing syntax validation lands in results errors."""
        import gnn.gui.oxdraw.processor as processor_module

        def fake_convert(
            gnn_file: Any, mermaid_file: Any, *_args: Any, **_kwargs: Any
        ) -> str:
            path = Path(mermaid_file)
            path.write_text("this is not a flowchart")
            return "this is not a flowchart"

        monkeypatch.setattr(
            processor_module, "convert_gnn_file_to_mermaid", fake_convert
        )

        success, results = self._run(
            sample_gnn_file, temp_dir, auto_convert=True, validate_on_save=True
        )

        assert any(e["phase"] == "mermaid_validation" for e in results["errors"])
        assert success is False
        assert results["summary"]["unexpected_failures"] >= 1

    def test_back_conversion_failure_reflects_in_success(
        self, sample_gnn_file: Any, temp_dir: Any, monkeypatch: Any
    ) -> Any:
        """A failing back-conversion marks the m2g entry and overall success."""
        import gnn.gui.oxdraw.processor as processor_module

        def failing_back(
            mermaid_file: Any, *_args: Any, **_kwargs: Any
        ) -> dict[str, Any]:
            raise RuntimeError("back-conversion failed")

        monkeypatch.setattr(
            processor_module, "convert_mermaid_file_to_gnn", failing_back
        )

        success, results = self._run(
            sample_gnn_file, temp_dir, auto_convert=True, validate_on_save=True
        )

        assert success is False
        m2g_entries = results["mermaid_to_gnn_conversions"]
        assert len(m2g_entries) == 1
        assert m2g_entries[0]["success"] is False
        assert any(e["phase"] == "mermaid_to_gnn" for e in results["errors"])
        assert results["summary"]["mermaid_to_gnn_total"] == 1
        assert results["summary"]["mermaid_to_gnn_success"] == 0

    def test_summary_counts_present_on_clean_run(
        self, sample_gnn_file: Any, temp_dir: Any
    ) -> Any:
        """A clean run reports the full summary block and discovered files."""
        success, results = self._run(
            sample_gnn_file, temp_dir, auto_convert=True, validate_on_save=True
        )

        assert set(results["summary"]) == {
            "gnn_to_mermaid_success",
            "gnn_to_mermaid_total",
            "mermaid_to_gnn_success",
            "mermaid_to_gnn_total",
            "unexpected_failures",
        }
        assert results["summary"]["unexpected_failures"] == 0
        assert results["files_processed"] == [str(sample_gnn_file)]
        assert success is True


class TestOxdrawGuiOutputs:
    """oxdraw_gui result keys: recursive discovery count and output listing."""

    EXEMPLAR = (
        Path(__file__).resolve().parents[2] / "input/gnn_files/discrete/hmm_baseline.md"
    )

    @staticmethod
    def _write_target(target: Path) -> None:
        """Materialize the real exemplar at target root and nested/sub."""
        exemplar_text = TestOxdrawGuiOutputs.EXEMPLAR.read_text()
        target.mkdir(parents=True, exist_ok=True)
        target.joinpath(TestOxdrawGuiOutputs.EXEMPLAR.name).write_text(exemplar_text)
        nested_sub = target / "nested" / "sub"
        nested_sub.mkdir(parents=True, exist_ok=True)
        nested_sub.joinpath(TestOxdrawGuiOutputs.EXEMPLAR.name).write_text(
            exemplar_text
        )

    def _run_oxdraw_gui(self, tmp_dir: Path, validate_on_save: bool) -> dict[str, Any]:
        import logging

        from gnn.gui.oxdraw import oxdraw_gui

        target = tmp_dir / "input"
        output = tmp_dir / "output"
        self._write_target(target)

        result = oxdraw_gui(
            target_dir=target,
            output_dir=output,
            logger=logging.getLogger("test_oxdraw_integration"),
            validate_on_save=validate_on_save,
        )
        assert isinstance(result, dict)
        return result

    def test_files_processed_is_recursive_discovery_count(self, tmp_path: Any) -> None:
        """files_processed counts recursively discovered GNN files, not glob."""
        from gnn.processing.processor import discover_gnn_files

        result = self._run_oxdraw_gui(tmp_path, validate_on_save=False)

        target = tmp_path / "input"
        assert result["files_processed"] == len(
            discover_gnn_files(target, recursive=True)
        )
        assert result["files_processed"] == 2

    def test_outputs_include_from_mermaid_files(self, tmp_path: Any) -> None:
        """Validated runs list back-converted *_from_mermaid.md outputs."""
        result = self._run_oxdraw_gui(tmp_path, validate_on_save=True)

        assert any(str(o).endswith("_from_mermaid.md") for o in result["outputs"])
        assert result["success"] is True


class TestOxdrawGuiModeKwarg:
    """oxdraw_gui accepts an explicit ``mode`` kwarg without TypeError.

    Regression: oxdraw_gui splatted ``**kwargs`` into the closed-signature
    ``process_oxdraw`` after passing ``mode=mode`` explicitly, so any caller
    that forwarded ``mode`` raised
    ``TypeError: process_oxdraw() got multiple values for keyword argument 'mode'``
    and oxdraw_gui reported success=False.
    """

    EXEMPLAR = (
        Path(__file__).resolve().parents[2] / "input/gnn_files/discrete/hmm_baseline.md"
    )

    @staticmethod
    def _write_target(target: Path) -> None:
        """Materialize one real exemplar as the step-22 target directory."""
        target.mkdir(parents=True, exist_ok=True)
        target.joinpath(TestOxdrawGuiModeKwarg.EXEMPLAR.name).write_text(
            TestOxdrawGuiModeKwarg.EXEMPLAR.read_text()
        )

    def _run_oxdraw_gui(self, tmp_dir: Path, **kwargs: Any) -> dict[str, Any]:
        import logging

        from gnn.gui.oxdraw import oxdraw_gui

        target = tmp_dir / "input"
        output = tmp_dir / "output"
        self._write_target(target)

        result = oxdraw_gui(
            target_dir=target,
            output_dir=output,
            logger=logging.getLogger("test_oxdraw_integration"),
            verbose=False,
            validate_on_save=False,
            **kwargs,
        )
        assert isinstance(result, dict)
        return result

    @pytest.mark.unit
    @pytest.mark.fast
    def test_explicit_mode_kwarg_is_accepted(self, tmp_path: Any) -> None:
        """Passing mode="headless" explicitly must not crash the wrapper."""
        result = self._run_oxdraw_gui(tmp_path, mode="headless")

        assert result["success"] is True, f"oxdraw_gui failed: {result.get('error')}"
        assert result["mode"] == "headless"
        mmd_files = list((tmp_path / "output" / "oxdraw_output").glob("*.mmd"))
        assert mmd_files, "expected headless conversion to emit .mmd outputs"
        assert all(p.is_file() for p in mmd_files)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_derived_mode_headless_default(self, tmp_path: Any) -> None:
        """Without mode, headless=True (the pipeline default) derives "headless"."""
        result = self._run_oxdraw_gui(tmp_path, headless=True)

        assert result["success"] is True, f"oxdraw_gui failed: {result.get('error')}"
        assert result["mode"] == "headless"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_derived_mode_interactive_when_headless_false(self, tmp_path: Any) -> None:
        """Without mode, headless=False derives "interactive" without launching."""
        result = self._run_oxdraw_gui(tmp_path, headless=False, launch_editor=False)

        assert result["mode"] == "interactive"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_non_str_mode_is_derived(self, tmp_path: Any) -> None:
        """A non-str mode (e.g. True) derives the mode instead of crashing."""
        result = self._run_oxdraw_gui(tmp_path, mode=True)

        assert result["success"] is True, f"oxdraw_gui failed: {result.get('error')}"
        assert result["mode"] == "headless"


class TestMetadataGeneration:
    """Test metadata generation utilities."""

    def test_generate_mermaid_metadata(self, sample_gnn_model: Any) -> Any:
        """Test metadata generation."""
        metadata = generate_mermaid_metadata(sample_gnn_model)

        assert isinstance(metadata, dict)
        assert "model_name" in metadata
        assert "variables" in metadata
        assert "connections" in metadata
        assert "ontology_mappings" in metadata

        # Verify variables serialized correctly
        assert len(metadata["variables"]) == len(sample_gnn_model["variables"])

        # Verify connections serialized correctly
        assert len(metadata["connections"]) == len(sample_gnn_model["connections"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
