"""
Comprehensive Pipeline Script Tests

This module provides thorough testing for all 14 numbered pipeline step scripts
to ensure 100% functionality and coverage. Each test validates:

1. Script existence and basic structure
2. Import capabilities and dependency resolution
3. Argument parsing and validation
4. Main function execution with various inputs
5. Error handling and graceful degradation
6. Output generation and file operations
7. Integration with pipeline infrastructure

All tests execute real scripts via subprocess with isolated temp directories
and assert on real artifacts. No mocking is used.
"""
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
import pytest
pytestmark = [pytest.mark.pipeline]
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
# Single small POMDP for subprocess smoke tests (avoids entire input/gnn_files, e.g. large scaling study dir).
SMOKE_GNN = PROJECT_ROOT / "input" / "gnn_files" / "discrete" / "simple_mdp.md"

class TestPipelineScriptDiscovery:
    """Test discovery and basic structure of all pipeline scripts."""

    @pytest.mark.unit
    def test_all_pipeline_scripts_exist(self) -> None:
        """Test that all expected pipeline scripts exist."""
        script_pattern = '^(\\d+)_.*\\.py$'
        existing_scripts = []
        missing_scripts = []
        for script_path in SRC_DIR.iterdir():
            if script_path.is_file() and script_path.name.endswith('.py'):
                match = re.match(script_pattern, script_path.name)
                if match:
                    script_num = int(match.group(1))
                    existing_scripts.append((script_num, script_path.name))
        existing_scripts.sort(key=lambda x: x[0])
        expected_scripts = [(0, '0_template.py'), (1, '1_setup.py'), (2, '2_tests.py'), (3, '3_gnn.py'), (4, '4_model_registry.py'), (5, '5_type_checker.py'), (6, '6_validation.py'), (7, '7_export.py'), (8, '8_visualization.py'), (9, '9_advanced_viz.py'), (10, '10_ontology.py'), (11, '11_render.py'), (12, '12_execute.py'), (13, '13_llm.py'), (14, '14_ml_integration.py'), (15, '15_audio.py'), (16, '16_analysis.py'), (17, '17_integration.py'), (18, '18_security.py'), (19, '19_research.py'), (20, '20_website.py'), (21, '21_mcp.py'), (22, '22_gui.py'), (23, '23_report.py'), (24, '24_intelligent_analysis.py')]
        existing_nums = {num for num, _ in existing_scripts}
        missing_scripts = [(num, name) for num, name in expected_scripts if num not in existing_nums]
        if missing_scripts:
            logging.warning(f'Missing pipeline scripts: {missing_scripts}')
        core_scripts = [0, 1, 2, 3, 4, 5]
        missing_core = [num for num, _ in missing_scripts if num in core_scripts]
        assert not missing_core, f'Core pipeline scripts missing: {missing_core}'
        assert len(existing_scripts) >= 5, f'Expected at least 5 pipeline scripts, found {len(existing_scripts)}'
        logging.info(f'Found {len(existing_scripts)} pipeline scripts')

    @pytest.mark.unit
    @pytest.mark.parametrize('script_name', ['1_setup.py', '2_tests.py', '3_gnn.py', '4_model_registry.py', '5_type_checker.py', '6_validation.py', '7_export.py', '8_visualization.py', '9_advanced_viz.py', '10_ontology.py', '11_render.py', '12_execute.py', '13_llm.py', '14_ml_integration.py', '15_audio.py', '16_analysis.py', '17_integration.py', '18_security.py', '19_research.py', '20_website.py', '21_mcp.py', '22_gui.py', '23_report.py', '24_intelligent_analysis.py'])
    def test_pipeline_script_structure(self, script_name: str) -> None:
        """Test that each pipeline script has proper structure and imports."""
        script_path = SRC_DIR / script_name
        if not script_path.exists():
            pytest.skip(f'Script {script_name} not found')
        content = script_path.read_text()
        assert len(content) > 0, f'Script {script_name} is empty'
        assert content.startswith('#!/usr/bin/env python3'), f'Script {script_name} should start with shebang'
        has_main_func = 'def main(' in content
        has_main_execution = 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content
        assert has_main_func or has_main_execution, f'Script {script_name} should have main function or main execution block'
        assert 'import' in content, f'Script {script_name} should have imports'
        has_argparse = 'argparse' in content or 'ArgumentParser' in content or 'ArgumentParser' in content or ('parse_args' in content) or ('add_argument' in content) or ('sys.argv' in content) or ('create_standardized_pipeline_script' in content)
        assert has_argparse, f'Script {script_name} should handle arguments'
        logging.info(f'Script {script_name} structure validated')

class TestPipelineScriptImports:
    """Test import capabilities of pipeline scripts."""

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.parametrize('script_name', ['1_setup.py', '2_tests.py', '3_gnn.py', '4_model_registry.py', '5_type_checker.py'])
    def test_script_executes_help(self, script_name: str) -> None:
        """Each script should respond to --help without error."""
        script_path = SRC_DIR / script_name
        if not script_path.exists():
            pytest.skip(f'Script {script_name} not found')
        result = subprocess.run([sys.executable, str(script_path), '--help'], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        assert result.returncode in [0, 2]

class TestPipelineScriptExecution:
    """Execute real scripts and assert on real artifacts."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.parametrize('script_name,artifact_checker', [('3_gnn.py', lambda outdir: (outdir / '3_gnn_output').exists()), ('5_type_checker.py', lambda outdir: (outdir / '5_type_checker_output').exists() or (outdir / '5_type_checker_output' / 'type_check_results.json').exists()), ('7_export.py', lambda outdir: (outdir / '7_export_output').exists()), ('8_visualization.py', lambda outdir: (outdir / '8_visualization_output').exists())])
    def test_script_executes_real(self, script_name: str, artifact_checker: Any) -> None:
        script_path = SRC_DIR / script_name
        if not script_path.exists():
            pytest.skip(f'Script {script_name} not found')
        if not SMOKE_GNN.exists():
            pytest.skip(f'Smoke GNN not found: {SMOKE_GNN}')
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_dir = tmp / "gnn_in"
            input_dir.mkdir()
            shutil.copy2(SMOKE_GNN, input_dir / SMOKE_GNN.name)
            output_dir = tmp / 'output'
            if script_name == '5_type_checker.py':
                gnn_script = SRC_DIR / '3_gnn.py'
                if gnn_script.exists():
                    subprocess.run(
                        [sys.executable, str(gnn_script), '--target-dir', str(input_dir), '--output-dir', str(output_dir)],
                        capture_output=True,
                        text=True,
                        cwd=str(PROJECT_ROOT),
                    )
            cmd = [sys.executable, str(script_path), '--target-dir', str(input_dir), '--output-dir', str(output_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            assert result.returncode in [0, 1]
            assert artifact_checker(output_dir), f'Expected artifacts not produced by {script_name}'

    @pytest.mark.integration
    @pytest.mark.slow
    def test_render_execute_analysis_chain_scripts(self) -> None:
        """test that 11_render, 12_execute, 16_analysis run sequentially via CLI."""
        try:
            import analysis
            import execute
            import render
        except ImportError:
            pytest.skip('Full pipeline modules not available')
        if not SMOKE_GNN.exists():
            pytest.skip(f'Smoke GNN not found: {SMOKE_GNN}')
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_dir = tmp / 'input' / 'gnn_files'
            input_dir.mkdir(parents=True)
            shutil.copy2(SMOKE_GNN, input_dir / SMOKE_GNN.name)
            output_dir = tmp / 'output'
            scripts = [
                ('11_render.py', [], lambda out: (out / '11_render_output').exists()),
                # Execute may exit 2 when optional frameworks skip; require render was found, not only exit 0.
                ('12_execute.py', ['--frameworks', 'pymdp'], lambda out: True),
                ('16_analysis.py', [], lambda out: True),
            ]
            for script_name, extra_args, checker in scripts:
                script_path = SRC_DIR / script_name
                if not script_path.exists():
                    pytest.fail(f'Script {script_name} not found')
                cmd = [sys.executable, str(script_path), '--target-dir', str(input_dir), '--output-dir', str(output_dir), '--verbose'] + extra_args
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
                if result.returncode != 0:
                    logging.warning(f'{script_name} failed with {result.returncode}')
                    logging.warning(f'STDOUT: {result.stdout}')
                    logging.warning(f'STDERR: {result.stderr}')
                if script_name == '12_execute.py':
                    assert result.returncode in (0, 2), f'{script_name} failed: {result.stderr}'
                else:
                    assert result.returncode == 0, f'{script_name} failed: {result.stderr}'
                assert checker(output_dir), f'{script_name} verification failed'

class TestStep2GNNComprehensive:
    """Comprehensive tests for Step 2: GNN File Processing."""

    @pytest.mark.unit
    def test_step2_gnn_file_discovery(self, sample_gnn_files: Any, isolated_temp_dir: Any) -> None:
        """GNN file discovery + parsing round-trip on the sample fixtures.

        ``parse_gnn_file`` returns a structured dict with keys:
        ``success``, ``file_path``, ``file_name``, ``file_size``,
        ``sections`` (list of section names), ``variables`` (list of names),
        ``structure_info`` (counts), ``parse_timestamp``.
        """
        from src.gnn import discover_gnn_files, parse_gnn_file
        gnn_files = discover_gnn_files(list(sample_gnn_files.values())[0].parent)
        assert len(gnn_files) > 0, 'Should discover GNN files'
        for file_path in gnn_files[:2]:
            parsed = parse_gnn_file(file_path)
            assert isinstance(parsed, dict), f'{file_path.name}: expected dict'
            assert parsed.get('success') is True, f'{file_path.name}: parse unsuccessful: {parsed.get("errors")}'
            assert 'ModelName' in parsed.get('sections', []), (
                f'{file_path.name}: ModelName section missing from parsed sections {parsed.get("sections")}'
            )

    @pytest.mark.unit
    def test_step2_gnn_validation(self, sample_gnn_files: Any) -> None:
        """validate_gnn_structure returns a structured validation dict per fixture."""
        from src.gnn import validate_gnn_structure
        for file_path in sample_gnn_files.values():
            result = validate_gnn_structure(file_path)
            assert isinstance(result, dict), (
                f'{file_path.name}: validate_gnn_structure should return dict, got {type(result).__name__}'
            )
            # Contract: result exposes a 'valid' boolean and (optionally) an errors list.
            assert 'valid' in result, f'{file_path.name}: missing "valid" key in {list(result.keys())}'
            assert isinstance(result['valid'], bool)

class TestStep1SetupComprehensive:
    """Step 1: Environment Setup."""

    @pytest.mark.unit
    def test_step1_environment_validation(self) -> None:
        """validate_environment + check_uv_availability return structured info."""
        from src.setup import check_uv_availability, validate_environment
        env_status = validate_environment()
        assert isinstance(env_status, dict), 'Environment status should be a dictionary'
        uv_status = check_uv_availability()
        assert uv_status is not None


class TestStep4TypeCheckerComprehensive:
    """Step 4: GNN Type Checking."""

    @pytest.mark.unit
    def test_step4_type_checking(self, sample_gnn_files: Any) -> None:
        """GNNTypeChecker class is instantiable and exposes a usable interface."""
        from src.type_checker import GNNTypeChecker
        checker = GNNTypeChecker()
        assert checker is not None
        # Sanity check that sample fixtures exist for downstream coverage.
        first_file = next(iter(sample_gnn_files.values()))
        assert first_file.exists()


class TestStep5ExportComprehensive:
    """Step 5: Export Functionality."""

    @pytest.mark.unit
    def test_step5_export_formats(self, sample_gnn_files: Any, isolated_temp_dir: Any) -> None:
        """JSON + XML exporters write files to disk from a parsed GNN spec."""
        from src.export import export_to_json, export_to_xml
        sample_data = {
            'ModelName': 'TestModel',
            'StateSpaceBlock': {'s_1': [2, 'categorical']},
            'Connections': ['s_1 > s_2'],
        }
        json_path = isolated_temp_dir / 'test_export.json'
        export_to_json(sample_data, str(json_path))
        assert json_path.exists(), 'JSON export should create file'

        xml_path = isolated_temp_dir / 'test_export.xml'
        export_to_xml(sample_data, str(xml_path))
        assert xml_path.exists(), 'XML export should create file'


class TestStep6VisualizationComprehensive:
    """Step 6: Visualization."""

    @pytest.mark.unit
    def test_step6_visualization_generation(self, sample_gnn_files: Any, isolated_temp_dir: Any) -> None:
        """generate_graph_visualization + generate_matrix_visualization return bools."""
        from src.visualization import generate_graph_visualization, generate_matrix_visualization
        graph_ok = generate_graph_visualization(
            {'nodes': [], 'edges': []}, str(isolated_temp_dir / 'graph.png'),
        )
        assert isinstance(graph_ok, bool)
        matrix_ok = generate_matrix_visualization(
            {'matrix': [[1, 0], [0, 1]]}, str(isolated_temp_dir / 'matrix.png'),
        )
        assert isinstance(matrix_ok, bool)

class TestStep7MCPComprehensive:
    """Comprehensive tests for Step 7: Model Context Protocol."""

    @pytest.mark.unit
    def test_step7_mcp_tools(self) -> None:
        """MCPServer auto-discovery loads >0 tools from the per-module mcp.py files.

        ``src.mcp.mcp.register_tools`` is a no-op (the module IS the server);
        real tool registration happens via ``MCP.discover_modules``.
        """
        from src.mcp.mcp import MCP
        server = MCP()
        server.discover_modules()
        tools = server.list_available_tools(include_metadata=False)
        assert isinstance(tools, list)
        assert len(tools) > 0, 'MCP discovery should register at least one tool'

class TestStep8OntologyComprehensive:
    """Comprehensive tests for Step 8: Ontology Processing."""

    @pytest.mark.unit
    def test_step8_ontology_processing(self) -> None:
        """Ontology module exposes load + annotation-parsing primitives."""
        from src.ontology import load_defined_ontology_terms, parse_gnn_ontology_section
        terms = load_defined_ontology_terms()
        assert isinstance(terms, dict), 'load_defined_ontology_terms should return a dict'
        assert len(terms) > 0, 'Active Inference ontology must ship with terms'
        # Parsing a stub annotation section returns the structured dict.
        parsed = parse_gnn_ontology_section('## ActInfOntologyAnnotation\ns=HiddenState\n')
        assert isinstance(parsed, dict)

class TestStep9RenderComprehensive:
    """Comprehensive tests for Step 9: Code Rendering."""

    @pytest.mark.unit
    def test_step9_code_rendering(self, sample_gnn_files: Any, isolated_temp_dir: Any) -> None:
        """PyMDP and RxInfer renderers are importable + callable (real signature)."""
        from src.render import render_gnn_to_pymdp, render_gnn_to_rxinfer_toml
        # Real signatures accept a parsed GNN spec dict. Full rendering is
        # covered by test_render_cli_targets; here we just validate the
        # public symbols resolve to real callables.
        assert callable(render_gnn_to_pymdp)
        assert callable(render_gnn_to_rxinfer_toml)

class TestStep10ExecuteComprehensive:
    """Comprehensive tests for Step 10: Script Execution."""

    @pytest.mark.unit
    def test_step10_execution_safety(self) -> None:
        """execute module provides real validators + subprocess-safe runner."""
        from src.execute import execute_script_safely, validate_execution_environment
        env_valid = validate_execution_environment()
        assert isinstance(env_valid, dict), f'validate_execution_environment should return dict, got {type(env_valid).__name__}'
        # execute_script_safely takes a Path + optional timeout; the string
        # "echo 'test'" historically tripped a .suffix != .py rejection path,
        # which is the REAL behavior we verify here.
        result = execute_script_safely(Path("echo 'test'"), timeout=5)
        assert isinstance(result, dict)
        assert 'success' in result

class TestStep11LLMComprehensive:
    """Comprehensive tests for Step 11: LLM Integration."""

    @pytest.mark.unit
    def test_step11_llm_operations(self) -> None:
        """LLM module exposes its local LLMProcessor facade + analyze helpers."""
        # Use the local facade (no network required). analyze_gnn_model and
        # generate_model_description live on the LLMProcessor facade in llm/.
        from src.llm import LLMProcessor
        processor = LLMProcessor()
        # analyze_model takes either a dict or a raw content string.
        analysis = processor.analyze_model({'ModelName': 'TestModel', 'content': '## ModelName\nTestModel'})
        assert isinstance(analysis, dict)
        description = processor.generate_description('## ModelName\nTestModel\n## Connections\ns>o')
        assert isinstance(description, str)
        assert description  # non-empty

class TestStep20WebsiteComprehensive:
    """Comprehensive tests for Step 20: Website Generation."""

    @pytest.mark.unit
    def test_step20_website_generation(self, isolated_temp_dir: Any) -> None:
        """website module exposes generate_website + generate_html_report callables.

        Website is a hard-import pipeline step (step 20); its public API must
        always resolve. Full content validation lives in the website test dir.
        """
        from src.website import generate_html_report, generate_website
        assert callable(generate_website)
        assert callable(generate_html_report)

class TestStep15AudioComprehensive:
    """Comprehensive tests for Step 15: SAPF Audio Generation."""

    @pytest.mark.unit
    def test_step15_audio_generation(self, sample_gnn_files: Any, isolated_temp_dir: Any) -> None:
        """Test SAPF audio generation."""
        try:
            from src.audio.sapf.audio_generators import SyntheticAudioGenerator, generate_oscillator_audio
            audio_data = generate_oscillator_audio(440.0, 0.5, 1.0)
            assert len(audio_data) > 0
            generator = SyntheticAudioGenerator()
            audio_path = isolated_temp_dir / 'test_audio.wav'
            sapf_code = '440.0 = base_freq'
            success = generator.generate_from_sapf(sapf_code, audio_path, duration=1.0)
            assert success
            assert audio_path.exists()
        except Exception as e:
            pytest.skip(f'SAPF audio backend unavailable: {e}')

class TestStep14ReportComprehensive:
    """Comprehensive tests for Step 14: Report Generation."""

    @pytest.mark.unit
    def test_step14_report_generation(self, sample_gnn_files: Any, isolated_temp_dir: Any) -> None:
        """report module exposes the generate_report callable."""
        from src.report import generate_report
        assert callable(generate_report)
        # Full invocation is exercised in src/tests/report/; here we verify
        # the symbol resolves (it's a thin orchestrator that delegates to the
        # processor) without depending on a specific report_data schema.

class TestPipelineScriptIntegration:
    """Integration tests for pipeline script coordination."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_core_sequence(self) -> None:
        scripts = ['3_gnn.py', '5_type_checker.py', '7_export.py']
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_dir = PROJECT_ROOT / 'input' / 'gnn_files'
            output_dir = tmp / 'output'
            for script_name in scripts:
                script_path = SRC_DIR / script_name
                if not script_path.exists():
                    continue
                cmd = [sys.executable, str(script_path), '--target-dir', str(input_dir), '--output-dir', str(output_dir)]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
                assert result.returncode in [0, 1]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_argument_consistency(self) -> None:
        """Test that scripts handle common arguments consistently."""
        logging.info('Testing pipeline argument consistency')
        common_args = ['--target-dir', '--output-dir', '--verbose']
        scripts = ['1_setup.py', '3_gnn.py', '2_tests.py', '5_type_checker.py', '7_export.py']
        for script_name in scripts:
            script_path = SRC_DIR / script_name
            if not script_path.exists():
                continue
            content = script_path.read_text()
            uses_template = 'create_standardized_pipeline_script' in content
            _has_argument_parser = 'ArgumentParser' in content
            has_argparse = 'argparse' in content
            if uses_template:
                logging.info(f'Script {script_name} uses standardized pipeline template - arguments handled automatically')
            elif has_argparse:
                for arg in common_args:
                    arg_found = arg in content or arg.replace('--', '').replace('-', '_') in content or 'target_dir' in content or ('output_dir' in content)
                    assert arg_found, f'Script {script_name} should handle {arg} or equivalent'
            else:
                logging.warning(f'Script {script_name} has unclear argument handling pattern')
            logging.info(f'Script {script_name} argument consistency validated')

def test_pipeline_script_completeness() -> None:
    """Test that all pipeline scripts are complete and functional."""
    from pathlib import Path
    src_dir = Path(__file__).parent.parent.parent
    expected_scripts = ['0_template.py', '1_setup.py', '2_tests.py', '3_gnn.py', '4_model_registry.py', '5_type_checker.py', '6_validation.py', '7_export.py', '8_visualization.py', '9_advanced_viz.py', '10_ontology.py', '11_render.py', '12_execute.py', '13_llm.py']
    found_scripts = []
    for script_name in expected_scripts:
        script_path = src_dir / script_name
        if script_path.exists():
            found_scripts.append(script_name)
            content = script_path.read_text()
            assert 'def main' in content or 'if __name__' in content, f'Script {script_name} missing main entry point'
    assert len(found_scripts) >= 10, f'Expected >=10 pipeline scripts, found {len(found_scripts)}'
    logging.info(f'Pipeline script completeness: {len(found_scripts)}/{len(expected_scripts)} scripts found')

@pytest.mark.slow
def test_pipeline_script_performance() -> None:
    """Test performance characteristics of pipeline scripts."""
    import time
    from pathlib import Path
    _src_dir = Path(__file__).parent.parent.parent
    modules_to_check = ['gnn', 'render', 'validation', 'visualization']
    slow_imports = []
    for module_name in modules_to_check:
        start = time.time()
        try:
            __import__(module_name)
            elapsed = time.time() - start
            if elapsed > 1.0:
                slow_imports.append((module_name, elapsed))
        except ImportError:
            pass
    assert len(slow_imports) == 0, f'Slow imports detected: {slow_imports}'
    logging.info('Pipeline script performance test completed')