import json
from typing import Any
import pytest
from analysis.processor import process_analysis

class TestAnalysisOverall:
    """Test suite for Analysis module."""

    @pytest.fixture
    def sample_gnn_for_analysis(self, safe_filesystem: Any) -> Any:
        """Create a sample GNN file to analyze."""
        content = '\n# Analysis Target\n\n## StateSpaceBlock\ns[10, type=float]\n\n## Connections\ns->s\n\n## Time\nDynamic\n'
        return safe_filesystem.create_file('model_analysis.md', content)

    def test_process_analysis_flow(self, safe_filesystem: Any, sample_gnn_for_analysis: Any) -> None:
        """Test the analysis processing workflow."""
        target_dir = sample_gnn_for_analysis.parent
        output_dir = safe_filesystem.create_dir('analysis_output')
        try:
            success = process_analysis(target_dir, output_dir, verbose=True)
            assert success is True
            results_dir = output_dir
            assert results_dir.exists()
            assert (results_dir / 'analysis_results.json').exists()
            assert (results_dir / 'analysis_summary.md').exists()
            with open(results_dir / 'analysis_results.json', 'r') as f:
                data = json.load(f)
            assert data['processed_files'] == 1
            assert len(data['statistical_analysis']) == 1
        except ImportError:
            pytest.skip('Skipping analysis test due to missing dependencies (numpy/matplotlib)')
        except Exception as e:
            pytest.fail(f'Analysis processing failed: {e}')

    def test_process_analysis_no_files(self, safe_filesystem: Any) -> None:
        """Test behavior with no files.

        Phase 1.1 contract: "no input" is a warning (exit-code 2), NOT a hard
        error. Previously this returned False, which the pipeline template then
        translated to exit-code 1 (error). Now it returns 2 so operators see
        "nothing to do" as a distinct signal from "ran and failed".
        """
        empty_dir = safe_filesystem.create_dir('empty')
        output_dir = safe_filesystem.create_dir('output')
        result = process_analysis(empty_dir, output_dir)
        assert result == 2, f'expected exit-code 2 for no-input, got {result!r}'

class TestPostSimulationVisualization:
    """Test suite for post-simulation visualization functions."""

    def test_generate_belief_heatmaps(self, safe_filesystem: Any) -> None:
        """Test belief heatmap generation."""
        from analysis.post_simulation import generate_belief_heatmaps
        beliefs = [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.6, 0.2], [0.1, 0.7, 0.2], [0.1, 0.5, 0.4], [0.1, 0.3, 0.6], [0.1, 0.2, 0.7], [0.1, 0.1, 0.8]]
        output_dir = safe_filesystem.create_dir('viz_output')
        output_file = output_dir / 'belief_heatmap.png'
        result = generate_belief_heatmaps(beliefs, output_file, 'Test Belief Heatmap')
        assert result == str(output_file)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_generate_action_analysis(self, safe_filesystem: Any) -> None:
        """Test action analysis visualization."""
        from analysis.post_simulation import generate_action_analysis
        actions = [0, 1, 2, 0, 1, 1, 2, 2, 0, 0, 1, 2, 2, 1, 0, 1, 2, 0, 1, 2]
        output_dir = safe_filesystem.create_dir('viz_output')
        output_file = output_dir / 'action_analysis.png'
        result = generate_action_analysis(actions, output_file, 'Test Action Analysis')
        assert result == str(output_file)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_generate_free_energy_plots(self, safe_filesystem: Any) -> None:
        """Test free energy plot generation."""
        from analysis.post_simulation import generate_free_energy_plots
        free_energy = [10.0, 9.5, 9.0, 8.5, 8.2, 7.8, 7.5, 7.2, 7.0, 6.8, 6.5, 6.3, 6.1, 6.0, 5.9, 5.8, 5.7, 5.6, 5.5, 5.5]
        output_dir = safe_filesystem.create_dir('viz_output')
        output_file = output_dir / 'free_energy.png'
        result = generate_free_energy_plots(free_energy, output_file, 'Test Free Energy')
        assert result == str(output_file)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_generate_observation_analysis(self, safe_filesystem: Any) -> None:
        """Test observation analysis visualization."""
        from analysis.post_simulation import generate_observation_analysis
        observations = [0, 1, 0, 2, 1, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1]
        output_dir = safe_filesystem.create_dir('viz_output')
        output_file = output_dir / 'observations.png'
        result = generate_observation_analysis(observations, output_file, 'Test Observations')
        assert result == str(output_file)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_analyze_free_energy(self) -> None:
        """Test free energy analysis function."""
        from analysis.post_simulation import analyze_free_energy
        fe_values = [10.0, 8.0, 6.0, 4.0, 3.0, 2.5, 2.2, 2.1, 2.05, 2.02]
        result = analyze_free_energy(fe_values, 'pymdp', 'test_model')
        assert result['framework'] == 'pymdp'
        assert result['model_name'] == 'test_model'
        assert result['free_energy_count'] == 10
        assert 'mean_free_energy' in result
        assert 'std_free_energy' in result
        assert result['free_energy_decreasing']

    def test_analyze_simulation_traces(self) -> None:
        """Test simulation trace analysis function."""
        from analysis.post_simulation import analyze_simulation_traces
        traces = [[0, 1, 2, 1, 0], [1, 2, 2, 0, 1, 2], [0, 0, 1, 2]]
        result = analyze_simulation_traces(traces, 'rxinfer', 'test_model')
        assert result['framework'] == 'rxinfer'
        assert result['trace_count'] == 3
        assert result['trace_lengths'] == [5, 6, 4]
        assert result['avg_trace_length'] == 5.0

    def test_analyze_policy_convergence(self) -> None:
        """Test policy convergence analysis."""
        from analysis.post_simulation import analyze_policy_convergence
        policy_traces = [[0.33, 0.33, 0.34], [0.4, 0.3, 0.3], [0.5, 0.25, 0.25], [0.7, 0.15, 0.15], [0.9, 0.05, 0.05]]
        result = analyze_policy_convergence(policy_traces, 'jax', 'test_model')
        assert result['framework'] == 'jax'
        assert result['policy_count'] == 5
        assert len(result['policy_entropy']) == 5
        assert result['policy_entropy'][0] > result['policy_entropy'][-1]

    def test_compare_framework_results(self) -> None:
        """Test cross-framework comparison."""
        from analysis.post_simulation import compare_framework_results
        framework_results = {'pymdp': {'success': True, 'execution_time': 1.5, 'simulation_data': {'free_energy': [10.0, 8.0, 6.0]}}, 'rxinfer': {'success': True, 'execution_time': 0.8, 'simulation_data': {'free_energy': [12.0, 9.0, 7.0]}}}
        result = compare_framework_results(framework_results, 'test_model')
        assert result['framework_count'] == 2
        assert 'pymdp' in result['frameworks_compared']
        assert 'rxinfer' in result['frameworks_compared']
        assert result['comparisons']['fastest_execution']['framework'] == 'rxinfer'

class TestAnalysisModuleImports:
    """Test that all new visualization functions are properly exported."""

    def test_visualization_function_exports(self) -> None:
        """Test that new visualization functions are exported from analysis module."""
        from analysis import animate_belief_evolution, generate_action_analysis, generate_belief_heatmaps, generate_cross_framework_comparison, generate_free_energy_plots, generate_observation_analysis, plot_belief_evolution, visualize_all_framework_outputs
        assert callable(visualize_all_framework_outputs)
        assert callable(generate_belief_heatmaps)
        assert callable(generate_action_analysis)
        assert callable(generate_free_energy_plots)
        assert callable(generate_observation_analysis)
        assert callable(generate_cross_framework_comparison)
        assert callable(plot_belief_evolution)
        assert callable(animate_belief_evolution)

class TestActiveInferenceJLAnalyzer:

    def test_module_importable(self):
        from analysis.activeinference_jl import analyzer

    def test_generate_analysis_from_logs_missing_dir(self, tmp_path):
        from analysis.activeinference_jl.analyzer import generate_analysis_from_logs
        result = generate_analysis_from_logs(tmp_path / 'nonexistent', tmp_path / 'out')
        assert isinstance(result, list)

    def test_generate_analysis_from_logs_empty_dir(self, tmp_path):
        from analysis.activeinference_jl.analyzer import generate_analysis_from_logs
        result = generate_analysis_from_logs(tmp_path, tmp_path / 'out')
        assert isinstance(result, list)

class TestDisCoPyAnalyzer:

    def test_module_importable(self):
        from analysis.discopy import analyzer

    def test_extract_circuit_data_empty_dir(self, tmp_path):
        from analysis.discopy.analyzer import extract_circuit_data
        result = extract_circuit_data(tmp_path)
        assert isinstance(result, dict)

    def test_analyze_diagram_structure_empty(self):
        from analysis.discopy.analyzer import analyze_diagram_structure
        result = analyze_diagram_structure([])
        assert isinstance(result, dict)

    def test_generate_analysis_from_logs_empty_dir(self, tmp_path):
        from analysis.discopy.analyzer import generate_analysis_from_logs
        result = generate_analysis_from_logs(tmp_path, tmp_path / 'out')
        assert isinstance(result, list)

class TestJAXAnalyzer:

    def test_module_importable(self):
        from analysis.jax import analyzer

    def test_parse_raw_output_empty_string(self):
        from analysis.jax.analyzer import parse_raw_output
        result = parse_raw_output('')
        assert isinstance(result, dict)

    def test_extract_simulation_data_empty_dir(self, tmp_path):
        from analysis.jax.analyzer import extract_simulation_data
        result = extract_simulation_data(tmp_path)
        assert isinstance(result, dict)

    def test_generate_analysis_from_logs_empty_dir(self, tmp_path):
        from analysis.jax.analyzer import generate_analysis_from_logs
        result = generate_analysis_from_logs(tmp_path, tmp_path / 'out')
        assert isinstance(result, list)

class TestPyMDPAnalyzer:

    def test_module_importable(self):
        from analysis.pymdp import analyzer

    def test_generate_analysis_from_logs_missing_dir(self, tmp_path):
        from analysis.pymdp.analyzer import generate_analysis_from_logs
        result = generate_analysis_from_logs(tmp_path / 'nonexistent', tmp_path / 'out')
        assert isinstance(result, list)

    def test_generate_analysis_from_logs_empty_dir(self, tmp_path):
        from analysis.pymdp.analyzer import generate_analysis_from_logs
        result = generate_analysis_from_logs(tmp_path, tmp_path / 'out')
        assert isinstance(result, list)

class TestRxInferAnalyzer:

    def test_module_importable(self):
        from analysis.rxinfer import analyzer

    def test_extract_simulation_data_empty_dir(self, tmp_path):
        from analysis.rxinfer.analyzer import extract_simulation_data
        result = extract_simulation_data(tmp_path)
        assert isinstance(result, dict)

    def test_generate_analysis_from_logs_empty_dir(self, tmp_path):
        from analysis.rxinfer.analyzer import generate_analysis_from_logs
        result = generate_analysis_from_logs(tmp_path, tmp_path / 'out')
        assert isinstance(result, list)

class TestAnalyzerSimulationMetrics:
    """Behavioral tests for analyzer.py private simulation metric functions."""

    def _make_logger(self):
        import logging
        return logging.getLogger('test_analyzer')

    def test_extract_simulation_metrics_returns_dict(self, tmp_path):
        """_extract_simulation_metrics returns a dict with expected keys."""
        from analysis.analyzer import _extract_simulation_metrics
        logger = self._make_logger()
        result = _extract_simulation_metrics('pymdp', [], tmp_path, logger)
        assert isinstance(result, dict)
        assert 'beliefs' in result
        assert 'actions' in result
        assert 'observations' in result
        assert 'free_energy' in result
        assert 'execution_times' in result

    def test_extract_simulation_metrics_reads_json(self, tmp_path):
        """_extract_simulation_metrics loads simulation_results.json when present."""
        import json
        from analysis.analyzer import _extract_simulation_metrics
        sim_dir = tmp_path / 'sim_data'
        sim_dir.mkdir()
        sim_results = {'beliefs': [[0.9, 0.1], [0.8, 0.2]], 'actions': [0, 1], 'observations': [2, 3], 'free_energy': [-1.5, -1.3]}
        (sim_dir / 'simulation_results.json').write_text(json.dumps(sim_results))
        detail = {'implementation_directory': str(sim_dir), 'execution_time': 0.5}
        logger = self._make_logger()
        result = _extract_simulation_metrics('pymdp', [detail], tmp_path, logger)
        assert result['beliefs'] == sim_results['beliefs']
        assert result['actions'] == sim_results['actions']
        assert result['free_energy'] == sim_results['free_energy']
        assert result['execution_times'] == [0.5]

    def test_extract_simulation_metrics_missing_dir(self, tmp_path):
        """_extract_simulation_metrics handles nonexistent impl_dir gracefully."""
        from analysis.analyzer import _extract_simulation_metrics
        logger = self._make_logger()
        detail = {'implementation_directory': str(tmp_path / 'nonexistent'), 'execution_time': 1.0}
        result = _extract_simulation_metrics('rxinfer', [detail], tmp_path, logger)
        assert isinstance(result, dict)
        assert result['execution_times'] == [1.0]

    def test_extract_simulation_metrics_bnlearn_execution_logs(self, tmp_path):
        """bnlearn writes execution_logs/*_results.json; metrics should still record completion."""
        import json
        from analysis.analyzer import _extract_simulation_metrics
        impl = tmp_path / 'markov_chain' / 'bnlearn'
        el = impl / 'execution_logs'
        el.mkdir(parents=True)
        structured = {'framework': 'bnlearn', 'model_name': 'markov_chain', 'success': True, 'simulation_data': {'beliefs': [], 'actions': [], 'observations': []}}
        (el / 'Simple_bnlearn.py_results.json').write_text(json.dumps(structured))
        detail = {'implementation_directory': str(impl), 'execution_time': 1.0}
        logger = self._make_logger()
        result = _extract_simulation_metrics('bnlearn', [detail], tmp_path, logger)
        assert result['model_parameters'].get('bnlearn_completed') is True
        assert result['model_parameters'].get('model_name') == 'markov_chain'
        assert result['data_source']

    def test_extract_simulation_metrics_rxinfer_prefers_simulation_data(self, tmp_path):
        """simulation_data/simulation_results.json must win over sparse execution_logs."""
        import json
        from analysis.analyzer import _extract_simulation_metrics
        impl = tmp_path / 'rx' / 'rxinfer'
        el = impl / 'execution_logs'
        sd = impl / 'simulation_data'
        el.mkdir(parents=True)
        sd.mkdir(parents=True)
        sparse = {'framework': 'rxinfer', 'model_name': 'markov_chain', 'success': True, 'simulation_data': {'beliefs': [], 'actions': [], 'observations': []}}
        (el / 'Model_rxinfer.jl_results.json').write_text(json.dumps(sparse))
        rich = {'beliefs': [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]], 'actions': [1, 1], 'observations': [2, 3], 'efe_history': [0.1, 0.2]}
        (sd / 'simulation_results.json').write_text(json.dumps(rich))
        detail = {'implementation_directory': str(impl), 'execution_time': 1.0}
        logger = self._make_logger()
        result = _extract_simulation_metrics('rxinfer', [detail], tmp_path, logger)
        assert result['beliefs'] == rich['beliefs']
        assert 'simulation_data' in result['data_source'].replace('\\', '/')
        assert 'simulation_results.json' in result['data_source'].replace('\\', '/')

    def test_extract_simulation_metrics_discopy_supplements_circuit_info(self, tmp_path):
        """DisCoPy: execution log plus circuit_info.json yields circuit metrics (no empty extract)."""
        import json
        from analysis.analyzer import _extract_simulation_metrics
        impl = tmp_path / 'm' / 'discopy'
        el = impl / 'execution_logs'
        sd = impl / 'simulation_data'
        el.mkdir(parents=True)
        sd.mkdir(parents=True)
        stub = {'framework': 'discopy', 'model_name': 'markov_chain', 'success': True, 'simulation_data': {'traces': [], 'beliefs': [], 'actions': [], 'observations': []}}
        (el / 'Model_discopy.py_results.json').write_text(json.dumps(stub))
        circuit_data = {'model_name': 'markov_chain', 'components': ['A_matrix', 'B_matrix'], 'analysis': {'num_components': 8}, 'parameters': {'num_states': 3, 'num_observations': 3, 'num_actions': 1}}
        (sd / 'circuit_info.json').write_text(json.dumps(circuit_data))
        detail = {'implementation_directory': str(impl), 'execution_time': 0.5}
        logger = self._make_logger()
        result = _extract_simulation_metrics('discopy', [detail], tmp_path, logger)
        assert result.get('circuit_info') is not None
        assert result['circuit_info'].get('num_components') == 8
        assert result.get('model_parameters', {}).get('num_states') == 3

    def test_compare_framework_results_empty_input(self):
        """_compare_framework_results returns dict with expected keys for empty input."""
        from analysis.analyzer import _compare_framework_results
        logger = self._make_logger()
        result = _compare_framework_results({}, logger)
        assert isinstance(result, dict)
        assert 'success_rates' in result
        assert 'performance_comparison' in result
        assert 'data_coverage' in result
        assert 'simulation_statistics' in result

    def test_compare_framework_results_success_rates(self):
        """_compare_framework_results computes success rates correctly."""
        from analysis.analyzer import _compare_framework_results
        logger = self._make_logger()
        framework_data = {'pymdp': {'success_count': 3, 'total_count': 4, 'execution_times': []}, 'jax': {'success_count': 4, 'total_count': 4, 'execution_times': []}}
        result = _compare_framework_results(framework_data, logger)
        assert abs(result['success_rates']['pymdp'] - 0.75) < 1e-06
        assert abs(result['success_rates']['jax'] - 1.0) < 1e-06

    def test_compare_framework_results_execution_times(self):
        """_compare_framework_results computes perf stats when times present."""
        from analysis.analyzer import _compare_framework_results
        logger = self._make_logger()
        framework_data = {'pymdp': {'success_count': 2, 'total_count': 2, 'execution_times': [1.0, 2.0]}}
        result = _compare_framework_results(framework_data, logger)
        perf = result['performance_comparison']['pymdp']
        assert abs(perf['mean'] - 1.5) < 1e-06
        assert perf['min'] == 1.0
        assert perf['max'] == 2.0

    def test_visualize_simulation_results_no_details(self, tmp_path):
        """visualize_simulation_results returns list (empty) when no details."""
        from analysis.analyzer import visualize_simulation_results
        result = visualize_simulation_results({'execution_details': []}, tmp_path)
        assert isinstance(result, list)