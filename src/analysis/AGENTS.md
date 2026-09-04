# Analysis Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced statistical analysis, performance benchmarking, and complexity metrics calculation for GNN models

**Pipeline Step**: Step 16: Analysis (16_analysis.py)

**Category**: Statistical Analysis / Performance Evaluation

**Status**: Production Ready

**Version**: 3.3.0

**Last Updated**: 2026-09-04

---

## Core Functionality

### Primary Responsibilities
1. Perform comprehensive statistical analysis on GNN model structures
2. Calculate complexity metrics and maintainability indices
3. Generate performance benchmarks and comparison reports
4. Extract and analyze variable distributions and correlations
5. Provide technical debt assessment and optimization recommendations
6. **Generate ALL PyMDP visualizations** from execution raw data (moved from Execute step)

### Key Capabilities
- Statistical analysis of model variables and connections
- Complexity metrics calculation (cyclomatic, cognitive, structural)
- Performance benchmarking and profiling
- Model comparison and differential analysis
- Distribution analysis and correlation studies
- **PyMDP Visualization** - belief evolution, state sequences, performance metrics plots
- **Cross-framework comparison** - uses whatever execution (Step 12) produced. `_extract_simulation_metrics` (in `analyzer.py`) prefers `simulation_data/simulation_results.json` (and other canonical JSON) before `execution_logs/*_results.json`, so backends that write full traces to `simulation_data/` (e.g. RxInfer) are not masked by sparse structured logs. DisCoPy: inline `simulation_data.analysis` / `parameters` from structured logs populate `circuit_info`; if still missing, `simulation_data/circuit_info.json` is merged when present. bnlearn structured logs populate `model_parameters` when vector traces are absent. If every run for a framework was skipped (`skipped: true` in the execution summary), logs INFO instead of WARNING for bnlearn. Otherwise missing data is reported as "[framework] No simulation data found". Python backends are in core `uv sync`; Julia coverage needs Julia + packages installed, then re-run Step 12.
- **Kronecker-factorized JAX (MAJ-02)** - `extract_jax_data` dispatches on
  schema: `jax_kronecker_factorized_v1` payloads (top-level, nested
  `simulation_data`, or implementation-directory files) are extracted by
  `extract_jax_kronecker_data` into per-factor fields — beliefs/states/
  observations/actions per factor, per-step total EFE (sum over factors),
  factorised policy, validation, and model parameters with
  `joint_state_space_size` / `joint_materialized: False`. pymdp-compatible
  JAX payloads keep the historical path.
- **GridWorld animations** - current PyMDP, RxInfer.jl, and ActiveInference.jl
  schemas emit belief GIFs, 3x3 state trajectory GIFs, a cross-framework
  trajectory GIF, and `cross_framework/gridworld_analysis_manifest.json`.

---

## API Reference

### Public Functions

#### `process_analysis(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool | int`
**Description**: Main analysis processing function called by orchestrator (16_analysis.py). Performs statistical analysis, complexity metrics, performance benchmarks, post-simulation framework analysis, and visualizations.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to analyze
- `output_dir` (Path): Output directory for analysis results
- `verbose` (bool): Enable verbose output (default: False)
- `generate_animations` (bool, optional): Generate current-schema GridWorld GIF
  artifacts (default: True; CLI: `--no-animations` disables this). This is
  the canonical programmatic key. Compatibility callers may pass
  `no_animations`, but it is normalized as the inverse and conflicts with
  `generate_animations` are rejected.
- `**kwargs`: Additional pipeline options (unused kwargs are ignored)

**Returns**: `True` if analysis artifacts were produced, `2` when there is no
input or other warning-only recovery, and `False` for hard failures.


**Example**:
```python
from analysis import process_analysis
from pathlib import Path

success = process_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/16_analysis_output"),
    verbose=True,
)
```

#### `perform_statistical_analysis(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
**Description**: Perform comprehensive statistical analysis on a GNN file.

**Parameters**:
- `file_path` (Path): Path to the GNN file to analyze
- `verbose` (bool, optional): Enable verbose output (default: False)

**Returns**: `Dict[str, Any]` - Statistical analysis results with:
- `variable_statistics` / `connection_statistics` / `section_statistics` (Dict[str, Any])
- `distributions` and `correlations` (Dict[str, Any])
- `file_path`, `file_name`, `file_size`, `line_count`, `analysis_timestamp`

Raises `RuntimeError` if the file cannot be analyzed.

#### `calculate_complexity_metrics(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
**Description**: Calculate various complexity metrics for a GNN file.

**Parameters**:
- `file_path` (Path): Path to the GNN file to analyze
- `verbose` (bool, optional): Enable verbose output (default: False)

**Returns**: `Dict[str, Any]` - Complexity metrics with:
- `cyclomatic_complexity` (float): Cyclomatic complexity score
- `cognitive_complexity` (float): Cognitive complexity score
- `structural_complexity` (float): Structural complexity score
- `maintainability_index` (float): Maintainability index (0-100)
- `technical_debt` (float): Technical debt score

Raises `RuntimeError` if metrics cannot be computed.

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations and statistical analysis
- `pandas` - Data manipulation and analysis
- `scipy` - Advanced statistical functions

### Optional Dependencies
- `matplotlib` - Statistical visualization (recovery: text-based reports)
- `seaborn` - Enhanced statistical plots (recovery: matplotlib)

### Internal Dependencies
- `utils.pipeline_template` - Standardized pipeline processing patterns
- `pipeline.config` - Pipeline configuration management

---

## Configuration

### Environment Variables

None dedicated to this module. Analysis behavior is configured through
`process_analysis()` kwargs (e.g. `analysis_type`, `benchmark_iterations`,
`generate_animations`) and `input/config.yaml` pipeline settings.

### Default Settings

Complexity thresholds and benchmark parameters are defined in
`analysis/analyzer.py`; see the `perform_statistical_analysis` and
`calculate_complexity_metrics` functions above.

---

## Usage Examples

### Basic Usage
```python
from analysis.processor import process_analysis
from pathlib import Path

success = process_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/16_analysis_output"),
    verbose=True,
)
```

### Statistical Analysis
```python
from analysis.analyzer import perform_statistical_analysis

stats = perform_statistical_analysis(Path("models/my_model.md"))
print(f"Variable count: {len(stats['variable_statistics'])}")
```

### Complexity Assessment
```python
from analysis.analyzer import calculate_complexity_metrics

metrics = calculate_complexity_metrics(Path("models/my_model.md"))
print(f"Cyclomatic complexity: {metrics['cyclomatic_complexity']}")
print(f"Maintainability index: {metrics['maintainability_index']}")
```

---

## Output Specification

### Output Products

`process_analysis` writes to the step output directory:

- `analysis_results.json` - Full step results (statistical, complexity, benchmarks, comparisons)
- `analysis_summary.md` - Human-readable analysis report
- `cross_model_comparison_report.md` - Cross-framework comparison (when execution data exists)
- `{model}_post_simulation_analysis.json` - Per-model post-simulation analysis (in the cross-framework analysis subdirectory)
- Visualization directories (`comprehensive_visualizations/`, framework GIFs and PyMDP visualizations, `cross_framework/gridworld_analysis_manifest.json` when animations are enabled)

---

## Error Handling

- **No input**: returns exit code `2` with a warning (not a failure)
- **Missing execution summary**: logs a warning and skips post-simulation analysis
- **Malformed data**: per-file exceptions are collected in `results["errors"]`; the step continues with other files
- **Animation flag conflict**: `generate_animations` conflicting with `no_animations` aborts with `False`

## Integration Points

### Imported By
- `src/tests/analysis/` - Module-level analysis tests
- `report.generator` - Report generation uses analysis results

### Data Flow
```
GNN Files → Analysis → Statistical Reports → Model Comparisons → Optimization Recommendations
```

---

## Testing

### Test Files
- `src/tests/analysis/test_analysis_overall.py` - Module-level tests
- `src/tests/analysis/test_analysis_post_simulation.py` - Post-simulation analysis tests
- `src/tests/analysis/test_analysis_extraction.py` - Result extraction tests
- `src/tests/analysis/test_framework_common.py` - Shared framework-common helper tests
- `src/tests/analysis/test_flat_payload_analyzer.py` - Shared flat-payload analyzer engine tests

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/analysis/ \
    --cov=src/analysis --cov-report=term-missing
```

### Key Test Scenarios
1. Statistical analysis with various model sizes
2. Complexity metric calculation accuracy
3. Performance benchmarking under load
4. Error handling with malformed data

---

## MCP Integration

### Tools Registered

`analysis/mcp.py` `register_tools` registers four tools:

- `process_analysis` - Run statistical and complexity analysis on GNN files in a directory
- `get_analysis_results` - Read saved analysis JSON results from a previous run
- `compute_complexity_metrics` - Compute complexity metrics for GNN content supplied as a string
- `list_analysis_tools` - Report available analysis tools and capabilities (honest availability probe — no fake fallback)

### MCP File Location
- `src/analysis/mcp.py` - MCP tool registrations

---


## Shared Composability Helpers

### `framework_common.py`

Single source of truth for framework-name normalization, path inference, and
current-schema simulation-results discovery. Consumed by `processor.py` (and
available to all framework analyzers):

- `FRAMEWORK_DIR_NAMES` — frozenset of all 8 pipeline framework dir names
  (incl. `bnlearn`, which is rendered+executed but has no analyzer).
- `SCHEMA_GATED_FRAMEWORKS` — frozenset `{pymdp, rxinfer, activeinference_jl}`.
- `CURRENT_SIMULATION_SCHEMAS` — frozenset of `*_simulation_v1` schema strings.
- `normalize_framework_name(framework) -> str` — `"ActiveInference.jl"` → `"activeinference_jl"`.
- `model_name_from_path(path) -> str` — infers the model name from the path
  segment preceding a framework segment.
- `framework_from_path(path) -> str | None` — returns the framework dir name
  found in a path, or `None`.
- `iter_current_schema_results(execution_dir, pattern) -> list[tuple[Path, dict]]` —
  discovers current-schema `simulation_results.json` payloads; schema-gated
  frameworks must match `CURRENT_SIMULATION_SCHEMAS`, others accepted as-is.
- `resolve_execution_dir(output_dir) -> Path` — resolves the Step 12 execution
  output directory (prefers `pipeline.config`, falls back to `12_execute_output`).
- `load_execution_summary(execution_dir) -> tuple[Path, dict | None]` — prefers
  `summaries/execution_summary.json` then root; returns `None` on missing/unreadable.
- `filter_paths_by_scope(path, framework, allowed_frameworks, allowed_model_names) -> bool`.

### `flat_payload_analyzer.py`

Shared analyzer engine for PyTorch/NumPyro flat-payload simulation results.
Each framework's `analyzer.py` binds a `FlatPayloadSpec` (framework name, file
patterns, analysis filename, plot labels, bar color) and re-exports
`generate_analysis_from_logs` / `_generate_plots` — the public call sites
(processor's importlib discovery, `test_numpyro_pytorch_analyzers.py`) are
unchanged. Exports: `FlatPayloadSpec`, `compute_flat_payload_metrics` (pure),
`discover_result_files`, `generate_analysis_from_logs`.

---

## Troubleshooting

### Common Issues

#### Issue 1: Analysis fails on large models
**Symptom**: Analysis is slow or memory-heavy  
**Cause**: Model too complex for comprehensive analysis  
**Solution**: 
- Process models individually instead of batch
- Increase system memory or use sampling

#### Issue 2: Complexity metrics return zero
**Symptom**: Complexity calculations return zero or invalid values  
**Cause**: Model structure not properly extracted or missing components  
**Solution**:
- Verify GNN processing (step 3) completed successfully
- Check that model has variables and connections
- Use `--verbose` flag for detailed extraction logs

---

## Version History

### Current Version: 3.3.0 (2026-09-04)

**Features**:
- Statistical analysis
- Complexity metrics calculation
- Performance benchmarking
- Model comparison
- Framework output analysis
- `framework_common.py` — shared framework-name normalization, path inference, and current-schema simulation-results discovery (dedupes processor.py / visualizations.py copies; now includes bnlearn in the framework dir set so bnlearn results are discoverable by the analysis scope)
- `flat_payload_analyzer.py` — shared analyzer engine for PyTorch/NumPyro flat-payload simulation results (frozen `FlatPayloadSpec` + pure `compute_flat_payload_metrics` + shared discovery/plots); each framework's `analyzer.py` is now a thin spec binding
- `mcp.list_analysis_tools_mcp` honest-availability probe (fake `"available": True` fallback removed)
- `visualizations.py` matplotlib routed through `viz_base.safe_savefig` (single save/close/error path; 13 duplicated boilerplate sites consolidated)

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced visualization of analysis results
- **Future**: Real-time analysis dashboard

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Execute Module](../execute/AGENTS.md)
- [Analysis Module](../analysis/README.md)

### External Resources
- [NetworkX Documentation](https://networkx.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://scipy.org/)

---

**Last Updated**: 2026-09-04
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: 3.3.0
**Architecture Compliance**: 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
