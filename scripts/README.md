# Scripts Toolkit
# Version: 1.1.0

The `scripts/` directory is the repository's hub for standalone maintenance, linting, and developer acceleration utilities. These tools operate externally to the core `src/` orchestrator pipeline, actively protecting codebase integrity over time.

## Key Files
- `check_gnn_doc_patterns.py`: A strict RegEx-enforced documentation linter that audits `doc/` and `src/gnn/` against deprecated path aliases and import references.
- `run_pymdp_gnn_scaling_analysis.py`: A thin orchestrator (v1.1.0) that programmatically generates configured GNN specs and triggers the main pipeline to conduct a PyMDP scaling study. It utilizes `pymdp_scaling_config.yaml` for central configuration.

## Execution
Tools are designed to be executed via `uv run`:

```bash
# Run documentation validation
uv run python scripts/check_gnn_doc_patterns.py --strict

# Execute a PyMDP scaling sweep using current config
uv run python scripts/run_pymdp_gnn_scaling_analysis.py

# Override config values via CLI
uv run python scripts/run_pymdp_gnn_scaling_analysis.py --n-values 2,4,8,16 --t-values 10,100,500
```

## PyMDP scaling preflight
`run_pymdp_gnn_scaling_analysis.py` plans the sweep first, then applies a **preflight resource gate** (free volume space vs. policy, then estimated total spec bytes vs. free space with margin). This is critical because dense B tensors grow as **O(n³)** in file size.

That gate is conceptually aligned with **Pipeline Step 5** (type checker) storage estimation (`resource_estimator` / `estimate_storage`); after `.md` files are generated, use `uv run python src/5_type_checker.py --target-dir <dir> --estimate-resources` for full per-file resource reports.

Relative paths are resolved from the repository root. By default the orchestrator writes generated specs to `input/gnn_files/pymdp_scaling_study` and uses an isolated pipeline output directory at `output/pymdp_scaling_pipeline`.

## Configuration
The orchestrator is driven by `scripts/pymdp_scaling_config.yaml`. Key safety fields include:
- `max_n`: Skips state counts above this value to prevent exponential disk use.
- `max_file_size_mb`: Caps individual specification file size.
- `min_free_disk_mb`: Aborts if the target volume is near capacity.

## Observability
- **Visual Logging**: Uses `VisualLogger` for premium terminal output with progress bars and status icons.
- **Run Manifest**: Every run writes `output/pymdp_scaling_pipeline/pymdp_scaling_run_manifest.json` recording effective config, planned grid, skipped cells, and phase statuses.
- **Completion Banner**: Displays a detailed summary of files generated, duration, and report locations.

Expected outputs:
- Generated specs: `input/gnn_files/pymdp_scaling_study/`
- Scaling manifest: `output/pymdp_scaling_pipeline/pymdp_scaling_run_manifest.json`
- Pipeline summary: `output/pymdp_scaling_pipeline/00_pipeline_summary/pipeline_execution_summary.json`
- Integration reports: `output/pymdp_scaling_pipeline/17_integration_output/integration_results/meta_analysis/`
  - Includes **Publication-Grade Scientific Visualizations** (high-contrast white theme, log-log scaling exponents, $R^2$ metrics).
  - Includes **Analytical Meta-Analysis Report** with automated O(N^α) and O(T^β) law derivation.
