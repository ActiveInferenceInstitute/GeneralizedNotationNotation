# Specification: scripts/

## Design Requirements
This module contains decoupled development utilities that execute asynchronously to the core pipeline runtime. It includes local file system inspection tools, maintenance integrations, pre-commit logic, and top-level thin orchestrators that configure and trigger the core pipeline for specific scientific studies.

## Components
Expected available types: Standalone `__main__` entry-pointed Python modules leveraging local parsing strategies (i.e. regex logic, abstract syntax trees) or `subprocess` orchestrators for complex analytical sweeps.

Core tools include:
1. `check_gnn_doc_patterns.py`: Pattern matching validation engine for localized GNN standards.
2. `run_pymdp_gnn_scaling_analysis.py`: End-to-end thin orchestrator executing parameterized PyMDP scaling studies through Steps 3, 11, 12, and 17 of `src/main.py`.

## Technical Rules
- Dependencies referenced across tools must rely natively on the unified project lockfile environment and the standard library, requiring no unique `requirements.txt` scope isolation.
- Scripts must default to explicit logging out.
- Scaling runs must resolve relative paths from the repository root so invocation from `scripts/` and the repository root produces the same target locations.
- Scaling runs must keep generated GNN specs separate from pipeline artifacts: specs default to `input/gnn_files/pymdp_scaling_study`, while pipeline outputs default to `output/pymdp_scaling_pipeline`.
- Scaling runs must pass `--frameworks` to render and execute. The default is `pymdp`; `all` is an explicit opt-in for every backend.
- Scaling runs must pass `--render-output-dir <pipeline_output_dir>/11_render_output` to Step 12 so execution reads the current run's renders, not stale shared output.
- Scaling runs must pass `--execution-workers` to Step 12. The default is `1`; higher values parallelize rendered scripts, not timesteps inside one simulation.
- Scaling runs may pass `--distributed --backend ray|dask` to Step 12 when cluster dispatch is intended. Local process workers remain the default.
- Scaling runs should use strict render success by default. `strict_framework_success` forwards `--strict-framework-success` to Step 11, requiring every requested framework render to succeed.
- Scaling runs must treat Step 17 integration as dependent on complete execution by default. For `pipeline_steps: 3,11,12,17`, run `3,11,12` first and run `17` only if that phase succeeds.
- Scaling runs may opt into partial integration with `run_integration_on_failure: true`, but this mode must be explicit because it can create reports from incomplete Step 12 outputs.
- Scaling runs must write `pymdp_scaling_run_manifest.json` under `pipeline_output_dir` with effective configuration, resolved paths, planned grid cells, skipped cells, and phase statuses.
- Resource gate failures must provide machine-readable details. Prefer `scripts/pymdp_scaling_last_resource_gate.json`; if the disk is full, print compact one-line `resource_gate` JSON to stderr.
