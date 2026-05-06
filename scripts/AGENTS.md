# Developer Tooling Agents (scripts/)

## Purpose

This folder hosts the explicit, stateless developer workflow validation agents responsible for maintaining continuous documentation compliance and performance benchmarking across the GNN ecosystem.

## Components

The primary agents deployed in this module are:

- `check_gnn_doc_patterns.py` (Pattern Validation Agent): Recursively scans `.md` files to eradicate obsolete architectural references to staled imports (e.g., `gnn.parser`), legacy routing, and deprecated file syntax. Operates defensively via CI implementations `--strict` enforcing non-zero process exits.
- `run_pymdp_gnn_scaling_analysis.py` (Study Orchestrator Agent v1.1.0): A thin orchestrator that programmatically generates configured GNN matrices (N×T parameter grids with controlled noise) and invokes the main `src/main.py` pipeline to conduct scientific scaling analysis. It features a **Visual Logger** for real-time progress tracking and a **preflight resource gate** to protect against O(n³) disk expansion.

## PyMDP Scaling Contract (v1.2.0)

- **Configuration**: Driven by `scripts/pymdp_scaling_config.yaml`. Supports `max_n`, `max_file_size_mb`, and `min_free_disk_mb` safety guards, plus **`matplotlib_headless`** (sets `MPLBACKEND=Agg` on pipeline subprocesses), **`gnn_serialize_preset`** (`full` | `minimal`, forwarded as `--serialize-preset`), and **`execution_benchmark_repeats`** (forwarded as `--execution-benchmark-repeats` when > 1).
- **Visual Meta-Analysis**: Generates log-log scaling curves, runtime heatmaps, and accuracy-entropy correlation plots using a high-contrast scientific theme.
- **Statistical Reporting**: Automatically fits scaling laws ($O(N^\alpha)$, $O(T^\beta)$) and calculates regression coefficients ($R^2$, $r$) for performance and quality metrics.
- **Resource Gating**: Reports volume usage, shortfall, and planned total estimated bytes. Aligned in policy terms with **Pipeline Step 5** (type checker `GNNResourceEstimator`).
- **Observability**: Generates a detailed `pymdp_scaling_run_manifest.json` recording effective configuration, resolved paths, planned grid cells, and phase statuses.
- **Concurrency**: `execution_workers` supports parallel Step 12 execution. `distributed: true` enables Ray/Dask dispatch for cluster-style workloads.
- **Composability**: Runs `3,11,12` and `17` as distinct phases to ensure integration reports are only published on successful execution.
- **Output Isolation**: Uses a dedicated `pipeline_output_dir` (default: `output/pymdp_scaling_pipeline`) to avoid polluting the shared `output/` tree.

## Operational Standards

- Strict adherence to Pythonic PEP validation and Zero-Mock principles.
- All scripts must contain structured `argparse` implementations mapped for headless CI.
- Orchestrators must implement explicit safety guardrails for resource-intensive operations.
- Automated manifest generation is required for all batch processing studies.
