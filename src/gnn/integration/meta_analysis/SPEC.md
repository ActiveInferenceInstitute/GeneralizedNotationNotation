# Meta-Analysis Module Specification

## Purpose
The `meta_analysis` module provides automated statistical analysis and visualization of GNN pipeline performance sweeps. It transforms raw simulation records into publication-quality reports and charts.

## Components

### 1. SweepDataCollector (`collector.py`)
- **Responsibility**: Harvests metadata and results from the `12_execute_output` directory.
- **Inputs**: 
    - `execution_summary.json`: Orchestrator summary (**slim aggregate** by default: timing/benchmark fields retained; full stdout/stderr/`simulation_data` live in optional `execution_summary_detail.json` when Step 12 `--execution-summary-detail` is enabled).
    - `execution_logs/*.json`: Per-model raw results.
    - `simulation_results.json`: Accuracy and entropy data.
- **Output**: A list of `SweepRecord` dataclass instances (including optional `execution_time_std`, `execution_time_mean`, `execution_benchmark_repeats`, and `execution_time_samples` when Step 12 benchmark repeats are enabled).

### 2. Validator (`validator.py`)
- **Responsibility**: Non-fatal consistency checks over collected `SweepRecord` rows.
- **Output**: Structured payload written as `sweep_validation.json` (grid coverage, timestep mismatches when simulation JSON is readable, benchmark coherence).

### 3. Statistics (`statistics.py`)
- **Responsibility**: Aggregate numeric summaries across frameworks and sweep cells.
- **Output**: `meta_statistics.json` with `schema_version`, per-framework runtime stats, best framework per (N,T), and log-log slopes for PyMDP vs N by fixed T.

### 4. SweepVisualizer (`visualizer.py`)
- **Responsibility**: Generates charts using `matplotlib`.
- **Theme**: "Scientific White" (High-contrast, publication-grade).
- **Styling (`_STYLE`)**:
    - Background: White (`#FFFFFF`) / Light Gray Axis (`#F8F9FA`).
    - Typography: Title 18pt, Labels 14pt, Ticks 12pt.
    - Line Width: 2.5pt.
    - Marker Size: 8pt.
- **Charts Generated**:
    - **Heatmaps**: Runtime, Accuracy, Entropy (with statistics in subtitle).
    - **Scaling Curves**: Log-log plots with Power-Law fits ($y = a \cdot x^b$); optional **linear** ±σ error bars per point when ``execution_time_std`` is populated from Step 12 repeats.
    - **Comparison Charts**: Cross-framework runtime and per-step cost (with global median lines).
    - **3D Surfaces**: Runtime response surfaces across N and T.
    - **Correlation Plots**: Accuracy vs. Belief Entropy scatter with linear regression ($r$ coefficient).
    - **Footprint chart**: `visualizations/data/gnn_serialization_footprint.png` from Step 3 `format_statistics.json` when present under `{pipeline_root}/3_gnn_output/format_statistics.json`.
    - **Runtime uncertainty**: `visualizations/cross_framework/comparisons/runtime_uncertainty.png` when any record has `execution_time_std > 0`.

### 5. SweepReporter (`reporter.py`)
- **Responsibility**: Generates a comprehensive Markdown report.
- **Sections**:
    - **Sweep Overview**: Grid configuration (N, T, frameworks).
    - **Runtime Comparison**: Formatted tables with automatic units (s/m).
    - **Framework Summary**: Aggregate stats (Avg/Min/Max/ms-per-step).
    - **Simulation Metrics**: Detailed Accuracy and Entropy reports.
    - **Scaling Analysis**: Power-law exponents for O(N^α) and O(T^β).
    - **Complexity Analysis**: LOC scaling reports ($O(N^3)$ warning).
    - **Validation summary**: Severity counts and link to `sweep_validation.json`.
    - **Aggregate statistics**: Short table with link to `meta_statistics.json`.
    - **Step 3 footprint**: Table of serialization totals when `format_statistics.json` was loaded.

## Statistical Methods

### Power-Law Fitting
Scaling exponents are calculated using a linear fit on log-transformed data:
$\log(y) = \log(a) + b \cdot \log(x)$
where $b$ is the reported scaling exponent ($\alpha$ or $\beta$).

### Runtime uncertainty (Step 12 benchmark repeats)
When ``--execution-benchmark-repeats`` is greater than 1, each rendered script is executed sequentially ``K`` times in the same worker. Reported ``execution_time`` is the **median** of ``K`` wall-clock samples; ``execution_time_std`` is the **population** standard deviation across those ``K`` samples ($\sqrt{\frac{1}{K}\sum (t_i - \bar{t})^2}$), zero when ``K`` is 1. Meta-analysis scaling curves plot **linear** $\pm\sigma$ error bars on runtime points when $\sigma$ is present (log-scaled axes unchanged).

### Linear Regression
Correlation plots include a linear regression line and Pearson correlation coefficient ($r$):
$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$

## Directory Structure
```
output/[run_id]/17_integration_output/integration_results/meta_analysis/
├── meta_analysis_report.md
├── sweep_validation.json
├── meta_statistics.json
└── visualizations/
    ├── data/
    │   └── gnn_serialization_footprint.png   # when Step 3 format_statistics.json loaded
    ├── cross_framework/
    │   ├── comparisons/
    │   │   └── runtime_uncertainty.png       # when benchmark σ > 0
    │   └── scaling/
    └── [framework]/
        ├── heatmaps/
        └── surfaces/
```
