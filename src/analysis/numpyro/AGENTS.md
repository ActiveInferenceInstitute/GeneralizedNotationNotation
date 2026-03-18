# NumPyro Analysis Backend - Agent Scaffolding

## Module Overview

**Purpose**: Analyze NumPyro execution outputs (Step 12) and generate summary metrics and plots.

**Parent**: `src/analysis/` (Step 16: Analysis)

---

## Public API

From `src/analysis/numpyro/__init__.py`:

- `generate_analysis_from_logs(results_dir: Path, output_dir: Optional[Path] = None, verbose: bool = False) -> List[str]`

---

## Inputs and outputs

**Inputs**:
- searches recursively for `simulation_results.json` under `results_dir` (notably under `**/numpyro/**/simulation_results.json`)

**Outputs**:
- writes `numpyro_analysis.json` under `output_dir/<model_name>/`
- optionally writes plots under `output_dir/<model_name>/` when `matplotlib` is available:
  - `belief_trajectory.png`
  - `action_distribution.png`
  - `efe_history.png`

