# REPORT — Analysis & Visualization Scope (mission-analysis-viz)

## Scope
Work confined to `src/analysis/`, `src/visualization/`, `src/advanced_visualization/`
and their mirror tests under `src/tests/{analysis,visualization,advanced_visualization}/`.
Nothing outside this scope was modified. All changes are left **uncommitted** (no
commit / push / stage), per the mission HARD RULE.

## Bugs fixed

### 1. `src/analysis/math_utils.py` — metric helpers made total on degenerate input (goals 1 & 2)
Reproduced two real crashes and a NaN-propagation hazard before fixing:

| Input | Before | After |
|---|---|---|
| `compute_expected_free_energy` with empty beliefs | **`ValueError`** (numpy matmul: size-0 vs 2 mismatch) | `0.0` |
| `analyze_active_inference_metrics` with a flat belief list | **`IndexError`** (`tuple index out of range` on `beliefs_array.shape[1]`) | graceful, certainty computed |
| `compute_shannon_entropy` / `compute_kl_divergence` with empty or NaN input | returned `-0.0`/max-entropy/`NaN` (silent garbage) | `0.0`, NaN never propagates |

Concretely changed:
- `compute_shannon_entropy`: returns `0.0` for empty input; neutralises non-finite
  entries and guards a zero total so no NaN is produced.
- `compute_kl_divergence`: returns `0.0` for empty or zero-sum p/q; NaN-safe.
- `compute_variational_free_energy`: returns `0.0` on empty beliefs (guards the
  `np.ones_like(q_s)/len(q_s)` division).
- `compute_expected_free_energy`: returns `0.0` on empty beliefs or any empty
  A/B/C matrix, eliminating the matmul dimension-mismatch crash.
- `analyze_active_inference_metrics`: belief-certainty normalisation now guards
  `beliefs_array.ndim < 2`, so a single flat belief vector no longer raises.

### 2. `jax_kronecker_factorized_v1` extraction path (goal 4)
Confirmed the dispatch (`extract_jax_data` → `extract_jax_kronecker_data`) is wired
top-level, nested-`simulation_data`, and impl-dir. Added first regression coverage
pinning that a `jax_kronecker_factorized_v1` payload (top-level, nested, and from an
impl-dir `simulation_data/*.json`) is routed to the per-factor extractor and yields
the summed per-step total EFE, per-factor beliefs, and `model_parameters`
(`joint_state_space_size`) — and that a non-kronecker JAX result still falls back to
the pymdp-compatible path.

### 3. Visual robustness / backend output (goals 2 & 3)
Audited the scoped viz code for division-by-empty and NaN sites. Empty-variable
statistical panels, single/malformed-matrix correlation (guarded by
`np.errstate` + `np.nan_to_num`), network stats with zero nodes, and real Agg-backend
PNG output are already covered by the existing suite; no additional source changes
were needed. Verified they all pass.

## Tests added
- **`src/tests/analysis/test_math_utils_edge.py`** (new, 15 tests): empty / zero-sum /
  NaN / uniform / certain inputs for `compute_shannon_entropy`, `compute_kl_divergence`,
  `compute_variational_free_energy`, `compute_expected_free_energy`,
  `analyze_active_inference_metrics`; plus an empty-trajectory early-return check.
- **`src/tests/analysis/test_analysis_post_simulation.py`** (extended, 4 tests):
  `TestExtractJaxData` kronecker dispatch — top-level payload, nested `simulation_data`,
  impl-dir JSON discovery, and pymdp-compatible fallback.

## Files changed
- `src/analysis/math_utils.py` (modified — 60 insertions / 16 deletions)
- `src/tests/analysis/test_math_utils_edge.py` (new — 15 tests)
- `src/tests/analysis/test_analysis_post_simulation.py` (modified — 4 new tests)

## Scoped verification (mission-required)
- `uv run ruff check src/analysis src/visualization src/advanced_visualization` → **All checks passed!**
- `uv run ruff format --check src/analysis src/visualization src/advanced_visualization` → **79 files already formatted**
- `uv run pytest src/tests/analysis src/tests/visualization src/tests/advanced_visualization -q --tb=no -x` → **403 passed, 0 failed** (baseline 384 + 19 new)
- `uv run mypy src/analysis src/visualization src/advanced_visualization --config-file pyproject.toml` → **Success: no issues in 79 source files**

Working tree left uncommitted and unstaged; changes confined to the scoped paths.