# Analysis & Visualization Scope — mission-analysis-viz.md

You own these paths ONLY within the GNN repo at
`/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation`:

- src/analysis/  (statistical analysis + analyzer + visualizations)
- src/visualization/  (matrix/graph visualization)
- src/advanced_visualization/  (advanced/interactive plots)
- mirror tests: src/tests/analysis/, src/tests/visualization/,
  src/tests/advanced_visualization/

DO NOT TOUCH anything outside this scope (other modules, root config,
conftest, helpers — see mission-parse.md's no-touch rule).

GOAL: shallow→deep improvements:
1. Analyzer/metric correctness: fix any material statistical/logic bugs;
   add regression tests. Make best-effort paths total.
2. Visualization robustness: plots/matrices should render without crashes
   on edge dimensions (empty, 1x1, degenerate shapes). Guard divisions
   and NaN propagation.
3. Interactive/advanced plots: ensure they actually produce output in the
   configured backend; do not leave brittle assumptions.
4. Analysis result completeness in the numbered-pipeline (Step 16
   extraction incl. `jax_kronecker_factorized_v1`) — fix missed extraction
   paths.

VERIFY (scoped only):
- `uv run ruff check src/analysis src/visualization src/advanced_visualization`
- `uv run ruff format --check src/analysis src/visualization src/advanced_visualization`
- `uv run pytest src/tests/analysis src/tests/visualization src/tests/advanced_visualization -q --tb=no -x`
- `uv run mypy src/analysis src/visualization src/advanced_visualization --config-file pyproject.toml`

HARD RULE: leave ALL changes uncommitted; no commit/push/stage.

## Finish
Write a concise report to
`/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-improve/REPORT-analysis-viz.md`
files changed, bugs fixed, tests added, scoped verification results.
Reply with only the absolute path to your report.