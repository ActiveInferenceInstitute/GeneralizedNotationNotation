# Render & Execute Scope — mission-render-execute.md

You own one of these paths ONLY within the GNN repo at
`/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation`:

- src/gnn/render/   (code generation for PyMDP, RxInfer.jl, JAX, NumPyro, Stan,
  PyTorch, ActiveInference.jl, DisCoPy, bnlearn + stigmergic multi-agent)
- src/gnn/execute/  (simulation / execution of rendered .py and .jl scripts)
- mirror tests: tests/render/, tests/execute/

DO NOT TOUCH anything outside that scope (root config files, conftest,
macros uagents/, utils/ etc — see the shared "do NOT touch" rule in mission-parse.md).

GOAL: Deep, strategic improvement of the render→execute→log lifecycle.
1. Render correctness: verify generated framework code actually runs. Fix
   real template/render bugs (incomplete signatures, missing imports,
   wrong data shapes for each backend). Do NOT invent frameworks.
2. RxInfer.jl / ActiveInference.jl execution: validate the committed
   `Project.toml`-based execution path pins (RxInfer 5.5.0). Ensure
   execution produces genuine model/inference results, not stubs.
3. Stigmergic multi-agent: the TO-DO.md lists an open residual —
   "env-conditioned action selection for stigmergic swarms" (infer
   `env_signal` as a latent from observations and condition per-agent
   action selection on it). If you can complete this cleanly with tests
   pinned by `tests/render/test_stigmergic_multi_agent.py`, do it.
   Otherwise scope it honestly in your report. Do NOT claim completion
   without a live test.
4. Execution failure/skip/retry reporting: make best-effort paths total
   and failures explicit.
5. Remove mypy strict smells (union returns, loose `Any`) where cleaning
   the type profile is genuinely better.

CONSTRAINTS: public API stable; no artificial tests; every change earn
its keep. Do NOT run Julia/RxInfer heavyweight live runs unless you
already can quickly; prefer the focused pytest that pins the render path.

VERIFY (scoped only):
- `python -m ruff check src/gnn/render src/gnn/execute`
- `python -m ruff format --check src/gnn/render src/gnn/execute`
- `uv run pytest tests/render tests/execute -q --tb=no -x`
- `uv run mypy src/gnn/render src/gnn/execute --config-file pyproject.toml`

HARD RULE: leave ALL changes uncommitted. No commit/push/stage. Other
agents own disjoint paths; if you see new files you don't own, leave them.

## Finish
Write a concise report to
`/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-improve/REPORT-render-execute.md`
files changed, what you fixed/completed (incl. whether the stigmergic
residual is closed and how it's tested), scoped verification results.
Reply with only the absolute path to your report.