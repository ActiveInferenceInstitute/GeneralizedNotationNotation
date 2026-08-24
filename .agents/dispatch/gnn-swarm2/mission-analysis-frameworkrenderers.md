# Analysis-FrameworkRenderers — GNN swarm-2 scope

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation
YOU OWN these paths ONLY (disjoint scope — no other agent touches them):
- src/analysis/numpyro/  (analyzer + init)
- src/analysis/pytorch/  (analyzer + init)
- src/analysis/generate_cross_model_report.py
- mirror tests: src/tests/analysis/  (add/extend test files only)

DO NOT TOUCH anything outside this scope. In particular NEVER edit:
- pyproject.toml, justfile, uv.lock, pytest.ini, .gitignore, AGENTS.md,
  CLAUDE.md, README.md, CHANGELOG.md, src/main.py
- src/tests/conftest.py, src/tests/helpers/, src/tests/categories.py
- files owned by other agents (disjoint paths).

GOAL
These analysis sub-packages (numpyro/pytorch analyzers, each ~109 LOC) and
generate_cross_model_report have poor/zero direct coverage. Add tests that:
1. Exercise the numpyro and pytorch analyzers end-to-end on realistic inputs
   (or verify their documented graceful-degradation when the backend is absent),
   asserting they produce the documented result structure.
2. Cover generate_cross_model_report — verify it produces a valid output report
   from sample analysis data and handles empty/missing inputs.
Fix any real bugs found (e.g. crashes on empty input). Do NOT force-backend-import
heavy packages; test the graceful-degraded and synth paths.

Drive shallow/tactical through deep/strategic improvements. Do NOT pad — every
change must earn its keep. Fix root causes, not symptoms. Add targeted
regression tests for any behaviour you pin or bug you fix. Keep public API
stable unless clearly justified.

VERIFY (scoped only — do NOT run the full suite):
- uv run ruff check src/analysis/numpyro src/analysis/pytorch src/analysis/generate_cross_model_report.py
- uv run pytest src/tests/analysis -q --tb=no -x
- uv run mypy (same paths) --config-file pyproject.toml

HARD RULE: DO NOT commit, DO NOT push, DO NOT stage. Leave ALL changes
uncommitted. Do not touch git index/HEAD. Other agents work on disjoint paths;
ignore files you do not own even if they look unfinished.

## Finish
Write a concise report to {report}
Summarize exactly what you changed (files + nature), tests added/passing,
and scoped verification results. Reply with only the absolute path to your report.

