# TypeChecker-Validation — GNN swarm-2 scope

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation
YOU OWN these paths ONLY (disjoint scope — no other agent touches them):
- src/type_checker/  (type checking + resource estimation)
- src/validation/  (advanced validation + consistency checking)
- mirror tests: src/tests/type_checker/, src/tests/validation/

DO NOT TOUCH anything outside this scope. In particular NEVER edit:
- pyproject.toml, justfile, uv.lock, pytest.ini, .gitignore, AGENTS.md,
  CLAUDE.md, README.md, CHANGELOG.md, src/main.py
- src/tests/conftest.py, src/tests/helpers/, src/tests/categories.py
- files owned by other agents (disjoint paths).

GOAL
Deepen coverage and harden these consistency modules:
1. Resource estimation: exercise estimate_file_resources / type-checking on
   real GNN files and degenerate inputs; fix crashes on empty/malformed specs.
2. Validation: exercise cross-reference and consistency checks; make best-effort
   paths total (no swallowed errors), add regression tests for invariants.
3. Remove mypy strict smells (union returns, `x | None` + raise, loose Any)
   where a cleaner type contract is genuinely better.
Fix root causes; add targeted tests for pinned behaviour.

Drive shallow/tactical through deep/strategic improvements. Do NOT pad — every
change must earn its keep. Fix root causes, not symptoms. Add targeted
regression tests for any behaviour you pin or bug you fix. Keep public API
stable unless clearly justified.

VERIFY (scoped only — do NOT run the full suite):
- uv run ruff check src/type_checker src/validation
- uv run pytest src/tests/type_checker src/tests/validation -q --tb=no -x
- uv run mypy src/type_checker src/validation --config-file pyproject.toml

HARD RULE: DO NOT commit, DO NOT push, DO NOT stage. Leave ALL changes
uncommitted. Do not touch git index/HEAD. Other agents work on disjoint paths;
ignore files you do not own even if they look unfinished.

## Finish
Write a concise report to {report}
Summarize exactly what you changed (files + nature), tests added/passing,
and scoped verification results. Reply with only the absolute path to your report.

