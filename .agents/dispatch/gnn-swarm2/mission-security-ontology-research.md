# Security-Ontology-Research — GNN swarm-2 scope

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation
YOU OWN these paths ONLY (disjoint scope — no other agent touches them):
- src/security/  (threat policies, sanitization, access control)
- src/ontology/  (Active Inference term handling)
- src/research/  (research tools)
- mirror tests: src/tests/security/, src/tests/ontology/, src/tests/research/

DO NOT TOUCH anything outside this scope. In particular NEVER edit:
- pyproject.toml, justfile, uv.lock, pytest.ini, .gitignore, AGENTS.md,
  CLAUDE.md, README.md, CHANGELOG.md, src/main.py
- src/tests/conftest.py, src/tests/helpers/, src/tests/categories.py
- files owned by other agents (disjoint paths).

GOAL
Deepen coverage and harden these smaller modules:
1. Security: exercise threat-policy application (basic/standard/strict),
   sanitization, file-inspection determinism; verify fail-closed receipts and
   subprocess-call detection; add regression tests. Fix any real bug found.
2. Ontology: exercise term handling / reasoning paths; verify strict_validation
   and determinism on real inputs.
3. Research: exercise the parsing/hypothesis-marking paths.
Fix root causes; add targeted tests. Do NOT degrade security guarantees.

Drive shallow/tactical through deep/strategic improvements. Do NOT pad — every
change must earn its keep. Fix root causes, not symptoms. Add targeted
regression tests for any behaviour you pin or bug you fix. Keep public API
stable unless clearly justified.

VERIFY (scoped only — do NOT run the full suite):
- uv run ruff check src/security src/ontology src/research
- uv run pytest src/tests/security src/tests/ontology src/tests/research -q --tb=no -x
- uv run mypy src/security src/ontology src/research --config-file pyproject.toml

HARD RULE: DO NOT commit, DO NOT push, DO NOT stage. Leave ALL changes
uncommitted. Do not touch git index/HEAD. Other agents work on disjoint paths;
ignore files you do not own even if they look unfinished.

## Finish
Write a concise report to {report}
Summarize exactly what you changed (files + nature), tests added/passing,
and scoped verification results. Reply with only the absolute path to your report.

