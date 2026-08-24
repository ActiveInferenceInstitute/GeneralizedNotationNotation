# Doc-Integrity — GNN swarm-2 scope

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation
YOU OWN these paths ONLY (disjoint scope — no other agent touches them):
- doc/troubleshooting/  (add pages only under this tree)
- doc/quickstart.md, doc/START_HERE.md, doc/INDEX.md, doc/CROSS_REFERENCE_INDEX.md
- dxd: focus on fixing broken cross-references and filling real coverage gaps

DO NOT TOUCH anything outside this scope. In particular NEVER edit:
- pyproject.toml, justfile, uv.lock, pytest.ini, .gitignore, AGENTS.md,
  CLAUDE.md, README.md, CHANGELOG.md, src/main.py
- src/tests/conftest.py, src/tests/helpers/, src/tests/categories.py
- files owned by other agents (disjoint paths).

GOAL
The repo has strict doc-audit gates that already pass, so do NOT manufacture
changes. Instead, find and fix REAL documentation gaps/errors:
1. Run the strict audit to confirm the baseline is clean
   (docs_audit.py --strict --check-anchors --no-write).
2. Look for genuinely stale or missing cross-file references in the docs you
   own; only fix ones that are actually wrong (verify the target path/heading
   exists).
3. Add ONE genuinely useful troubleshooting or quickstart page ONLY if there is
   a real gap (e.g. the export-module duplication, or cross-framework run
   guidance) — do not pad.
Verify the strict docs audit stays green after your changes.

Drive shallow/tactical through deep/strategic improvements. Do NOT pad — every
change must earn its keep. Fix root causes, not symptoms. Add targeted
regression tests for any behaviour you pin or bug you fix. Keep public API
stable unless clearly justified.

VERIFY (scoped only — do NOT run the full suite):
- uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write
- uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict
- uv run --extra dev python scripts/check_repo_terminology.py --strict

HARD RULE: DO NOT commit, DO NOT push, DO NOT stage. Leave ALL changes
uncommitted. Do not touch git index/HEAD. Other agents work on disjoint paths;
ignore files you do not own even if they look unfinished.

## Finish
Write a concise report to {report}
Summarize exactly what you changed (files + nature), tests added/passing,
and scoped verification results. Reply with only the absolute path to your report.

