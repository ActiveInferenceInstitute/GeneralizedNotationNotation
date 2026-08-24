# Core-GNN-EdgeModules — GNN swarm-2 scope

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation
YOU OWN these paths ONLY (disjoint scope — no other agent touches them):
- src/gnn/watcher.py  (GNNWatcher — CLI watch mode)
- src/gnn/dep_graph.py  (render_graph_from_file — CLI + website dashboard)
- src/gnn/parse_cache.py
- src/gnn/multimodel.py
- mirror tests: src/tests/gnn/  (add new test files; do not overwrite existing)

DO NOT TOUCH anything outside this scope. In particular NEVER edit:
- pyproject.toml, justfile, uv.lock, pytest.ini, .gitignore, AGENTS.md,
  CLAUDE.md, README.md, CHANGELOG.md, src/main.py
- src/tests/conftest.py, src/tests/helpers/, src/tests/categories.py
- files owned by other agents (disjoint paths).

GOAL
These four reachable GNN modules have ZERO test references. Add real test
coverage that exercises them against real inputs:
1. watcher.GNNWatcher — verify start/stop/scan lifecycle, watch_dir discovery,
   and that it detects new/changed GNN files; make it safe to run/cancel.
2. dep_graph.render_graph_from_file — verify it builds a dependency graph from a
   real GNN file and returns a usable artifact (and handles missing/malformed
   input without crashing).
3. parse_cache — verify caching behaviour (hit/miss, invalidation, thread-safety
   if relevant).
4. multimodel — verify its multi-model composition/parsing behaviour.
Pin behaviour with regression tests; fix any real bugs found. Do NOT delete these
modules; they are reachable from the CLI and website surfaces.

Drive shallow/tactical through deep/strategic improvements. Do NOT pad — every
change must earn its keep. Fix root causes, not symptoms. Add targeted
regression tests for any behaviour you pin or bug you fix. Keep public API
stable unless clearly justified.

VERIFY (scoped only — do NOT run the full suite):
- uv run ruff check src/gnn/watcher.py src/gnn/dep_graph.py src/gnn/parse_cache.py src/gnn/multimodel.py
- uv run ruff format --check (same)
- uv run pytest src/tests/gnn -q --tb=no -x
- uv run mypy (same files) --config-file pyproject.toml

HARD RULE: DO NOT commit, DO NOT push, DO NOT stage. Leave ALL changes
uncommitted. Do not touch git index/HEAD. Other agents work on disjoint paths;
ignore files you do not own even if they look unfinished.

## Finish
Write a concise report to {report}
Summarize exactly what you changed (files + nature), tests added/passing,
and scoped verification results. Reply with only the absolute path to your report.

