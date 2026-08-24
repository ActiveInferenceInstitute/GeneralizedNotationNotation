# REPORT — Core-GNN-EdgeModules (gnn-swarm-2 scope)

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation

## Summary
Added real, regression-pinning test coverage for the four reachable GNN modules
that previously had zero test references (watcher, dep_graph, parse_cache,
multimodel). Made **two modest production changes** to harden behavior discovered
while writing the tests, and pinned the rest of the current behavior as-is.
All changes are **uncommitted** per the hard rule (nothing staged, nothing pushed,
git index/HEAD untouched).

## Production changes (files + nature)
1. **src/gnn/watcher.py** — `_debounced_fire`: a misbehaving `on_change` callback
   previously propagated its exception out of the polling/watchdog loop and would
   silently kill the watcher thread. The read and the callback are now separated,
   and a callback failure is caught and logged (`Watcher callback failed for …`)
   so the watch loop keeps running. Public API unchanged.
2. **src/gnn/dep_graph.py** — `render_graph_from_file`: a missing non-regular file
   previously raised an unhandled `FileNotFoundError` even though both callers
   (CLI `_cmd_graph` and the website dashboard) treat the renderer as a
   best-effort display helper. It now logs a warning and renders a valid empty
   graph for missing/unreadable input instead of crashing. Public API unchanged
   (still returns `str`).
3. **src/gnn/parse_cache.py** — added a `threading.Lock` guarding the shared hit/
   miss counters, cache-file reads/writes, and invalidation, so concurrent access
   cannot lose counter updates or interleave partial JSON. Public API unchanged.

No change to multimodel.py: the only suspicious behavior (two-char `->` arrows
producing GNN-E005 "Unparseable connection") is **not a bug** — the canonical GNN
grammar (doc/gnn/gnn_syntax.md, about_gnn.md) specifies single-char `source>target`
or `source-target` with no spaces, and the parser correctly rejects `->`. Since
this lives in out-of-scope `src/gnn/schema.py`, it was verified, not changed.

## Tests added (new files under src/tests/gnn/, none overwritten)
- **test_gnn_watcher.py** (8) — default validation callback on valid/invalid GNN
  (real gnn.schema path), debounced firing, callback-receives-content,
  misbehaving-callback-does-not-raise (pins fix 1), and end-to-end polling
  lifecycle: new-file discovery, extension filtering (.md only, .txt ignored),
  clean start→stop.
- **test_gnn_dep_graph.py** (11) — build_dependency_graph nodes/edges,
  shared-variable edge inference, empty/no-shared cases, mermaid + adjacency
  rendering, and render_graph_from_file: mermaid/text/default formats from a real
  two-model GNN file, plus missing/malformed/empty input without crashing
  (pins fix 2).
- **test_gnn_parse_cache.py** (10) — hit/miss semantics, content-change and
  per-file keying, hit-ratio stats, per-file and whole-cache invalidation, write-
  after-clear, and a deterministic concurrent hit/miss atomicity test (pins fix 3:
  hits==misses==N under 8 threads, no lost counter updates).
- **test_gnn_multimodel.py** (10) — split_models single/multi/front-matter-stripped/
  empty/blank-filtered, and parse_multimodel single/multi result shape, indices,
  empty and malformed input without crashing, serializable variable dicts.

Total: **39 new tests**, all passing.

## Scoped verification results (exact charter commands)
- `uv run ruff check` (4 owned files): ✅ All checks passed.
- `uv run ruff format --check` (4 owned files): ✅ 4 files already formatted.
- `uv run --extra dev python -m pytest src/tests/gnn -q --tb=no -x`: ✅ **358 passed** (319 pre-existing + 39 new), 0 failed.
- `uv run mypy` (4 owned files) `--config-file pyproject.toml`: ✅ Success, no issues.

## Notes
- Other agents' in-progress changes (ontology, security, type_checker, gui,
  doc/troubleshooting, security tests) were left untouched, as were the pre-existing
  untracked probe files. Only the three source files + four new test files in scope
  were modified/added.
- Nothing committed or staged.
