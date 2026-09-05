# visualization-worker REPORT — fleet 3, 2026-09-04

Scope: `src/visualization/` (entire module incl. AGENTS.md/README/SPEC/SKILL) + `src/8_visualization.py`.
Verification: ruff clean · mypy clean (39 files) · pytest 179 passed / 2 pre-existing skips.

## Files changed + why

### New files (8)
| File | Why |
|---|---|
| `core/sampling.py` | Pure `sample_parsed_data()` (+ `SamplingSummary` TypedDict, `VARIABLE/MATRIX_SAMPLE_LIMIT`) — extracted from the `process_single_gnn_file` monolith; testable with no plotting deps |
| `graph/stats.py` | Pure `compute_connection_statistics()` — moved the degree-stats impl out of `__init__.py`; pinned at package root as `_generate_network_statistics` (test-visible name preserved) |
| `backends.py` | `backend_status()` — one-call matplotlib/numpy/seaborn/networkx/plotly availability report (the AGENTS troubleshooting first step) |
| `src/tests/visualization/test_visualization_sampling.py` | 7 tests: no-op below limit, truncation+connection filtering, matrix cap, custom limits |
| `src/tests/visualization/test_visualization_matrix_collect.py` | 8 tests: parameters→variables→raw-matrices cascade, default names, non-numeric skips |
| `src/tests/visualization/test_visualization_stats.py` | 4 tests incl. package-root alias identity |
| `src/tests/visualization/test_visualization_backends.py` | 5 tests incl. theme-SSOT regression (every `_determine_connection_type` output has a themed style) |
| `src/tests/visualization/test_visualization_pkg_api.py` | 4 tests: README-table exports resolve, new helpers exported, `__all__` resolves, injected-logger honored |

### Modified files (19)
| File | Change |
|---|---|
| `core/process.py` | Decomposed monolith into `load_cached_artifacts` / `render_matrix_artifacts` / `write_viz_manifest` / `write_sampling_note`; uses pure sampling+collection; `process_visualization(..., *, logger=None)` DI (pipeline logger was previously silently discarded via `**kwargs`); replaced inline `np.array` with `convert_to_matrix` path; behavior contracts (return codes, summary JSON, manifest) unchanged |
| `matrix/extract.py` | Added `collect_visualization_matrices()` pure function |
| `matrix/visualizer.py` | Import hygiene: removed dead `import ast`, dead `from matplotlib import cm`, mid-file duplicate imports; `_safe_figsize -> Tuple[float, float]` |
| `visualizer.py` | Removed ~230 lines of provably-dead private methods (`_process_state_space_and_visualize`, `_process_connections_and_visualize`, `_create_basic_text_visualization`, `_save_model_metadata`, `_visualize_state_space`, `_visualize_combined`, `_extract_model_name` — zero callers repo-wide, grep-verified incl. tests); removed dead `ast`/`cm`/`np` imports, unused `json`; remaining 5 `print()` sites → `logger` (no test or output contract depends on them) |
| `graph/network_visualizations.py` | Node colors now from `theme.VAR_TYPE_COLORS`, edge styles from `theme.get_edge_style` (deleted 70-line private copy); deleted dead `edge_attrs` (only `connection_type` was ever read); replaced O(variables×edges) per-edge type rescan with a prebuilt `var_types` map; renamed `_generate_network_statistics` → `_compute_graph_metrics` (kills the same-name collision with the package-root helper); prints → `logger.warning`; compat/theme imports dedupe |
| `graph/bipartite.py` | Guarded matplotlib import replaced by `plotting.utils` import (dedupe) |
| `analysis/combined_analysis.py` | `_viz_var_type` → direct `viz_var_type` import; `GENERATIVE_MODEL_COLORS` from theme (deleted hex-duplicate dict); shared `_count_var_types` for both pie charts; nested per-exception `count_elements` def → module-level `_recursive_element_count`; prints → `logger.warning`; compat import dedupe |
| `ontology/visualizer.py` | `encoding="utf-8"` on file open; precise list/tuple typing |
| `mcp.py` | `get_visualization_options_mcp` no longer returns fabricated options with `success: True` on error — now `success: False` + `logger.error` (silent-fallback fix; unreachable in practice, static dict) |
| `__init__.py` | Full documented public API re-exported at package root (12 additions incl. `load_visualization_model`, `GNNParser`, `generate_network_visualizations`, `generate_combined_analysis`, pure helpers, `backend_status`); removed unused `np`/`Path`/`Optional`/`Union` imports; stats impl moved to `graph/stats.py` with alias |
| `processor.py` | Re-exports `generate_combined_analysis`/`generate_combined_visualizations` (imported after `.core.process` to respect the existing import-order-sensitive cycle) |
| AGENTS.md ×5, README.md, SPEC.md, SKILL.md | New API documented, theme-SSOT note, dated "Documentation update (2026-09-04)" changelog entry under Current Version 3.2.0 (pipeline release numbers left to CHANGELOG), layout/structure updates, test-file list |

`8_visualization.py`: untouched — already a correct thin orchestrator (58 lines).

## API deltas (all additive; breaking changes: none)
- New: `backend_status`, `sample_parsed_data`, `collect_visualization_matrices`, `compute_connection_statistics` (package root + defining modules).
- `process_visualization` gains keyword-only `logger=None`; positional contract unchanged; default logging behavior unchanged when not injected.
- Package root `__all__` grew from 10 to 22 entries; `visualization._generate_network_statistics` kept as the same callable (pinned test).
- Internal renames: `graph.network_visualizations._generate_network_statistics` → `_compute_graph_metrics` (private, no external callers).
- Behavioral notes (rendering): `state_observation`, `state_transition_matrix`, `policy_action` edges now use themed styles instead of gray fallback; network-graph legend gains a `free_energy` row (palette is now the shared theme). Stats JSON, manifest, summary, exit codes unchanged.
- Degenerate-input deltas in `collect_visualization_matrices`: raw `matrices` entries with size-0 data (e.g. `[]`) are now skipped instead of stored as empty arrays — strictly safer downstream (an empty heatmap cannot render); empty-string `name` fallback behavior matches the original loop exactly.

## Verification output tails (final run, after advisory-response fixes)
```
ruff check src/visualization src/tests/visualization  → All checks passed! (1 I001 auto-fixed post-conversions)
mypy src/visualization --config-file pyproject.toml   → Success: no issues found in 39 source files
pytest src/tests/visualization/ -q                    → 179 passed, 2 skipped in 122.68s
git status --porcelain (scope)                        → 19 M + 8 ?? (all in scope)
```

## Follow-ups for other workers (not my scope)
- `doc/development/docs_audit.py --strict` fails on pre-existing `src/tests/tests` (has `.py`, no AGENTS.md) — untouched by me, lives in the src/tests worker's scope.
- `doc/gnn/integration/gnn_visualization.md` and `doc/gnn/modules/08_visualization.md` document `visualization.processor` imports that still work; if doc workers touch them, the new package-root exports (`backend_status` etc.) are worth adding.
- `advanced_visualization/_shared.py` + `network_viz.py` still lazily import `MatrixVisualizer` via the `matrix_visualizer` facade — works unchanged; they could import from `visualization` root for consistency.
- `src/tests/tests/` docs gap noted above; `analysis/combined_analysis.py` has two matrix-size loops with intentionally different fallback semantics (recursive count vs skip) — left unmerged deliberately.

## Follow-up ideas (visualization module)
- Unify the two `_parse_matrix_string` implementations (`visualizer.py` via `safe_literal_eval` vs `matrix/visualizer.py` regex floats) behind an explicit strategy parameter — different semantics, needs a behavioral decision first.
- `FEATURES["interactive_plots"]` is statically `True` while plotly is optional; consider deriving from `backend_status()`.
- MatrixVisualizer `create_heatmap()` writes to a hardcoded `cwd/output/2_tests_output` path — candidate for deprecation or output_dir parameter.
