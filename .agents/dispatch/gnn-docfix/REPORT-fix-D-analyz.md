# REPORT — D-analyz: GNN documentation accuracy fixes (DOC-ONLY)

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation
Type: Documentation-only edits (no source code, tests, config, or dependencies touched).
Commits: none (changes left uncommitted, per HARD RULE).

## Files changed and edits applied

### 1. doc/gnn/modules/16_analysis.md
- **Fix 1 (signature logger -> verbose)**: API reference signature changed to
  `process_analysis(target_dir, output_dir, verbose=False, **kwargs)`; parameter list
  updated (`verbose: bool, optional` replacing `logger`); both usage examples updated
  (`logger=logger` -> `verbose=True`).
- **Fix 2 (return keys)**: Documented return keys corrected from
  `variable_count / connection_count / type_distribution / dimension_statistics / density_metrics`
  to `variable_statistics / connection_statistics / section_statistics / distributions / correlations`.
- **Fix 3 (usage example)**: `perform_statistical_analysis` example now calls
  `perform_statistical_analysis(Path(".../some_model.gnn"), verbose=True)` and reads
  `stats['variable_statistics']['count']` and `stats['connection_statistics']['count']`
  (dropped fabricated `['connection_statistics']['density']`).
- **Fix 4 (env vars / config not consumed)**: `ANALYSIS_PERFORMANCE_MODE`, `ANALYSIS_TIMEOUT`,
  and `analysis_config.yaml` annotated as *reserved/illustrative — not consumed/tracked by
  the current implementation*.
- **Fix 5 (DEFAULT_COMPLEXITY_THRESHOLDS)**: Added a note that this is not a defined constant;
  thresholds are computed inline in `calculate_complexity_metrics`; block marked illustrative.
- **Fix 6 (output products)**: Replaced fabricated artifact names with real ones:
  `{model}_statistical_analysis.json` -> `analysis_results.json`,
  `{model}_complexity_metrics.json` -> `{model}_post_simulation_analysis.json`,
  `{model}_performance_benchmarks.json` -> `analysis_summary.md`,
  `{model}_analysis_summary.md` -> `cross_model_comparison_report.md`.
  Output-directory-structure block updated to match.

### 2. doc/gnn/modules/08_visualization.md
- **Fix 7 (test glob)**: Measurement command updated from `src/tests/test_visualization_*.py`
  to `src/tests/visualization/`.

### 3. src/visualization/AGENTS.md
- **Fix 7 (test glob)**: Same correction as above (`src/tests/test_visualization_*.py` ->
  `src/tests/visualization/`), applied to the module AGENTS.md.

### 4. doc/gnn/modules/09_advanced_viz.md
- **Fix 8 (MCP tools)**: Documented registered tools replaced with the real ones:
  `process_advanced_visualization`, `check_visualization_capabilities`,
  `list_d2_visualization_types`, `get_advanced_visualization_module_info`
  (removed fabricated `generate_3d` / `create_dashboard` / `generate_d2` / `analyze_statistics`).
  Tool-endpoint code block rewritten for `process_advanced_visualization_mcp`.
- **Fix 9 (3d product is PNG)**: Output product `{model}_3d_visualization.html` ->
  `{model}_3d_visualization.png` (interactive form noted as `{model}_dashboard.html`);
  matching directory-structure line updated.
- **Fix 10 (visualization_data.json not written)**: Removed the fabricated
  `{model}_visualization_data.json` product and its directory-structure entry.

## Verification
- `uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write`
  -> green (Broken links: 0, Bad anchors: 0, all coverage 0).
- `uv run --extra dev python scripts/check_repo_terminology.py --strict` -> "maintained tree clean."
- `uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict` -> "no banned patterns in doc/ and src/gnn/".

No banned words (legacy/stub/placeholder/deprecated) were introduced. Changes are uncommitted, as required.