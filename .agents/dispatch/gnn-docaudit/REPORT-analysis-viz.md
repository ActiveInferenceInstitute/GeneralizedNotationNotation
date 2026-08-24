# REPORT — analysis-viz (docs-vs-code audit)

Region: doc/visualization/, doc/advanced_visualization/, doc/performance/,
doc/petri_nets/, doc/research/, doc/spm/ + Step 8/9/16 module pages and any
pages referencing src/analysis|visualization|advanced_visualization.
Mode: REPORT-ONLY. No repository files were modified.
Verified via `git ls-files`, source grep, and `src/utils/arg_parsing.py`
ARGUMENT_DEFINITIONS (the canonical main/step CLI parser).

## 1. doc/gnn/modules/16_analysis.md

- doc/gnn/modules/16_analysis.md:117 | WARNING | Documented
  `process_analysis(target_dir, output_dir, logger=None, **kwargs)` — code
  signature is `process_analysis(target_dir, output_dir, verbose=False,
  **kwargs)` (src/analysis/processor.py:278); the parameter is `verbose`, not
  `logger`. The example at :140 passes `logger=logger` (absorbed by **kwargs,
  so it still runs, but the documented signature is wrong).
  Fix: change declared param `logger` -> `verbose` (align with processor).

- doc/gnn/modules/16_analysis.md:163-176 | WARNING | Documented
  `perform_statistical_analysis(file_path, verbose)` return schema
  (variable_count, connection_count, type_distribution, dimension_statistics,
  density_metrics) does not match the actual return keys — analyzer.py:63-77
  returns `variable_statistics`, `connection_statistics`, `section_statistics`,
  `distributions`, `correlations` (no `density_metrics`, `type_distribution`,
  `dimension_statistics` top-level keys exist).
  Fix: update return-key list to the real keys.

- doc/gnn/modules/16_analysis.md:249-253 | ERROR | Usage example calls
  `perform_statistical_analysis(variables, connections)` (two positional args)
  but the function takes `(file_path: Path, verbose: bool)`, and reads
  `stats['connection_statistics']['density']` — `connection_statistics`
  (analyzer.py:222-235) has `count`, `average_line`, `line_std`, NO `density`.
  The example would TypeError (`variables` is a list, not a Path) and KeyError.
  Fix: call with a real `file_path` and read `['variable_statistics']['count']`
  / `['connection_statistics']['count']`.

- doc/gnn/modules/16_analysis.md:216-220 | WARNING | Env vars
  `ANALYSIS_PERFORMANCE_MODE` and `ANALYSIS_TIMEOUT` are documented but not
  consumed anywhere under src/ (grep of src/analysis + src/utils .py finds no
  reads, only the doc + src/analysis/AGENTS.md text). `analysis_config.yaml`
  is also listed as a config file but is not tracked (git ls-files -> nothing).
  Fix: drop or implement the env vars and config file (or mark illustrative).

- doc/gnn/modules/16_analysis.md:224-228 | INFO | `DEFAULT_COMPLEXITY_THRESHOLDS`
  block presented as a default setting but no such constant is defined in
  src/analysis (grep finds it only in doc/AGENTS text; analyzer.py computes
  complexity inline at :331/:371+).
  Fix: mirror the actual computed thresholds or label as illustrative.

- doc/gnn/modules/16_analysis.md:270-274, 284-296 | WARNING | Documented
  output products `{model}_statistical_analysis.json`,
  `{model}_complexity_metrics.json`, `{model}_performance_benchmarks.json` are
  not written by the code. processor.py writes `analysis_results.json` (:767),
  `{model}_post_simulation_analysis.json` (:459), `analysis_summary.md` (:776),
  `cross_model_comparison_report.md` (:726), and `comprehensive_visualizations`
  (:620) — there is no `pymdp_visualizations/` dir in code (only a doc/AGENTS
  reference at src/analysis/AGENTS.md:215).
  Fix: replace output-product list with the real artifact names.

## 2. doc/gnn/modules/08_visualization.md

- doc/gnn/modules/08_visualization.md:276 | WARNING | Test-coverage command
  glob `src/tests/test_visualization_*.py` matches nothing — visualization
  tests live under `src/tests/visualization/` (13 tracked .py; 0 top-level).
  Command would report an empty collection. Same stale glob in
  src/visualization/AGENTS.md. Fix: `src/tests/visualization/`.

- Other claims verified clean: orchestrator "58 lines" (wc = 58); imports
  `process_visualization`, `generate_graph_visualization`,
  `generate_matrix_visualization`, `GNNVisualizer` all exported
  (src/visualization/__init__.py); `load_visualization_model` and
  `parse_gnn_content` exist (core/parsed_model.py, parse/markdown.py); MCP tool
  IDs match src/visualization/mcp.py; artifact names (network_graph.png,
  network_stats.json, viz_manifest.json, visualization_summary.json) present
  in source.

## 3. doc/gnn/modules/09_advanced_viz.md

- doc/gnn/modules/09_advanced_viz.md:470-481 | WARNING | MCP tools listed
  (`advanced_visualization.generate_3d`, `create_dashboard`, `generate_d2`,
  `analyze_statistics`) do not match the tools actually registered in
  src/advanced_visualization/mcp.py (process_advanced_visualization,
  check_visualization_capabilities, list_d2_visualization_types,
  get_advanced_visualization_module_info). The doc's fabricated tool names are
  not registered. Fix: list the real registered tool IDs.

- doc/gnn/modules/09_advanced_viz.md:280 | INFO | `{model}_3d_visualization.html`
  documented as a 3D output, but code writes `{model}_3d_visualization.png`
  (network_viz.py:180); interactive/dashboard output is
  `{model}_dashboard.html` / `{model}_interactive_dashboard.html`. Fix: `.png`.

- doc/gnn/modules/09_advanced_viz.md:283 | INFO | `{model}_visualization_data.json`
  listed as an output product but no `.py` writes that file (only
  `extract_visualization_data` helper exists; used for validation, not saved as
  that filename). Fix: correct/remove the product name.

- Orchestrator "65 lines" verified (wc = 65); `process_advanced_viz` signature
  matches processor.py:338; D2Visualizer + generate_all_diagrams_for_model
  exist (d2_visualizer.py:68/:688); test files exist
  (test_advanced_visualization_overall.py, api/test_comprehensive_api.py). OK.

## 4. doc/performance/README.md — CLEAN

Every command verified well-formed against ARGUMENT_DEFINITIONS /
src utils: `main.py --only-steps/--profile/--estimate-resources/--skip-llm/
--no-animations` all defined; `5_type_checker.py --estimate-resources`;
scripts/run_pymdp_gnn_scaling_analysis.py exists; `11_render.py
--frameworks/--strict-framework-success`; `12_execute.py
--render-output-dir/--execution-workers/--timeout/--distributed/--backend`
all defined. The "not main-pipeline options" list is accurate (none of
`--workers`, `--parallel-strategy`, `--memory-limit`, etc. are registered).
Output path `output/<run>/00_pipeline_summary/pipeline_execution_summary.json`
matches src/pipeline/context.py:215. Clean.

## 5. doc/visualization/README.md, doc/advanced_visualization/README.md,
   doc/research/README.md, doc/petri_nets/README.md — INFO counts only

- doc/visualization/README.md:36 "Files: 1 | Subdirectories: 0" — but 3 tracked
  files (README, AGENTS, SPEC). INFO.
- doc/advanced_visualization/README.md:35 same "Files: 1" — 3 tracked. INFO.
- doc/research/README.md:36 "Files: 1" — 3 tracked. INFO.
- doc/petri_nets/README.md:47 "Files: 4" — 6 tracked (README, AGENTS, SPEC,
  __init__.py, pnml.pnml, xml.xml). INFO.
All cross-reference targets checked via git ls-files resolve correctly
(glowstick, gnn_overview.md, gnn_tools.md, axiom, nock-gnn, gnn_multiagent,
advanced_modeling_patterns, poe-world, src/research/README.md, etc.).

## 6. doc/spm/ — clean (conceptual)

doc/spm/spm.md, doc/spm/spm_gnn.md are conceptual/informational (no CLI
commands, no src module exists for SPM). Traced references resolve. The
pseudocode `calibrate_gnn_from_spm` / `extract_connectivity_matrix` are clearly
illustrative (no src/spm package) and match the module's stated "integration
concept" framing; no fabricated source paths. INFO: doc/spm/AGENTS.md states
"Files: 3" while 5 tracked. INFO.

## Summary
- ERROR: 1 (16_analysis.md:249-253 usage example broken).
- WARNING: 6 (16_analysis.md:117 signature, :163-176 return schema,
  :216-220 env/config, :270-296 output products; 08:276 test glob;
  09:470-481 MCP tool names).
- INFO: 8 (cosmetic counts + file-extension/product-name drift).
- CLEAN: doc/performance/README.md; doc/spm conceptual docs.
No file, index, or git state was touched.
