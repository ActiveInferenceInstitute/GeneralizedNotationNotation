# D-analyz — GNN documentation accuracy fixes (SAFE DOC-ONLY EDITS)

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation

## Mission
Apply the documentation corrections listed below. These are SAFE, DOCUMENTATION-ONLY
edits: change markdown prose, code examples, import references, file paths, counts,
and doc metadata. Do NOT change any source code (.py), test logic, config, or
dependency. The fixes were verified against the current tree (imports resolve; paths
are tracked); your job is to apply them to the .md files exactly.

## Rules
- Edit ONLY the .md files named below. Do NOT touch src/**, scripts/**, pyproject.toml,
  tests/**, or any .py.
- Preserve surrounding formatting/markdown. Make the minimal change (one line / one
  token) per fix.
- For imports: replace the broken module path with the verified-correct one below.
- For counts: set the number to the value stated below (verified by git ls-files/wc).
- For prose claims that are wrong/unverifiable and no exact replacement exists: reword
  minimally to be accurate (e.g. mark as illustrative, or remove the fabricated claim).
- HARD RULE: do NOT commit, stage, or push. Leave changes uncommitted.

## Specific fixes to apply
FIXES:
1. doc/gnn/modules/16_analysis.md:117,140 — signature `process_analysis(target_dir, output_dir, logger=None, **kwargs)` -> `process_analysis(target_dir, output_dir, verbose=False, **kwargs)`
2. doc/gnn/modules/16_analysis.md:163-176 — return keys: actual is `variable_statistics, connection_statistics, section_statistics, distributions, correlations` (not variable_count/density_metrics/type_distribution/dimension_statistics)
3. doc/gnn/modules/16_analysis.md:249-253 — fix usage example: call `perform_statistical_analysis(file_path, verbose)` with a real file_path and read `['variable_statistics']['count']` / `['connection_statistics']['count']` (NOT stats['connection_statistics']['density'])
4. doc/gnn/modules/16_analysis.md:216-220 — env vars ANALYSIS_PERFORMANCE_MODE / ANALYSIS_TIMEOUT and analysis_config.yaml are not consumed/tracked — annotate as illustrative/reserved.
5. doc/gnn/modules/16_analysis.md:224-228 — DEFAULT_COMPLEXITY_THRESHOLDS is not a defined constant — annotate the block as illustrative or describe the inline-computed thresholds.
6. doc/gnn/modules/16_analysis.md:270-296 — output products: real artifacts are `analysis_results.json`, `{model}_post_simulation_analysis.json`, `analysis_summary.md`, `cross_model_comparison_report.md` — replace the fabricated names ({model}_statistical_analysis.json, {model}_complexity_metrics.json, {model}_performance_benchmarks.json).
7. doc/gnn/modules/08_visualization.md:276 (+ src/visualization/AGENTS.md) — test glob `src/tests/test_visualization_*.py` -> `src/tests/visualization/`
8. doc/gnn/modules/09_advanced_viz.md:470-481 — MCP tools: real registered tools are `process_advanced_visualization, check_visualization_capabilities, list_d2_visualization_types, get_advanced_visualization_module_info` (replace fabricated generate_3d/create_dashboard/generate_d2/analyze_statistics)
9. doc/gnn/modules/09_advanced_viz.md:280 — `{model}_3d_visualization.html` -> `{model}_3d_visualization.png` (interactive is {model}_dashboard.html)
10. doc/gnn/modules/09_advanced_viz.md:283 — `{model}_visualization_data.json` not written — correct/remove the product name.


## Verification
After editing, run:
- uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write   (must stay green)
- uv run --extra dev python scripts/check_repo_terminology.py --strict   (must stay clean — do NOT use banned words: legacy/stub/placeholder/deprecated)
- uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict

## Report
Write a concise report to /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-docfix/REPORT-fix-D-analyz.md listing each file you changed and the specific
edit(s) applied. Reply with only the absolute path to your report.
