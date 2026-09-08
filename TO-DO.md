# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-09-07 (wave 2: MAJ-07 closed - both pipeline utilities kept and wired with tests; setup_step_logging delegates + migration fossil retired; local-gates CI workflow added; dependency floors raised)
**Current Version**: 3.3.0
**Next Target**: v4.0.0 (bounded autonomy, pipeline stage consolidation, multi-agent stigmergic topologies, and high-dimensional active inference)

**Recently closed** (audit trail in `CHANGELOG.md` and git history, not here):
MAJ-02 (sparse Kronecker factorized execution + scaling sweep + numbered-pipeline
integration) and MAJ-03 (native stigmergic multi-agent compilation with
env-conditioned action selection; probe:
`uv run pytest tests/render/test_stigmergic_multi_agent.py -q`). The 3.2.0
release receipt (tests, mypy, ruff, documentation audits) is in `CHANGELOG.md`
§3.2.0.

GNN-02 (linear-Gaussian F/control/H/Q/R export:
`src/gnn/export/geo_infer_gaussian.py`, `tests/export/test_geo_infer_gaussian.py`,
paired analytic verification in `docs/development/geo_infer_2026_09.md`) and
GNN-03 (factor/modal dependency axes and multi-step policy enumeration:
`src/gnn/export/geo_infer_factored.py`, `tests/export/test_geo_infer_factored.py`)
closed 2026-09-07 after re-verification against the 3.3.0 tree.

MAJ-07 closed 2026-09-07: both pipeline utilities are KEPT and wired with
direct tests - they are documented public API (`gnn.pipeline.__all__` +
`pipeline/SKILL.md` usage examples) and `pipeline/pipeline_validator.py` is
the health check's live integration probe. Fixes landed while wiring: the
core-dependency check imported PyYAML by distribution name (`pyyaml` vs
`yaml`, so core deps always reported unhealthy), scipy/pathlib were listed
as core (scipy moved to the ml-ai extra in 3.3.0), the runtime validator
shelled out to the retired pre-restructure orchestrator (`main.py` at the
`src/` root; now `src/gnn/main.py`), and its import fallback could raise
NameError. Tests: `tests/pipeline/test_health_check.py` and
`tests/pipeline/test_pipeline_validator.py`.

MAJ-04 closed 2026-09-07: all six >2000-line modules decomposed via the
3.3.0 `execute/processor.py` split pattern (mechanical sibling extraction,
facade re-exports preserved, one module per PR) — `analysis/visualizations.py`
2412→58 (PR #29), `analysis/analyzer.py` 2031→263 (PR #32),
`render/jax/jax_renderer.py` 2200→170 (PR #33), `render/discopy/translator.py`
2150→303 (PR #34), `integration/meta_analysis/visualizer.py` 2871→283 (PR #40,
`Sweep*Mixin` variant preserving byte-identical class-method moves), and
`testing/test_round_trip.py` 2214→1356 (PR #43, `round_trip_*` siblings).
Class-method bodies moved verbatim into mixins where module-level extraction
was impossible. Every facade re-exports every moved name (no consumer
import-path changes; per-module facade-contract probes 31/34/27/36/13/47
names), moved code verified byte-identical modulo imports
(2347/2001/2190/2108/2811/2172 lines), full suite green at every PR
(4263→4272 passed), mypy/ruff 0 throughout. Session benchmark
`oversized_module_lines` 13878 → 0: no tracked `src/gnn` Python file exceeds
2000 lines (`gnn_python_lines` +0.6% across the series — code moved, not
deleted). The row's "shared subprocess envelope the nine per-framework
renderers duplicate" item is RESCOPED to its own future work: renderers
contain zero subprocess code (verified); the duplication is execute-side
(`rxinfer`/`stan`/`lean`/`activeinference` runners + 4 `executor.py` MCP
methods vs the canonical `execute_script_safely` at `execute/executor.py:1089-1200`)
and needs a behavior-preserving refactor with its own tests, not a
mechanical split.

## Open Scoped Roadmap

Every item below is cold-startable: scope, files, verification, and acceptance
are pinned.

| ID | Scope | Acceptance evidence |
| --- | --- | --- |


### Smaller scoped cleanups (independent of the majors)

- ruff F401/F811 policy ignore: RESOLVED 2026-09-07 - the global
  `F401`/`F811` ignore entries are gone from pyproject; nine genuine
  re-export surfaces (six MAJ-04 facades, `round_trip_availability`,
  `visualizer_style`, `execute/processor.py`) and the `src/gnn/parsers/*`
  guarded optional-backend probes hold documented per-file-ignores, and
  66 genuinely dead imports were removed (57 src/gnn, 10 scripts, 9 F811
  re-imports). `ruff --select F401,F811 src/gnn` now reports 0 findings;
  `ruff check src/gnn scripts`, mypy, and the full suite stayed green
  (4285 passed). Consumer safety: AST-resolved `from <module> import`
  scan across src/gnn, tests, and scripts against every removed name.
- Local/CI parity: tokens and skills-health are CI-wired via
  `.github/workflows/local-gates.yml` (2026-09-07; `skills-health` also
  needed a repo-root sys.path bootstrap). `just gridworld` remains
  unwired deliberately - the committed `output/` tree currently fails
  its contract and regeneration needs the Julia toolchains.
  ml-ai/torch extras parity: RESOLVED 2026-09-08 - verified
  `uv sync --extra dev --extra ml-ai --extra torch --frozen` resolves from
  the lock and un-skips the 12 environment-skipped tests (11 sklearn
  inference tests, 1 torch continuous-render test); all 22 tests in the two
  affected files pass with the extras present (no latent failures behind the
  skip). The local test-cov command should therefore run with
  `--extra ml-ai --extra torch` appended. The Ollama-ignore half of this
  item stays open-by-design: no local Ollama daemon exists, so
  `test-cov`'s `--ignore=tests/llm/test_llm_ollama*.py` remains correct
  locally while the CI coverage run exercises those tests where they
  degrade gracefully without a daemon. Coverage selection parity on the
  remaining axis: `just test-cov` now adopts CI's
  `-m "not pipeline and not mcp"` deselect so both invocations apply the
  same pipeline/mcp test policy (4326 collected locally; CI collects
  4352 - the 26-test Ollama delta is the open-by-design asymmetry
  recorded above).
- Dependency floors: RAISED 2026-09-07 for numpy (>=2.0), pandas
  (>=2.0), openai (>=2.0), pytest (>=8.0), mypy (>=1.0) - the lock
  resolved identically (only requires-dist metadata moved; zero package
  pins changed). Remaining cosmetic floors (networkx 2.6, plotly 5.15,
  scipy 1.7, ...) can follow at the next deliberate lock refresh.
- `gnn/utils/pipeline_validator.py` vs `gnn/pipeline/pipeline_validator.py`
  near-name collision: RESOLVED 2026-09-08 — the lower-traffic runtime
  integration tester renamed to
  `gnn/pipeline/pipeline_runtime_validator.py` (compatibility module at the
  old path emits `DeprecationWarning` and re-exports `PipelineValidator`/`main`;
  contract pinned in `tests/pipeline/test_pipeline_runtime_validator.py`);
  import-site grep has zero stragglers.
- Stale singular module paths in maintained docs: RESOLVED 2026-09-08 —
  all 21 occurrences (19 lines) of `src/gnn/parser.py`, `src/gnn/schema.py`,
  and `src/gnn/schema_validator.py` re-pointed to their verified real homes
  (`schema/parser.py`, `schema_validator/syntax.py`, `parsers/system.py`);
  regression gate `scripts/check_doc_path_references.py` is CI-wired
  (local-gates) and strict (cap 0).

## Deep horizon wave 2 - analysis + utils

Scoping for territory `src/gnn/{analysis,utils,advanced_visualization,type_checker,schemas,api}`
(branch `deep/gnn-utils-analysis`). Measured baseline via the deterministic
`autoresearch.sh` benchmark: `quality_violations=120` (mypy `--strict
--follow-imports=silent` over the territory; ruff = 0) with the territory test
subset at 808 passed / 0 failed. Items reach into `validation/`,
`schema_validator/`, `integration/meta_analysis/`, and `visualization/` only
where the contract fix is impossible from inside the territory.

| ID | Size | Scope | Acceptance evidence |
| --- | --- | --- | --- |
| W2-01 | major | Single canonical GNN section contract. Today three divergent "required sections" lists claim authority: `schemas/json.json:10-19` + `schemas/yaml.yaml:14-23` (require ModelAnnotation+InitialParameterization+Time, no Signature), `type_checker/output_utils.py:74-82` (require Signature, demote Time), `validation/mcp.py:187-192` (fourth divergent set). Hoist one required/optional table beside `type_checker/checking/sections.py` `CANONICAL_GNN_SECTIONS` (e.g. `gnn/schemas/section_contract.py`) and derive all consumers from it, with a drift test pinning `json.json`/`yaml.yaml` `required_sections` to the table. | A Signature-present/Time-absent fixture returns the same verdict from the type_checker and validation MCP surfaces; drift test fails if schema files and table disagree; validator/type_checker/api suites green; doc gates pass. |
| W2-02 | medium | utils dead-surface removal: delete `dependency_manager.py`, `dependency_audit.py`, `dependency_installer.py`, `diagnostic_logging.py`, `network_utils.py`, `simulation_monitor.py` (incl. import-time `global_monitor` side effect at simulation_monitor.py:199-201), `pipeline_planner.py` (lazy export at utils/__init__.py:128, zero callers), `script_validator.py` (test-only consumer); prune the lazy re-exports; fix the ghost `migration_helper.py` reference (utils/README.md:75, tests/pipeline/test_pipeline_infrastructure.py:351) and the stale "never implemented" comment (tests/pipeline/test_pipeline_error_scenarios.py:314-316). | Repo-wide grep shows zero imports of deleted modules across src/, scripts/, tests/; tests/utils plus affected pipeline/infrastructure tests green; ruff+mypy clean on touched files. |
| W2-03 | medium | type_checker result-contract unification: canonical `is_valid` key (core.py:423 emits "valid"; output_utils.py:46 reads "is_valid"; cli.py:151 shims per file); delete `_time_is_dynamic` in favor of the shared 4-marker `detect_time_dynamics` (core.py:598 vs checking/sections.py:170-176) so `model_type` and `time_dynamics.is_dynamic` cannot disagree; document the three sibling return contracts (check_file 4-tuple, validate_content dict, validate_gnn_files bool/int sentinel). | tests/type_checker pins that one content sample yields consistent `is_valid` and `time_dynamics.is_dynamic` across check_file/validate_content/CLI render; continuous-time fixture reports Dynamic AND is_dynamic=true; suite green; mypy clean. |
| W2-04 | medium | analysis internal dedup + typing: `extract_pymdp_data` routes through `_normalise_current_simulation_payload` (framework_extractors.py:110-131 vs 20-46); meta_analysis collector entropy reuses `math_utils.compute_shannon_entropy` (collector.py:355-358 vs analysis/math_utils.py:30-38); mcp.py complexity heuristics route through `analysis_extraction` (mcp.py:120-124); analysis_statistics complexity block calls analysis_complexity functions instead of re-inlining the formula (analysis_statistics.py:150-157 vs analysis_complexity.py:23-24); tighten `list[Any]`/`dict[Any, Any]` accumulators (analysis_extraction.py:19, analysis_statistics.py:82-83, processor.py:221); explicit `__all__` on analyzer.py excluding third-party names (stats, sns, SCIPY_AVAILABLE, SEABORN_AVAILABLE) so no_implicit_reexport holds. | Behavioral equivalence pinned by existing tests (same numeric outputs); mypy --strict count on analysis strictly lower than baseline; tests/analysis green. |
| W2-05 | medium | advanced_visualization consolidation: single palette (import `VAR_TYPE_COLORS` from `visualization/theme.py` or hoist both; _shared.py:22-32 == theme.py:32-43 value-for-value); move `normalize_connection_format` to a shared home fixing the inverted `visualization/graph/network_visualizations.py:5` import of `advanced_visualization._shared`; delete the duplicated node-render loop (network_viz.py:97-121 vs 156-176); route visualizer.py module imports through _shared guards (visualizer.py:18-24 unconditional matplotlib/numpy); judge the six zero-caller `create_*` helpers (visualizer.py) and the dual Step-9 orchestrators (processor.py vs visualizer.py). | Theme-parity test pins one palette; no src module outside the package imports `gnn.advanced_visualization._shared`; `from gnn.advanced_visualization import ...` degrades gracefully without matplotlib; viz suites green; mypy clean. |
| W2-06 | medium | api surface coherence: derive step count + LLM-step set from `pipeline.step_registry` (models.py:47-48 and app.py:103/172-174 hardcode 25/{13}; processor.py:264-275 already derives PIPELINE_STEPS); hoist the triplicated resolve_repo_path + PathValidationError-to-400 block (server.py:110-125, app.py:119-135, mcp.py:30-34) into one helper; make `gnn_submit_job` MCP tool honest (execute via execute_job_async or document + test the non-executing contract — today jobs sit pending forever, mcp.py:43-49); version single-source (app.py:81/104 "3.2.0" vs models.py:250 "2.0.0" default); MCP tools result gains `total` for REST parity. | New contract tests: registry-derived step counts (adding step 25 fails loudly in api, not silently), submit-tool contract pinned, MCP tools count == REST tools count; mcp tool-count gate unchanged (141 total / 5 api tools); api suite green. |
| W2-07 | minor | api metadata typing + stale defaults + contract-test gap: `__all__: list[Any]` and `__dependencies__: list[Any]` become `list[str]` (api/mcp.py:24, api/__init__.py:43); RunHealthResponse.version stale "2.0.0" default replaced by MODULE_VERSION (models.py:250); add the 404-on-missing-report envelope case to tests/api/test_api_response_contract.py. | mypy clean on api; response-contract test covers the mixed report contract (envelope 404 vs plain-text 200). |
| W2-08 | minor | analysis public-surface hygiene: judge the exported-but-never-called `count_type_distribution`/`build_connectivity_matrix` (analysis_statistics.py:81,90; only re-export chains reference them) — remove from the surface or wire into perform_statistical_analysis; resolve the `check_analysis_tools` name collision between analysis/__init__.py:86-95 (numpy/pandas/scipy/matplotlib) and intelligent_analysis/__init__.py:134-136 (llm_processor/numpy/pandas). | No zero-caller exports remain unjudged; collision resolved by rename/delegation with callers updated; tests green. |
| W2-09 | minor | schema_validator vestigial plumbing (cross-module reach from the schemas territory): remove the no-op `use_formal_parser`/`FORMAL_PARSER_AVAILABLE` path (validator.py:25,67,77), delete the production `__main__` demo block (validator.py:856-865), and either stop loading json.json/yaml.yaml at init when nothing reads `.schema` (validator.py:63-66) or wire real jsonschema validation; AGENTS.md:26-27 claims must match code. | schema_validator tests green; no init-time parse without a consumer (or genuine schema enforcement added); doc gates pass. |
| W2-10 | minor | utils typing pass: concrete result types for dict-soup APIs — `get_pipeline_step_info` TypedDict (arg_parsing.py:1053-1055), `batch_write_files` result (io_utils.py:20-22), `log_step_start` return (logging_utils.py:64), `validate_arguments` errors list (arg_parsing.py:1212). | mypy --strict count on utils strictly lower than baseline; tests/utils green. |
| W2-11 | minor | Direct unit tests for the analyzer.py quartet called by process_analysis: `perform_statistical_analysis`, `run_performance_benchmarks`, `perform_model_comparisons`, `generate_analysis_summary` (zero direct tests today; only MCP wrappers are covered). | New tests/analysis file pins the four functions' output shapes on a small fixture model; suite green. |

Verification commands: `bash autoresearch.sh` (benchmark), `uv run --extra dev mypy --strict --follow-imports=silent src/gnn/analysis src/gnn/utils src/gnn/advanced_visualization src/gnn/type_checker src/gnn/schemas src/gnn/api --config-file pyproject.toml`, and the targeted pytest subsets per touched module.

---

## v4.0.0 - Bounded Autonomy & Reviewed Self-Editing

The local bounded-autonomy surface emits proposal-only artifacts via
`--autonomous`: candidate scores, review gates, rollback descriptors, audit
events, and non-mutating security policy. No source edit, commit, container
run, or cluster mutation is automatic.

Concrete, cold-startable v4.0.0 work is scoped in the Open Scoped Roadmap
table above; this section records the unscoped vision and the current
proposal-only `--autonomous` surface.

---

## Verification Commands

Use `uv run` for roadmap verification checks:

```bash
PYTHONPATH=src uv run python scripts/run_v3_orchestration_acceptance.py --strict
PYTHONPATH=src uv run python scripts/emit_run_manifest.py output --out /tmp/gnn-v3-run-manifest
PYTHONPATH=src uv run python scripts/generate_pipeline_container_plan.py --config input/config.yaml --out /tmp/gnn-v3-container-plan.json
PYTHONPATH=src uv run python scripts/run_session_acceptance.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-v3-session-acceptance --session /tmp/gnn-v3-session.json --strict
PYTHONPATH=src uv run python src/gnn/main.py --autonomous --target-dir input/gnn_files --output-dir /tmp/gnn-autonomous-smoke

uv run python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run python scripts/check_gnn_doc_patterns.py --strict
uv run python scripts/check_maintained_doc_terms.py --strict
uv run python scripts/check_repo_terminology.py --strict
uv run python scripts/check_doc_path_references.py
uv run python scripts/check_capability_contracts.py
uv run python scripts/run_semantic_fidelity_gate.py --output-dir /tmp/semantic_fidelity --strict
uv run python scripts/run_cross_framework_reliability.py --output-dir /tmp/cross_framework --strict
git diff --check
```

## Conventions

- Keep this file limited to unchecked, forward-looking work.
- Move shipped-version details to release notes, changelog entries, or durable
  verification artifacts.
- Keep closed work out of this file: completed items are removed when they
  land; the audit trail lives in `CHANGELOG.md` and git history.
- Scope open items with concrete tasks, file paths, verification commands, and
  acceptance criteria so the next session can execute without re-deriving them.

## GEO-INFER contract expansion

The delivered opt-in v1 format is specified in `src/gnn/export/geo_infer_contract.md`.
Further work must preserve independently installable runtimes and explicit matrix,
space and time semantics.

| ID | Scope | Acceptance evidence |
| --- | --- | --- |
| GNN-04 | Pin paired repository revisions in cross-repository CI on the GNN side; the GEO side already hosts paired CI retaining both revisions plus categorical/H3/Gaussian/factored digests (`docs/development/geo_infer_2026_09.md`), and `.github/` has no GNN-side equivalent. | A GNN-side workflow (or documented receipt-pinning procedure) completes paired categorical and H3 round trips and records source/artifact digests for both revisions. |
| GNN-05 | Notation-driven metadata discovery for GEO-INFER export: derive step seconds/units/space kind from the GNN notation instead of explicit user JSON. The explicit-CLI wiring and original-source provenance already landed (`src/gnn/7_export.py`, `src/gnn/export/processor.py`, `tests/export/test_export_geo_pipeline.py`, `tests/export/test_geo_infer_gaussian.py`). | Notation-derived metadata passes the same visible-failure and unchanged-five-format-default tests that pin the explicit path. |
