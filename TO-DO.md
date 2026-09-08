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

## Deep horizon wave 2 - pipeline orchestration

Scouted 2026-09-08 against tip b73e467bf (six read-only scouts over
`src/gnn/pipeline/`, the 25 numbered orchestrators, and pipeline-facing
diagnostics). Import-surface retirement (v3.3.0) is clean; remaining defects
are stale-path bugs, silent-swallow hardening gaps, and untested wiring
surfaces.

| ID | Scope | Acceptance evidence |
| --- | --- | --- |
| W2-D1 | Medium: `src/gnn/pipeline/pipeline_runtime_validator.py:63-89` probes retired `src/render/...` paths (post-v3.3.0 they are `src/gnn/render/...`), so every renderer probe silently no-ops and the check validates nothing. Retarget probes to the real package paths via `importlib.util` source lookup; delete or fail-loud unreachable ones. | At least one probe reads an existing renderer file and one deliberate bad-path probe is covered by a test that fails when the path is missing; `uv run --extra dev python -m pytest tests/pipeline/test_pipeline_runtime_validator.py -q` green. |
| W2-D2 | Medium: `src/gnn/utils/pipeline_arguments.py:138` defaults `--ontology-terms-file` to the retired `src/ontology/act_inf_ontology_terms.json` (real file lives at `src/gnn/ontology/`), so the default always misses at runtime. Point the default at the packaged file; migrate fixture strings in `src/gnn/utils/test_utils.py:445,531`. | Pipeline runs without `--ontology-terms-file` resolve the packaged file; fixture-updated tests pass; no other `src/ontology` string remains in src/. |
| W2-D3 | Medium: `src/gnn/execute/executor.py:91` imports `get_output_dir_for_script` from the legacy `gnn.utils.pipeline_template` copy while `gnn.pipeline.config` is canonical. Cutover the import and add a module-level `DeprecationWarning` re-export shim on the legacy symbol (MAJ-05 pattern). | Same directory names for `12_execute.py` before/after; `gnn.utils.pipeline_template.get_output_dir_for_script` emits `DeprecationWarning`; zero internal non-test callers of the legacy symbol. |
| W2-D4 | Medium: 11→12 handoff is silent when `render_processing_summary.json` is missing (`execute/processor.py:549` defaults `require_render_summary=False`). Flip the default to True with an explicit opt-out; verify the public POMDP GridWorld run still passes (render always precedes execute there). | Step 12 without a render summary fails exit 1 unless opted out; `just pipeline` / public-run receipts unaffected; targeted execute tests green. |
| W2-D5 | Medium: preflight's cheap `validate_config` (`src/gnn/pipeline/preflight.py:165-184`) is never run by the pipeline itself - bad `pipeline.skip_steps` config is only caught if the user separately ran `gnn preflight`. Wire the config-only validation into `_prepare_pipeline_context` (`src/gnn/main.py`) before any step executes. | `pipeline.skip_steps: ["abc"]` in `input/config.yaml` fails the run fast with the preflight message before step 0; config-free runs unaffected; new wiring test green. |
| W2-D6 | Medium: `health_check.py` error branches untested (version_issues :210-214, julia subprocess failure :274-286, incomplete structure :334-337, limited/partial integration :358-366, scoring tiers :388-449, recommendations :456-520, `main()` exit codes :655-700, verbose report). Add monkeypatched deterministic tests; fix stale header comment (:15-16) and the `/24 available` vs 25-step strings (:593,598,684). | +8-10 tests in `tests/pipeline/test_health_check.py`, all offline; full health-check file green; cosmetic strings consistent. |
| W2-D7 | Medium: composition wiring untested - config-driven `only_steps`/`skip_steps` fallback, `skip_llm` auto-inject, `--autonomous` main() branch, serial/parallel loops with faked `execute_pipeline_step`, publish-gate failure → `_save_minimal_pipeline_summary`, mid-run crash receipt, `GNN_RUN_ID` env scope, `testing_matrix` global_steps skip + folder fan-out (all in `src/gnn/main.py`). Also delete worthless tests: `tests/pipeline/test_pipeline_overall.py` hasattr-façade checks and dict-replay step-numbering test. | Cheap (<10ms each) wiring tests assert real main.py decisions with faked executors; deleted tests' assertions replaced by behavior tests; orchestration subset green. |
| W2-M1 | Minor: registry dead metadata - `StepInfo.additional_args_key` points at nonexistent `STEP_ADDITIONAL_ARGUMENTS` (step_registry.py:35); `default_recursive` field never consumed (scripts pass it directly to the template); stale stage-name comment (step_registry.py:33); registry `module_function` for 7_export documents inner `process_export` while the script registers wrapper `_export_with_geo` (rename the wrapper). | No `additional_args_key`/`STEP_ADDITIONAL_ARGUMENTS`/`default_recursive` references remain; `STANDARD_MODULE_FUNCTION_NAMES["7_export"]` matches the registered callable; step-registry tests green. |
| W2-M2 | Minor: `config.py` hardening - unknown stems silently fall back to `<stem>_output` (config.py:193-211) so producer/consumer can diverge on typos; YAML parse failure logged at `debug` (config.py:50-59); unreachable `.py` branch (config.py:172-178). Add a warning naming the script on the recovery path, raise parse-failure logging to error with the path, remove the dead branch with a pinning unit test. | Unknown stem logs a visible warning; malformed config produces error-level log with path; `get_output_dir_for_script("7_export.py", ...)` still returns `7_export_output`; targeted tests green. |
| W2-M3 | Minor: retired dotted-module references - `src/gnn/cli/SPEC.md:26` says `gnn = "src.cli:main"`, `src/gnn/mcp/README.md:362` documents `-m src.mcp.cli`; stale pre-v3.3.0 PYTHONPATH comment `src/gnn/execute/pymdp/pymdp_runner.py:137-139`; stale Julia project paths `src/gnn/render/health.py:63-64` (real envs under `src/gnn/execute/`); comment-only stub `register_tools` in `src/gnn/utils/test_utils.py:1003-1007`. | No `src.cli`/`src.mcp` dotted refs in src/gnn; documented stdio config resolves; stub deleted; mcp tool-count audit stays ≥140. |
| W2-M4 | Minor: stale path prose in all 25 numbered-script docstrings + `main.py` usage examples (main.py:37-47) + manuscript prose strings (`src/gnn/manuscript/variables.py:12,686,698,746,760,834`). Mechanical: `python src/NN_*.py` → `uv run python src/gnn/NN_*.py`, `src/X/` → `gnn/X` package refs. | `grep -rn "python src/(main\\\\.py\|[0-9])"` over src/gnn returns zero; doc gates (`check_gnn_doc_patterns`, `check_maintained_doc_terms`, `check_repo_terminology`, `check_doc_path_references`) pass. |
| W2-J1 | Major: three independent base-output-dir reconstruction heuristics (`export/processor.py:537-544` name-prefix heuristic, `analysis/framework_common.py:127-132` ImportError sibling fallback, `gui/runner.py:26-30` broad-except fallback) duplicate path logic with silent divergence risk. Consolidate into one `resolve_step_output_dir(step_stem, output_dir)` helper in `gnn.pipeline.config` with explicit nested-dir tests; migrate all call sites. | Single helper; all three call sites migrated with identical resolved paths for `output/`, `output/7_export_output/`, and arbitrary subdirs; unit tests pin nested inputs; full orchestration gate green. |

Out-of-scope observations (recorded, not scoped): `PipelineContext`
(`src/gnn/pipeline/context.py`) is dead weight - never instantiated by
main.py or any step; docs claim otherwise (context.py:3-5,
pipeline/AGENTS.md:148). AGENTS.md "Data Dependencies" graph overstates
in-memory propagation (steps 5/6/8/10/11/13 re-parse input; only 3→7 and
11→12 consume artifacts). `run_session`/durable streams are not wired into
main.py composition (feature gap, not a test gap). These need a design
decision, not a mechanical fix.

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
