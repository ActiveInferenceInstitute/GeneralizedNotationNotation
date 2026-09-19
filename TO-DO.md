# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-09-15 (round-2 sweep + scope pass: MCP god-class
decomposed per the MAJ-04 pattern (`src/gnn/mcp/mcp.py` 1898 → 465 lines;
five responsibility mixins — discovery/registry/execution/introspection/
metrics — 44/44 byte-identical member moves, facade + import surface +
registered tool names unchanged, `tests/mcp/` 417 green);
`utils/simulation_utils.py` REMOVED as production-dead (R4 resolved; zero
non-test importers; matplotlib module-scope import gone; `_EXPORT_MAP`
unchanged at 113), remaining 1500–2000-line band recorded and re-verified
by this pass (main.py 1949, pomdp_extractor 1775, execute/processor 1543),
numbered-script docstring run commands migrated to `uv run` (27 lines) +
`.agent_rules` requirements.txt/PYTHONPATH residue retired; scope pass:
TO-DO truth pass (12 open rows cleared with fresh probes, 7
residual rows re-verified) completed by the corrective scope agent — 8 more
probe batches dispositioned MAJ-06 (landed: all 18 module `process_*_mcp`
wrappers delegate through `run_pipeline_step_mcp`,
`src/gnn/utils/mcp/dispatch.py`) and GNN-04 (pin current at
`.github/gnn-pair.json` @ `c0115779`), confirmed the F401/F811 split /
extras parity / dependency-floors rows RESOLVED-in-place, shrunk W2-M2 to
pinning tests (both behavior halves verified landed), confirmed the W2-J1
third site migrated, and wrote the `SCOPE-2026-09-15.md` spec
(14 rows cleared, 7 open rows, 2 major / 3 medium / 2 minor improvements);
execution wave 2026-09-15: N-1 + N-6 + N-7 landed (`d79c62756`), N-5 landed
(`5501ec7c1`), N-4 decision executed as delete with the StepStatus re-home
(`c89452e00`; the census falsified the closed-surface premise — see the
pipeline-orchestration section), manuscript token map + figures regenerated
(`250b7879c`, `b53a74a8a`), fep_lean re-sealed (pin cycle #8, `2b51c3d`) and
the pair bumped (`7453551c7`), GEO pin bumped (`d493253b` side; both
interchange pins current). Evidence in CHANGELOG 2026-09-15 and
`SCOPE-2026-09-15.md`.)
**Current Version**: 3.4.0
**Next Target**: v4.0.0 (bounded autonomy, pipeline stage consolidation, multi-agent stigmergic topologies, and high-dimensional active inference)

**Recently closed** (audit trail in `CHANGELOG.md` and git history, not here):
MAJ-02 (sparse Kronecker factorized execution + scaling sweep + numbered-pipeline
integration) and MAJ-03 (native stigmergic multi-agent compilation with
env-conditioned action selection; probe:
`uv run pytest tests/render/test_stigmergic_multi_agent.py -q`). The 3.2.0
release receipt (tests, mypy, ruff, documentation audits) is in `CHANGELOG.md`
§3.2.0.

Scope pass 2026-09-15 (truth pass; per-item evidence in
`SCOPE-2026-09-15.md` §Cleared-Item-Evidence): cleared W2-D1 (validator
probes package-rooted and fail-loud on missing targets,
`src/gnn/pipeline/pipeline_runtime_validator.py:68-85`), W2-D2 (ontology
default resolves the packaged file,
`src/gnn/utils/arguments/pipeline_arguments.py:13-15`; zero retired
`src/ontology` strings in src/gnn), W2-D3 (zero legacy
`gnn.utils.pipeline_template` imports; the numbered scripts import
`gnn.utils.pipeline_orchestration.pipeline_template` directly), W2-D4
(`src/gnn/execute/processor.py:558` defaults `require_render_summary=True`),
W2-D5 (`src/gnn/main.py:459-485` `_preflight_config_gate`, wired at `:608`),
W2-D7 (`tests/pipeline/test_main_wiring.py` exists; `test_pipeline_overall`
façade checks deleted per the 2026-09-11 wave), W2-M1 (dead registry tokens
absent; registry names `_export_with_geo`,
`src/gnn/pipeline/step_registry.py:104-105,273-275`), W2-M3 (zero
`src.cli`/`src.mcp`/stale-`PYTHONPATH=src`/retired `src/render/` strings in
src/gnn), W2-M4 (the row's own acceptance grep returns zero), MED-03b
(`src/gnn/execute/rxinfer/rxinfer_runner.py:66-69` evidence persistence),
MED-04/MIN-01 (transport limits + hygiene, 2026-09-11 wave). W2-D6 landed
only its strings half; W2-M2's second half was verified landed by the
corrective pass (error-level parse log with path, dead `.py` branch gone)
and the row is shrunk to pinning tests — both remain residuals in the
pipeline-orchestration table, with W2-J1's third site confirmed migrated
(`export/processor.py:585-588`).
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

Every open item below is cold-startable: scope, files, verification, and
acceptance are pinned. The former top-level roadmap table was emptied by
landings and is retired (2026-09-15 scope pass); open rows live in the
territory tables below and in the 'Still open (residuals)' table near the
end of this file.

### Smaller scoped cleanups (independent of the majors)

- ruff F401/F811 policy split: RESOLVED 2026-09-07 — global ignores gone,
  nine genuine re-export surfaces + the `src/gnn/parsers/*` guarded
  optional-backend probes hold documented per-file-ignores, 66 genuinely
  dead imports removed; `ruff --select F401,F811 src/gnn` reported 0 at the
  time (receipt in CHANGELOG; not re-run within the 2026-09-15 probe
  budget).
- Local/CI parity: tokens and skills-health CI-wired 2026-09-07;
  ml-ai/torch extras parity RESOLVED 2026-09-08 (`uv sync --extra dev
  --extra ml-ai --extra torch --frozen` un-skips the 12 env-skipped tests;
  all 22 affected tests pass with the extras present). Residuals,
  open-by-design: local `test-cov` keeps
  `--ignore=tests/llm/test_llm_ollama*.py` (no local daemon) while the CI
  coverage run exercises those tests (26-test asymmetry, documented);
  ~~`just gridworld` stays unwired until Julia toolchains exist.~~
  REWIRED 2026-09-19: the recipe now runs the full 25-step pipeline on
  `input/gnn_files/pomdp_gridworld` with `--frameworks all` (local Julia
  1.12.7 + the committed RxInfer/ActiveInference envs) and validates the
  result with `scripts/check_pomdp_gridworld_outputs.py output`. CI wiring
  stays off by design: the run-tree is volatile
  (docs/development/output_tracking.md), so CI has nothing committed to
  check until that boundary changes.
- Dependency floors: RAISED 2026-09-07 for numpy (>=2.0), pandas (>=2.0),
  openai (>=2.0), pytest (>=8.0), mypy (>=1.0). Residual: cosmetic floors
  (networkx 2.6, plotly 5.15, scipy 1.7, ...) at the next deliberate lock
  refresh.
- `gnn/utils/pipeline_validator.py` near-name collision: RESOLVED
  2026-09-08 — renamed to `gnn/pipeline/pipeline_runtime_validator.py`
  (compat module at the old path emits `DeprecationWarning`; zero
  import-site stragglers).
- Stale singular module paths in maintained docs: RESOLVED 2026-09-08 —
  21 occurrences repointed to their verified real homes; regression gate
  `scripts/check_doc_path_references.py` is CI-wired and strict (cap 0).

## Deep horizon wave 2 - analysis + utils

All eleven scoped rows (W2-01..W2-11) landed via PR #60 (2026-09-08);
audit trail: git history and the PR description. Final measured state via
`bash autoresearch.sh`: `quality_violations=0` (mypy `--strict
--follow-imports=silent` over `src/gnn/{analysis,utils,advanced_
visualization,type_checker,schemas,api}` — was 120) with ruff at 0 and
the territory test subset at 821 passed / 0 failed (was 808 passed).
Key landings: single canonical GNN section contract
(`gnn/schemas/section_contract.py` with schema-drift tests), eight
production-dead utils modules removed, type_checker/analysis/advanced_
visualization/api deduplication and contract unification, api step
surface registry-derived, whole-territory strict typing clean.

---


## Deep horizon wave 2 - pipeline orchestration

Scoped 2026-09-08 against tip b73e467bf (six read-only scouts over
`src/gnn/pipeline/`, the 25 numbered orchestrators, and pipeline-facing
diagnostics). Rows W2-D1..D5, W2-D7, W2-M1, W2-M3, W2-M4 landed via the
2026-09-11 wave and were re-verified against this tree by the 2026-09-15
scope pass (per-item evidence in `SCOPE-2026-09-15.md`
§Cleared-Item-Evidence); residuals below.

| ID | Scope | Acceptance evidence |
| --- | --- | --- |
| ~~W2-D6~~ | **LANDED 2026-09-15** (`5501ec7c1`): 10 new monkeypatched offline tests (12 → 22 total in `tests/pipeline/test_health_check.py`) cover all eight scoped branch areas — julia timeout, partial integration via enhancer failure, scoring bands incl. the 90.0 boundary, recommendations, `main()` exit codes 0/1/2 + `--output-file`, verbose printing; the `:15-16` header comment was settled as accurate, not stale. 22/22 green. | `uv run --extra dev python -m pytest tests/pipeline/test_health_check.py -q` → 22 passed. |
| ~~W2-M2~~ | **LANDED 2026-09-15** (`d79c62756`): the three pinning tests added (`.yaml`/`.yml`/`.json` suffix dispatch parity, error-level parse-failure log naming the path strengthened, unregistered-stem warning pin verified). | `tests/pipeline -q -k "config or resolve or yaml or output_dir"` → 63 passed. |
| ~~W2-J1~~ | **LANDED 2026-09-15** (`d79c62756`): one fallback policy in the helper (documented: caller-supplied directory on standalone use); both wrappers are now thin delegates (`framework_common.resolve_execution_dir`, `gui/runner.resolve_output_root`); both divergent `except ImportError` blocks deleted; call-site parity pinned over `output/`, `output/7_export_output/`, arbitrary subdirs; the GUI test that pinned the old fallback rewritten to the fail-loud contract. | `tests/pipeline tests/analysis -q` → 967 passed; `tests/gui/test_gui_composability.py` → 21 passed. |

Out-of-scope observations (recorded, not scoped): ~~`PipelineContext`
(`src/gnn/pipeline/context.py`) is dead weight~~ **RESOLVED 2026-09-15**
(`c89452e00`, SCOPE-2026-09-15.md N-4, decision: delete): the corrective
census falsified the "closed surface" premise — `StepStatus` was the live
coupling (`pipeline/schemas.py:10`, `intelligent_analysis/processor.py:19`,
production chain step 24) — so `StepStatus` was re-homed to
`pipeline/schemas.py`, the dead `PipelineContext`/`StepRecord` pair and
`context.py` were deleted (census-verified zero production importers), and
the doc claims were tombstoned. AGENTS.md "Data Dependencies" graph still
overstates in-memory propagation (steps 5/6/8/10/11/13 re-parse input; only
3→7 and 11→12 consume artifacts). `run_session`/durable streams are not
wired into main.py composition (feature gap, not a test gap). The remaining
two need a design decision, not a mechanical fix.

Scope evidence (re-verified 2026-09-15): the coverage floor is
`fail_under = 60` (raised 50 → 60, MAJ-T1, 2026-09-10), and
`tests/pipeline/test_pipeline_overall.py` is deleted (W2-D7, 2026-09-11).
~~`PipelineContext` remains production-unwired... removal is mechanical once
the delete-vs-wire-in decision is made~~ **EXECUTED 2026-09-15** (`c89452e00`,
N-4): deleted with its test and exports; `StepStatus` re-homed to
`pipeline/schemas.py`; see the observations note above.

Test hygiene (discovered 2026-09-17, open): `tests/test_manuscript_token_gate.py`
regenerates `output/data/manuscript_variables{,_receipt}.json` against the
REAL repo tree as a run side effect — a local combined pytest run then lets
the rewritten map (describing the new HEAD) cross into a same-session
custody assert in `test_manuscript_latex_log.py`, producing a false
map-vs-manifest failure (and any subsequent `git add -A` sweeps the
pollution into a commit; `9fc63b1db` needed the SC-22 rerun at `23bef35b5`
to repair exactly this). Fix: the token-gate suite must regenerate into
`tmp_path` copies (or restore HEAD state in a teardown), never the live
committed paths.

## Deep horizon wave 2 - render backends

Scoping for `src/gnn/render/**` (2026-09-08, deep-horizon session). RB-01
through RB-09 are RESOLVED (see `CHANGELOG.md`): the deterministic corpus x
framework conformance benchmark (`bash autoresearch.sh`;
`scripts/bench_render_backends.py`) drove contract modernization to the
maintained delegated-executor output shapes, and the conformance-validated
rendering count moved 142 -> 258 of 258 (0 contract violations, 0 syntax
errors, 0 render errors, 12 by-design unsupported). RB-08 deleted the
retired `toml_generator.py` emitter, migrating its live matrix parsers and
topology contract helpers to `src/gnn/render/rxinfer/model_contracts.py`.
RB-09 recorded the generator-facade decision: the
`generate_rxinfer_code` / `generate_activeinference_jl_code` exports are
the supported public surface, pinned end to end by
`tests/render/test_generators_coverage.py`; `pymdp_template.py` stays with
its only production caller (the non-POMDP basic fallback). Nothing remains
open in this subsection.

---


## Deep horizon wave 2 - MCP + execute

Scope from a six-lens read-only audit (registry, dispatcher/serialization,
subprocess envelope, per-framework executors, resources/docs, test gaps) of
`src/gnn/mcp/**` and `src/gnn/execute/**`. Session benchmark:
`bash autoresearch.sh` (`scripts/run_autoresearch_bench.py`, primary metric
`mcp_execute_bench_ms`, 1208 pinned determinism checks) — every row below
must leave it green. Segment-2 optimizations (PRs #79, #80, #84, #85, #87)
improved the primary metric from 1555.3 ms → 1288.1 ms (-17.2%) while
adding a GNNParsingSystem round-trip phase that exercises the 23-parser /
22-serializer system.

Completed rows (file:line evidence in PR descriptions):
- MAJ-08 ✓ (PR #65): shared `serialize_response` choke point — unserializable
  tool results no longer hang stdio or abort HTTP; NaN/Inf → canonical tokens;
  cache-key type-tag de-aliasing
- MAJ-09 ✓ (PR #65): `MCPTool.timeout` enforced via dedicated bounded pool,
  new wire code -32008
- MAJ-10 ✓ (PR #70): step-12 processor migrated onto canonical envelope;
  shared exit-code sentinels; kill+drain partial output; sandbox/julia_setup/
  lean delegated
- MED-01 ✓ (PR #66): signature mismatches → -32602; None output allowed;
  `validation_mode` in capabilities
- MED-02 ✓ (PR #76): real `list_available_resources`; HTTP gate/capability
  agreement; docs drift fixed; dead `npx_inspector.get_resource` routed
- MIN-02 ✓ (PR #73): 12 registry-internals tests
- MIN-03 ✓ (PR #73): envelope `input=` support; 141/141 schema-vs-signature
  audit

Remainders — ALL LANDED (receipts in SCOPE-2026-09-11.md §Wave 2 and
CHANGELOG 2026-09-11; re-verified 2026-09-15):
- MED-03a ✓ (PR #97): executor timeout alignment (60→3600/600), pymdp
  self-heal (.cleaned.py + discovery filter + 5 tests), lean temp-dir
  leak, rxinfer TOML --project=
- MED-03b ✓ (RxInferPersist): rxinfer execution evidence persistence —
  `{stem}_stdout.txt` / `{stem}_stderr.txt` / `{stem}_execution_log.json`
  written on every run (`src/gnn/execute/rxinfer/rxinfer_runner.py:66-69`,
  `:168-170`, `:190-192`; documented in `execute/rxinfer/AGENTS.md:60`)
- MED-04 ✓ + MIN-01 ✓ (MCPHygiene): HTTP body cap + 400/413 envelopes,
  stdio bounded read, response-size policy; dead duplicate-registration
  check and dead -32700 branch removed, cache deepcopy, ensure_ascii
  parity

The 10-row scoped table below was fully landed (PRs #65, #66, #70, #73,
#76, #97) and is removed from this file per the Conventions; audit trail:
CHANGELOG.md, git history, and the PR descriptions.


## v4.0.0 - Bounded Autonomy & Reviewed Self-Editing

The local bounded-autonomy surface emits proposal-only artifacts via
`--autonomous`: candidate scores, review gates, rollback descriptors, audit
events, and non-mutating security policy. No source edit, commit, container
run, or cluster mutation is automatic.

Concrete, cold-startable work is scoped in the open tables above (the
W2-D6/M2/J1 residuals; the validate_gnn* / SC-38-tail / V4-STAGE /
paired-repin rows in the 'Still open (residuals)' table) and in
`SCOPE-2026-09-15.md` §Improvements; this section records the unscoped
vision and the current proposal-only `--autonomous` surface.

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

## Remaining from SCOPE-2026-09-09/10 (post-scope2 truth state)

The SCOPE-2026-09-09 program closed on PR #109 and SCOPE-2026-09-10 executed
on 2026-09-10 (waves A/B/C, evidence per item in `SCOPE-2026-09-10.md`).
SCOPE-2026-09-11 executed on 2026-09-11: the utils/ concern-package split
Steps 0-6 (testing/, arguments/, pipeline_orchestration/, runtime_safety/,
observability/, mcp/ packages behind DeprecationWarning facades; lazy PEP 562
family inits where eager variants created import cycles — observability/,
runtime_safety/, mcp/), the four V4-STAGE consolidation slices (in-process
timeout+tee capture, whitelist {0,3,5,7,8,11}, parsed-model carrier,
parallel-tier dispatcher), V4-HD (`_run_factorized_sweep` routes through
`execute_kronecker_factorized`), MED-T4 coverage (cross_format / types /
type_systems / round-trip), the stale W2 rows (W2-D4 default flip, W2-D5
preflight wiring, W2-D6 step-count, W2-D7 wiring tests, W2-M4 stale
commands), MED-03b (rxinfer evidence persistence), MED-04 + MIN-01 (transport
limits, cache copies, ensure_ascii parity), the validate_gnn* deprecation
window, the `_level_rank` name-fallback fix, comment-polish, and the
SC-22-hosted custody re-render cron
(`.github/workflows/custody-re-render.yml`). GEO-INFER state: GNN-04 paired
interchange CI (pin `.github/gnn-pair.json` @ GEO `c0115779`) and GNN-05
notation-driven metadata (`src/gnn/export/notation_metadata.py`,
`--geo-derive-metadata`, 27 tests) are DELIVERED; both paired workflows
(fep-lean + GEO interchange) are green on hosted main
(runs 34562055364 / 34562055373, 2026-09-11).

Still open (residuals, in rough order):

| ID | Scope | Acceptance evidence |
| --- | --- | --- |
| validate_gnn* retirement | Window OPENED 2026-09-11 (target v4.0.0, current 3.4.0): all 10 alias sites emit `DeprecationWarning` (`stacklevel=2`) naming the canonical replacement + "will be removed in v4.0.0"; manifest canonical/old-name inversion fixed; warning emission pinned in `tests/test_validate_surface_aliases.py`. SRC/DOC CALLER MIGRATION COMPLETE 2026-09-14: zero live callers remain across src/gnn, scripts/, and maintained docs (audited against every alias name incl. package-root lazy exports). | Retirement in v4.0.0 = delete the alias defs, their pins in `tests/test_validate_surface_aliases.py`, and any registry entries (no remaining src/doc callers to migrate). |
| SC-38 tail (remainder) | Families `errors/`, `config_io/`, `system_env/` EXTRACTED 2026-09-12 (§5 Step 7; DeprecationWarning facades at old paths; `_EXPORT_MAP` values repointed, keys frozen at 113; consumers + tests migrated). Logging single-entry contract LANDED 2026-09-14: `base_processor.py` repointed to the `gnn.utils.logging_utils` facade; third forbidden-import contract live (`lint-imports` 3/3, two documented pre-split-tree ignores). `simulation_utils` RESOLVED 2026-09-15 as production-dead removal (R4's extraction question mooted: zero non-test importers, zero `_EXPORT_MAP` keys; module + DiagramAnalyzer deleted, pyproject forbidden-modules entry dropped) — §3.8 table updated. Remaining per design §8: facade deprecation-window end only (delete old paths, v4-gated). | `lint-imports` 3/3 contracts kept; migrated suites green (332 tests in `tests/utils`). |
| V4-STAGE limits | Consolidated executor now covers stems {0,3,5,7,8,11} with timeout/tee/carrier on serial + parallel tiers; remaining limits recorded in ADR 0001: in-process steps cannot be force-killed, stem 9 excluded (D2 CLI shell-out), matplotlib caveat in thread-pool tier. | Each landed slice pinned by parity tests in `tests/pipeline/`. |
| paired-repin discipline | The fep_lean source-pin seals GNN owner digests at pin time; ANY later owner-file edit re-drifts the pair (3 drift cycles documented on PR #110). Standing closeout ordering: all content edits → token ritual → bridge re-pin → fep_lean PR/merge → pair-pin bump as the FINAL commit, single push. Canonical ordering: `docs/development/fep_lean_paired_revision.md`. | `fep-lean bridge status --gnn-root .` green at the pin; zero post-bump pushes. |

Closed 2026-09-12 (families + custody campaign): SC-22-hosted — the custody
cron is proven end-to-end at full fidelity (pandoc + pinned
pandoc-crossref v0.3.25, zero `not on PATH` warnings; run 34660573491) AND
now certifies the committed chain via `verify_fresh_render` FAIL/WARN
semantics (`scripts/z_verify_fresh_render.py` runs between render and
re-record; 4 new tests in `tests/test_manuscript_latex_log.py`).

## Deep horizon wave 2 - tests + CI

Scope owner: `deep/gnn-tests-ci` (verification layer: `tests/**`, `ci.yml`,
`full-extras.yml`, `pytest.ini`, `justfile`). Additions only - no existing
gate, test, or selection may be weakened. Measurement harness: `bash
autoresearch.sh` (baseline coverage_percent=60.12; 4343 passed / 0 failed /
7 skipped under `--extra dev --extra ml-ai --extra torch`, CI coverage-parity
selection `-m "not pipeline and not mcp"`, fixed `-n 4` xdist, offline).

Landed (audit trail in `CHANGELOG.md` "Deep horizon wave 2 - tests + CI"):
PR-time extras CI job + `just test-extras` (MIN-T1), xdist-safe tmp paths
(MIN-T2), zero-skip contract completeness over bare-marker and
`unittest.skip*` forms (MIN-T3), weekly full-extras coverage artifact
(MIN-T4), dispatcher/envelope negative-path tests plus the TimeoutExpired
str normalization fix (MED-T1), `validate_tools.main()` coverage (MED-T2),
alias error-parity + stacklevel pins (MED-T3), `pipeline_validation`
coverage plus the `naming_violations` crash fix, load-hardened environment
performance smoke, and the coverage floor raise 50 -> 60 (MAJ-T1).

MED-T4 (remainder) LANDED 2026-09-11: `tests/schema_validator/test_cross_format.py`
and `tests/types/` (38 tests) plus `tests/type_systems/test_init.py` and
`tests/testing/test_round_trip_{comparison,markdown_parser,report}.py`
(38 tests) cover the previously-0% surfaces; observable behavior only.

Verification commands: `bash autoresearch.sh` (full harness), targeted
`uv run --extra dev python -m pytest <file> -q`, `just lint`,
`uv run --extra dev mypy src/gnn --show-error-codes`.

Scope pass 2026-09-15 — per-item evidence in SCOPE-2026-09-15.md
§Cleared-Item-Evidence.
