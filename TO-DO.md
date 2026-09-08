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

## Deep horizon wave 2 - tests + CI

Scope owner: `deep/gnn-tests-ci` (verification layer: `tests/**`, `ci.yml`,
`full-extras.yml`, `pytest.ini`, `justfile`). Additions only - no existing
gate, test, or selection may be weakened. Measurement harness: `bash
autoresearch.sh` (baseline coverage_percent=60.12; 4343 passed / 0 failed /
7 skipped under `--extra dev --extra ml-ai --extra torch`, CI coverage-parity
selection `-m "not pipeline and not mcp"`, fixed `-n 4` xdist, offline).

| ID | Scope | Acceptance evidence |
| --- | --- | --- |
| MIN-T1 | PR-time extras job in `ci.yml` (py3.12): `uv sync --frozen --extra dev --extra ml-ai --extra torch`, then run the two extras-unlock files (`tests/ml_integration/test_ml_integration_inference.py`, `tests/render/test_continuous_renderers.py`) with `-q`. Today those 12 tests run only in the weekly scheduled full-extras workflow. Mirror the invocation as a `just test-extras` recipe. | New job green on PR; existing `test`/`security` jobs untouched; `just test-extras` recipe present and green locally. |
| MIN-T2 | xdist-safe temp paths: `tests/render/test_jax_factorized_pipeline.py:285-289` (fixed `/tmp/gnn_test_analysis_model_jax.py`, `/tmp/gnn_test_analysis_out`) and `tests/api/test_comprehensive_api.py:303-304` (fixed `/tmp/test_output`) move to pytest `tmp_path`. | No fixed `/tmp` write targets remain in those files; assertions unchanged; suite green under `-n 4`. |
| MIN-T3 | Zero-skip contract completeness: `tests/execute/test_lean_runner.py` uses `pytest.mark.skipif` but is absent from `DEFAULT_SKIP_ALLOWLIST` (`tests/test_zero_skip_contracts.py:24-40`); `tests/visualization/test_d2_visualizer.py` evades `FORBIDDEN_SKIP_TOKENS` via `unittest.skipIf`. Extend the contract to enumerate every skip site (including `unittest.skip*`) and reconcile the allowlist with justifications. | `uv run --extra dev pytest tests/test_zero_skip_contracts.py` green; contract covers `unittest.skipIf`/`skipUnless`; no new skips introduced. |
| MIN-T4 | `full-extras.yml` weekly run gains `--cov=gnn --cov-report=json:junit/coverage-full-extras.json` so weekly-only paths (`pipeline`/`mcp`-marked, audio/gui/research code) appear in coverage reports. | Weekly (dispatch-triggered) workflow green; coverage JSON artifact produced alongside the existing JUnit artifact. |
| MED-T1 | Dispatcher/envelope negative-path tests: `src/gnn/utils/mcp_dispatch.py` - `message_builder` on the failure branch (:92-93), `static_extras` dropped when the step raises (:100-102), non-Mapping `build()` result through `run_tool_envelope` (:118), `BaseException` (e.g. `KeyboardInterrupt`) passthrough (:97,:116); `src/gnn/execute/subprocess_envelope.py` - `TimeoutExpired` captured-stream keys and `capture_output=False`+timeout (:105-107). Extend `tests/utils/test_mcp_dispatch.py` and `tests/execute/test_subprocess_envelope.py`. | Every listed branch is exercised by a test that fails on a plausible regression; `just lint` and `uv run --extra dev mypy src/gnn` clean. |
| MED-T2 | `src/gnn/mcp/validate_tools.py` `main()` has zero test coverage: MCP init failure exit 1 (:46-50), NOT_CALLABLE/UNDOCUMENTED issue classification (:113-124), spot-check SKIP/issue/exception branches (:155-175), `logging_miss` branch (:185-196). New `tests/mcp/test_validate_tools.py` (offline, no `mcp` marker so the default selection runs it). | Branches above covered; `bash autoresearch.sh` coverage_percent increases; suite green. |
| MED-T3 | `validate_gnn*` alias hardening: error-path equivalence with the canonical function on invalid inputs; package-root `gnn.validate_gnn_syntax_formal` deprecation pin (`src/gnn/__init__.py:27,:71,:169`); `DeprecationWarning` stacklevel pins for all 8 aliases. Extend `tests/test_validate_surface_aliases.py`. | Alias-vs-canonical error parity asserted; stacklevel pinned; suite green. |
| MED-T4 | Unit tests for currently untested `src/gnn` subpackages, priority order: `schema_validator`, `types`, `type_systems` (census: no dedicated test files). Tests assert observable behavior, not implementation. | New test files per package; harness coverage_percent measurably higher than baseline; suite green. |
| MAJ-T1 | Coverage floor: after MED-T1..T4 land, raise `[tool.coverage.report] fail_under` from 50 to a measured margin below the achieved harness coverage, pinning the improvement as a gate. | CI coverage job green at the new floor; floor value documented in `CHANGELOG.md`. |

Verification commands: `bash autoresearch.sh` (full harness), targeted
`uv run --extra dev python -m pytest <file> -q`, `just lint`,
`uv run --extra dev mypy src/gnn --show-error-codes`.
