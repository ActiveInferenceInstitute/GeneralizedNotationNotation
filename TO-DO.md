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

## Open Scoped Roadmap

Every item below is cold-startable: scope, files, verification, and acceptance
are pinned. Rough order: MAJ-05 -> MAJ-06 -> MAJ-04 (largest; one module
per PR).

| ID | Scope | Acceptance evidence |
| --- | --- | --- |
| MAJ-04 | Decompose the six >2000-line modules via the 3.3.0 `execute/processor.py` split pattern (mechanical extraction into sibling modules, facade re-exports preserved, one module per PR). **LANDED 2026-09-07 (deep-horizon session, 4/6):** `analysis/visualizations.py` 2412→58 (PR #29), `analysis/analyzer.py` 2031→263 (PR #32), `render/jax/jax_renderer.py` 2200→170 (PR #33), `render/discopy/translator.py` 2150→303 (PR #34). **REMAINING:** `integration/meta_analysis/visualizer.py` 2871 — class-method split must preserve byte-identity, so extract `Sweep*PlotMixin` siblings holding verbatim method blocks (runtime/metric/summary/export seams; facade keeps `__init__`, `generate_all`, `_safe_log_scale`, and the `_MPL_AVAILABLE` binding that `tests/integration/test_integration_meta_analysis_validation.py:287-289` monkeypatches); `testing/test_round_trip.py` 2214 — extract config dicts, result dataclasses, `_DirectMarkdownParser`, `_compare_*` helpers, and report writer into `round_trip_*` siblings (non-`test_` names so explicit-path collection is unchanged); facade keeps availability flags, `sys.path.insert`/`setrecursionlimit` side effects, tester core, unittest class. **RESCOPED:** the "shared subprocess envelope the nine per-framework renderers duplicate" — renderers contain zero subprocess code (verified); the duplication is execute-side (`rxinfer/stan/lean/activeinference` runners + 4 `executor.py` MCP methods vs the canonical `execute_script_safely` at `execute/executor.py:1089-1200`) and needs a behavior-preserving refactor with its own tests, not a mechanical split. | Landed modules: no import-path changes (per-module facade-contract probes: 31/34/27/36 names importable), mypy 0 errors, `ruff check src/gnn scripts` clean, targeted module tests green, full suite 4263 passed / 0 failed, moved code byte-identical modulo import lines (2347/2001/2190/2108 verified per module). Session benchmark `oversized_module_lines` 13878 → 5085 (-63.4%). |


### Smaller scoped cleanups (independent of the majors)

- `setup_step_logging` residue: RESOLVED 2026-09-07 where it was real -
  the `gnn.utils.pipeline` delegate and `utils/migration_helper.py` fossil
  are gone; `utils/logging_utils.py` stays (documented facade entry with
  its own tested `PipelineLogger`; retiring it is a rename-class change).
- Audit the 87 `ruff --select F401,F811` findings (currently
  policy-ignored in `pyproject.toml` with an "optional deps,
  import-or-skip probes" rationale): split the global ignore into
  per-file-ignores that keep the guarded optional-dependency probes and
  facade re-exports while removing genuinely dead imports
  (e.g. `src/gnn/parsers/*`, `src/gnn/api/app.py`).
- Local/CI parity: tokens and skills-health are CI-wired via
  `.github/workflows/local-gates.yml` (2026-09-07; `skills-health` also
  needed a repo-root sys.path bootstrap). `just gridworld` remains
  unwired deliberately - the committed `output/` tree currently fails
  its contract and regeneration needs the Julia toolchains. Still open:
  the `ml-ai`/`torch` extras would unlock 12 environment-skipped tests
  (11 sklearn, 1 torch); `test-cov` locally ignores the Ollama tests
  while the CI coverage run does not.
- Dependency floors: RAISED 2026-09-07 for numpy (>=2.0), pandas
  (>=2.0), openai (>=2.0), pytest (>=8.0), mypy (>=1.0) - the lock
  resolved identically (only requires-dist metadata moved; zero package
  pins changed). Remaining cosmetic floors (networkx 2.6, plotly 5.15,
  scipy 1.7, ...) can follow at the next deliberate lock refresh.
- `gnn/utils/pipeline_validator.py` vs `gnn/pipeline/pipeline_validator.py`
  near-name collision (recorded during the MAJ-05 validate-surface pass;
  unrelated to the `validate_gnn*` function surface, which is resolved):
  audit both modules' roles and repo-wide consumers, then rename the
  lower-traffic module to an unambiguous name with a compatibility re-export
  of the old import path. Verify: import-site grep updated with zero
  stragglers, `uv run --extra dev mypy src` clean, MCP tools and CLI paths
  unchanged, module tests green.
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
