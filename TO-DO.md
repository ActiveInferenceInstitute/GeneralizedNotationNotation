# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-09-07 (3.3.0 post-release hygiene: GNN-02/GNN-03 closed with evidence, GNN-04/GNN-05 narrowed, majors MAJ-04..07 scoped)
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

## Open Scoped Roadmap

Every item below is cold-startable: scope, files, verification, and acceptance
are pinned. Rough order: MAJ-07 (decision-first) -> MAJ-05 -> MAJ-06 -> MAJ-04
(largest; one module per PR).

| ID | Scope | Acceptance evidence |
| --- | --- | --- |
| MAJ-04 | Decompose the six >2000-line modules (`integration/meta_analysis/visualizer.py` 2871, `analysis/visualizations.py` 2412, `testing/test_round_trip.py` 2214, `render/jax/jax_renderer.py` 2200, `render/discopy/translator.py` 2150, `analysis/analyzer.py` 2031) following the 3.3.0 `execute/processor.py` split pattern (mechanical extraction into sibling modules, facade re-exports preserved, one module per PR), and extract the shared subprocess envelope the nine per-framework renderers duplicate. | Per module: no import path changes (old names still importable), `uv run --extra dev mypy src` clean, `just lint` and `just format-check` clean, module tests plus `just test` green, moved code byte-identical modulo import lines. |
| MAJ-05 | De-duplicate the `validate_gnn*` public surface - 6+ unrelated semantics share the name (`gnn/__init__.py` `validate_gnn_file`, `llm/llm_operations.py` `validate_gnn`, `parsers/basic.py` `validate_gnn` / `validate_gnn_syntax_formal`, `processing/processor.py` `validate_gnn_structure`, `mcp/processors.py` `validate_gnn_cross_format_consistency`, `execute/pymdp/pymdp_utils.py`). Rename to unambiguous names with deprecation aliases, one module per PR. | One unambiguous `def validate_gnn*` name per semantic; every old name re-exported with a `DeprecationWarning`; old-name and new-name tests pass; MCP tool registry unchanged. |
| MAJ-06 | Collapse the ~10 copy-pasted `process_<module>_mcp(target_directory, output_directory, verbose)` wrappers (`advanced_visualization/mcp.py`, `analysis/mcp.py`, `audio/mcp.py`, `execute/mcp.py`, `export/mcp.py`, `gui/mcp.py`, `integration/mcp.py`, plus the render variants) into one generic dispatcher with per-module registration. | Tools register under identical names/signatures (`just skills-health` green, the mcp-audit CI job green); per-module MCP tests pass; wrapper files shrink to registration calls. |
| MAJ-07 | Consolidate the pipeline health/utility cluster: `src/gnn/pipeline/health_check.py` (703 lines, zero direct tests; the CLI `health` command routes via `render.health` instead) and `src/gnn/pipeline/pipeline_validator.py` (467 lines; near-name collision with the tested `gnn.utils.pipeline_validator`, plus `pipeline_validation.py` and `verify_pipeline.py`). Decide per file - delete, or test and wire - and record the decision here. | If deleted: no inbound imports remain (`grep -rn "pipeline.health_check\|pipeline.pipeline_validator" src/gnn tests`) and `just test` green. If kept: `tests/pipeline/` covers the public functions and the live consumer chain is named. |

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
