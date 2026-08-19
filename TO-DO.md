# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-08-19 (forward-looking open items only)
**Current Version**: 3.0.0
**Next Target**: v4.0.0 (bounded autonomy, pipeline stage renumbering/consolidation, multi-agent scaling, and declarative ontology synthesis)

**Last reviewed**: 2026-08-19 — all past TODO items, test-infrastructure follow-ups, and RED_TEAM_REVIEW.md security residuals are closed. The complete test suite is **3,051 passed / 0 failed / 0 skipped** (zero skips; Julia and Python frameworks fully provisioned and executed live), all 25 pipeline steps are composable and verified end-to-end, and all documentation/cognitive-phenomena examples pass. Full audit trail lives in `CHANGELOG.md` and git history.

## Open Scoped Roadmap

### Minor (P3 - Developer Ergonomics & Diagnostics)
- **TODO-MIN-01: Auto-Detect Environment GPU Acceleration in Execution Metadata**:
  - *Problem*: Hardware accelerator status is logged during setup but not yet recorded in per-model `execution_metadata.json` across all backend simulation runners.
  - *Scope*: In `src/execute/processor.py` and `src/execute/pymdp/pymdp_simulation.py`, include `accelerator_type` and device memory fields in per-model output metadata.
  - *Probe*: `uv run python -c "from execute.pymdp.pymdp_simulation import PyMDPSimulation; sim = PyMDPSimulation({}); res = sim.run_simulation(num_timesteps=1); assert 'execution_metadata' in res or 'hardware' in res"`
- **TODO-MIN-02: Structured CLI Output Schema Envelope (`--json`) for Remaining Commands**:
  - *Problem*: `gnn report`, `gnn health`, `gnn preflight`, and `gnn graph` CLI commands should be continuously tested across all subcommands to ensure full envelope parity.
  - *Scope*: In `src/tests/cli/test_cli_public_api.py`, add parameterized tests asserting the `{status, data, error, meta}` envelope schema across all CLI subcommands.
  - *Probe*: `uv run pytest src/tests/cli/test_cli_public_api.py -k json` asserts envelope structure on all commands.
- **TODO-MIN-03: Model Registry Ontology Query CLI Integration**:
  - *Problem*: `--query-ontology` filter is implemented in `src/4_model_registry.py` and `src/model_registry/registry.py`; expose this capability as a flag on `gnn models list` CLI command.
  - *Scope*: In `src/cli/__init__.py`, add `--query-ontology` parameter to model listing subcommands.
  - *Probe*: `uv run python -m cli --help` confirms presence and documentation of ontology query option.

### Medium (P2 - Pipeline Architecture, Step Renumbering/Ordering & Performance)
- **TODO-MED-01: Full Pipeline Step Renumbering and Directory Path Migration**:
  - *Problem*: Historically, Step 15 (Audio) and Step 16 (Analysis) run after Step 13 (LLM) and Step 14 (ML Integration). Renumbering them to contiguous simulation analytics (`13_audio`, `14_analysis`, `15_llm`, `16_ml_integration`) improves data locality.
  - *Scope*: Complete the migration of physical filenames from `15_audio.py` → `13_audio.py` and `16_analysis.py` → `14_analysis.py`, while maintaining `CONSOLIDATED_STEP_ALIASES` in `src/pipeline/step_registry.py` for continuous alias resolution.
  - *Probe*: `uv run python src/main.py --target-dir input/gnn_files/basics --output-dir /tmp/gnn-reorder-smoke --only-steps 11,12,13,14` verifies contiguous execution without missing artifact warnings.
- **TODO-MED-02: Streaming Multi-Modal Audio Sonification Buffer in Step 15**:
  - *Problem*: Audio sonification processes complete simulation trajectories in batch mode at the end of the run.
  - *Scope*: In `src/audio/sapf/` and `src/15_audio.py`, implement a chunked rolling synthesizer buffer that emits audio chunks per tick, synchronizing with `durable_streams.py`.
  - *Probe*: `uv run pytest src/tests/audio/test_audio_generation.py` validates chunked streaming synthesis.
- **TODO-MED-03: Dynamic Parallel Tier Worker Pool Auto-Scaling**:
  - *Problem*: Parallel mode (`--parallel`) currently uses dynamic CPU detection; enhance it with per-step memory profile estimates from `src/utils/pipeline_planner.py` to prevent OOM on memory-intensive steps.
  - *Scope*: In orchestrator execution (`src/pipeline/dag.py` and parallel dispatch), dynamically calibrate worker pool bounds based on available CPU count and per-step memory requirements from `src/utils/pipeline_planner.py`.
  - *Probe*: `uv run python src/main.py --target-dir input/gnn_files/basics --output-dir /tmp/gnn-parallel-scaled --parallel` executes with resource-calibrated worker count.

### Major (P1 - Bounded Autonomy, Generative Scaling & Multi-Agent Topologies)
- **TODO-MAJ-01: v4.0.0 Bounded Autonomy & Model Mutation Proposal Engine**:
  - *Scope*: Implement the reviewed self-editing loop in `src/pipeline/autonomous.py` where the pipeline analyzes Step 16 (Analysis) and Step 24 (Intelligent Analysis) metrics (e.g. uninformative observations, high state entropy, non-convergent policies) and generates proposed parameter adjustments (e.g. Dirichlet prior sharpening, matrix pruning).
  - *Safety Boundary*: Proposals are emitted purely as non-mutating artifacts in `output/proposals/` (`proposal_manifest.json` and unified diff patches); no source overwrites, git commits, or external execution occur without operator authorization.
  - *Probe*: `uv run python src/main.py --autonomous --output-dir /tmp/gnn-autonomous-smoke` writes deterministic proposal bundles with verified score diffs and rollback manifests.
- **TODO-MAJ-02: High-Dimensional Kronecker Factorization in JAX PyMDP Backends**:
  - *Scope*: Scale multi-factor discrete active inference models beyond 2 factors using sparse Kronecker factorizations in JAX PyMDP backends, enabling N-factor POMDP exploration for large state spaces ($N \ge 64$).
  - *Probe*: `scripts/run_pymdp_gnn_scaling_analysis.py` successfully completes scaling runs for $N \ge 64$ states across factorized topologies.
- **TODO-MAJ-03: Declarative Multi-Agent Stigmergic Interaction Compiler**:
  - *Scope*: Add native multi-agent stigmergic communication compilation in `src/render/rxinfer/` and `src/render/activeinference_jl/`, where agents interact via shared environmental affordances without requiring global joint state space expansion.
  - *Probe*: `uv run pytest src/tests/render/test_rxinfer_model_strategies.py` verifies native multi-agent stigmergic compilation and execution.

---

## v4.0.0 - Bounded Autonomy & Reviewed Self-Editing

The local bounded-autonomy surface emits proposal-only artifacts via
`--autonomous`: candidate scores, review gates, rollback descriptors, audit
events, and non-mutating security policy. No source edit, commit, container
run, or cluster mutation is automatic.

---

## Verification Commands

Use `uv run` for roadmap verification checks:

```bash
PYTHONPATH=src uv run python scripts/run_v3_orchestration_acceptance.py --strict
PYTHONPATH=src uv run python scripts/emit_run_manifest.py output --out /tmp/gnn-v3-run-manifest
PYTHONPATH=src uv run python scripts/generate_pipeline_container_plan.py --config input/config.yaml --out /tmp/gnn-v3-container-plan.json
PYTHONPATH=src uv run python scripts/run_session_acceptance.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-v3-session-acceptance --session /tmp/gnn-v3-session.json --strict
PYTHONPATH=src uv run python src/main.py --autonomous --target-dir input/gnn_files --output-dir /tmp/gnn-autonomous-smoke

uv run python doc/development/docs_audit.py --strict --check-anchors --no-write
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
