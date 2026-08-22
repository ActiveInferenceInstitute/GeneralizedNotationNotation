# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-08-20 (v4.0.0 milestone close-out; roadmap refreshed)
**Current Version**: 3.0.0
**Next Target**: v4.0.0 (bounded autonomy, pipeline stage consolidation, multi-agent stigmergic topologies, and high-dimensional active inference)

**Last reviewed**: 2026-08-21 — MAJ-02 closed in full: milestone 1 (sparse
Kronecker factorized execution + scaling sweep) plus the numbered-pipeline
integration (Step 11 renders factorized specs natively for JAX, Step 12
collects `jax_kronecker_factorized_v1`, Step 16 analysis extracts it). MAJ-03
milestone 1 (native stigmergic multi-agent compilation) closed; the
env-conditioned action selection residual remains open below. The complete
test suite is **3,100+ passed / 0 failed / 0 skipped** (parallel, Ollama files
excluded per CI), mypy clean, ruff clean, and all documentation audits pass.
Full audit trail lives in `CHANGELOG.md` and git history.

## Open Scoped Roadmap

### Major (P1 - Generative Scaling & Multi-Agent Topologies)
- **TODO-MAJ-03 (residual): Env-Conditioned Action Selection for Stigmergic Swarms**:
  - *Status*: Milestone 1 landed 2026-08-20 — specs with >= 2 complete agent
    groups (`A_agentN`/`B_agentN`/`C_agentN`/`D_agentN`) render natively in
    both `src/render/rxinfer/` (`_strategies_multiagent.py`) and
    `src/render/activeinference_jl/`: one genuine model/inference per agent
    (no joint state-space expansion) coupled through the shared `env_signal`
    trace (deposit at MAP position, decay per timestep), stamped
    `model_kind: multi_agent`, with `env_signal_trace` in the results. Live
    Julia execution is pinned by `src/tests/render/test_stigmergic_multi_agent.py`.
  - *Open*: infer `env_signal` as a latent from observations and condition
    per-agent *action selection* on it (requires env-conditioned likelihoods
    the swarm exemplar does not currently declare; the shared affordance
    detection lives in `src/render/multi_agent_common.py`).
  - *Probe*: `uv run pytest src/tests/render/test_stigmergic_multi_agent.py -q`

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
