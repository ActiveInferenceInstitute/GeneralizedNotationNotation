# TO-DO — GNN Pipeline Roadmap

**Last Updated**: 2026-06-12
**Current Version**: 2.0.0
**Next Target**: v3.0.0 (long-running orchestration, durable streams, and auditable container plans)

**Current Evidence (2026-06-12)**: v2.0.0 semantic fidelity gate passed for 9 families (`gnn_semantic_fidelity_ledger_v1`); cross-framework reliability gate passed for 9 families (`gnn_cross_framework_reliability_ledger_v1`) with GridWorld compared PyMDP, RxInfer, and ActiveInference.jl and all other unprofiled backends recorded with explicit unsupported statuses. Command-of-record collect-only inventory is `2411` collected tests across 186 `test_*.py` files with the documented Ollama integration ignores. Latest full local suite evidence with the same Ollama ignores is `2393 passed, 17 skipped, 1 xfailed`. v1.9 all-family strict acceptance remains green for 9 families; continuous and hierarchical Step 11/12 remain profiled unsupported skips with `0` raw failed Step 11/12 counts. v1.8.0 focused release smokes passed for `gnn templates list`, `gnn templates show pomdp-gridworld-3x3`, dry-run `gnn pull` to `/tmp/gnn-pull`, and authenticated
MCP HTTP tests (`12 passed`; combined CLI/MCP/capability suite `32 passed`); `just lint` passes.

---

## ✅ v1.6.0 — Real-Implementation Stabilization & Core Infrastructure (Released)

> **Released**: 2026-04-15 (tag: `v1.6.0`)
> **Scope**: Production hardening, MCP integration, renderer expansion, documentation integrity.

- [x] **NumPyro/Stan Renderer Integration** — Complete end-to-end render pathway for NumPyro and Stan alongside existing PyMDP, RxInfer, JAX, DisCoPy, PyTorch, ActiveInference.jl backends. Use current backend tests before publishing operational pass counts.
- [x] **MCP Full Module Exposure** — All 25 pipeline modules + infrastructure modules expose tools via MCP files. Current audit coverage is tracked by `src/tests/mcp/test_mcp_audit.py`; the 2026-05-14 focused audit registered 133 tools and 1 resource.
- [x] **PyMDP Scaling Study** — Automated scaling analysis pipeline (`scripts/run_pymdp_gnn_scaling_analysis.py`) with configurable N=[2,256] grids, exponential state-space sweeps, and 19-artifact visualization suite.
- [x] **Test Suite Hardening** — Real-implementation coverage across all modules. Current collect-only inventory is tracked in `src/tests/TEST_SUITE_SUMMARY.md`; Hypothesis tests were refactored to deterministic parametric matrices.
- [x] **Documentation Integrity** — Maintained documentation coverage is enforced by `doc/development/docs_audit.py --strict --check-anchors --no-write`, GNN doc-pattern checks, maintained-doc terminology checks, and repo terminology checks. Zero phantom file references at the latest recorded verifier pass.
- [x] **Enhanced Visual Logging** — Progress bars, color-coded output, structured summaries, correlation ID tracking, screen reader support across all 25 pipeline steps.
- [x] **LLM & ML Fixes** — LLM recursive glob fix, ML cross-validation fold logic hardened (`min(5, len(X), min_class_count)`).

---

## 🧱 v1.7.0 — Foundation Track (Retired as Release Target)

> **Scope**: Push the pipeline from single-agent generation to interactive, multi-agent architectures with real-time editing and streaming capabilities.
> **Status**: Retired as a standalone release target. Foundation contracts landed in later release branches; runtime-depth ambitions moved to v2+ reliability/orchestration milestones.

- [ ] **Multi-Agent Message Passing (RxInfer)** — Expand the `execute/` layer to handle clustered topologies (100+ agents) passing states asynchronously utilizing graph factorization in Julia via RxInfer.jl.
- [ ] **Categorical Symmetries (DisCoPy)** — Sync matrix permutations natively to string diagrams, allowing visual topology validation before simulation generation.
- [ ] **Reactive WebSocket GUI** — Overhaul the GUI stack (Step 22) into a WebSocket-powered frontend allowing users to adjust agent matrices on the fly without pipeline re-execution. Build on the oxdraw WebSocket architecture.
- [ ] **Audio Parameter Streaming** — Bridge Step 15 (Audio/Pedalboard/SAPF) to accept dynamic telemetry updates from long-running PyMDP agent simulations in real time. Extend the existing `process_realtime_chunk` pattern.
- [ ] **3D Matrix Visualization** — Upgrade the Matrix Visualization module into interactive Three.js canvas structures for explorable generative model inspection.

Contract foundations now exist for these items: compact RxInfer agent
population keys, DisCoPy permutation metadata, GUI WebSocket message schemas,
audio telemetry chunk artifacts, and Three.js tensor explorer HTML artifacts.
The items stay unchecked until backend execution, UI runtime behavior, and
optional-framework integration are verified end to end.

Review grouping for the current branch:
docs/verifiers; RxInfer/DisCoPy; GUI/audio/visualization; CLI/MCP/autonomy;
setup/scaling/test stabilization.

### Acceptance
```bash
uv run --extra dev python scripts/check_capability_contracts.py --strict
uv run --extra dev python src/main.py --only-steps "11,12" --frameworks "rxinfer,discopy" --target-dir input/multi_agent_models --output-dir /tmp/gnn-v17-multi-agent-accept --verbose
uv run --extra dev python -m pytest src/tests/audio src/tests/gui src/tests/render/test_rxinfer_multiagent_contract.py src/tests/render/test_discopy_symmetry_contract.py -q
```

---

## ✅ v1.8.0 — Developer Kit & Template Ecosystem (Released)

> **Scope**: Standardizing GNN as the definitive orchestration language with developer-grade tooling and reusable template packages.
> **Released**: 2026-06-12 (tag: `v1.8.0`)

- [x] **GNN Template Library Engine** — Enable package-manager style downloads for specialized active-inference setups directly using `gnn pull [template_name]` via CLI (Step `src/cli/`). Maintained packaged template index, `gnn templates list`, `gnn templates show`, `gnn pull`, dry-run, checksum, collision, overwrite, wheel/install smoke, and index path-safety contracts are covered by focused release tests.
- [x] **Pre-commit Ecosystem** — Ship `.pre-commit-config.yaml`, `justfile`, `.devcontainer/` (Dockerfile + devcontainer.json), Ruff lint/format hooks, and general file-hygiene checks to make repository contributions more consistent. Dedicated secret-scanning is not currently claimed by this item.
- [x] **MCP Local HTTP Orchestration** — Extend MCP server from local tool discovery to authenticated local JSON-RPC HTTP orchestration with bearer-token auth, rate limiting, localhost default binding, safe-tool exposure, and default-denied resource reads by default. Missing/invalid tokens return `401`, configured rate limits return `429`, and resource reads are denied unless explicitly allowlisted.

### Acceptance
```bash
uv run gnn templates list
uv run gnn templates show pomdp-gridworld-3x3
uv run gnn pull pomdp-gridworld-3x3 --output-dir /tmp/gnn-pull --dry-run
GNN_MCP_TOKEN=local-dev-token uv run --extra dev python -m pytest src/tests/mcp/test_mcp_http_auth.py -q
just lint
```

---

## ✅ v1.9.0 — Model-Family Reliability & Interpretability (Released)

> **Scope**: Make broader families of generative models reliably traverse the maintained pipeline with validation, execution-status, telemetry, interpretability, and report evidence.
> **Released**: 2026-06-12 (tag: `v1.9.0`)

- [x] **Model-Family Acceptance Harness** — Maintain `input/model_family_manifest.json` and run representative basics, discrete, continuous, hierarchical, multi-agent, precision, structured, gridworld, and scaling-study fixtures through pipeline evidence steps with explicit passed/skipped/failed statuses. v1.9.0 focused family/report suite `17 passed`; all-family strict acceptance passed for 9 families.
- [x] **Cross-Step Evidence Ledger** — Link Step 3/5/6/11/12/15/16/23 evidence for each accepted family: parsed model identity, matrix dimensions, renderer status, execution status, telemetry, analysis, visualization, report artifacts, and artifact links. Continuous/hierarchical Step 11/12 recorded as profiled unsupported skips with `0` raw failed Step 11/12 counts.
- [x] **Interpretability Summaries** — Emit per-family variable/edge inventories, matrix-shape tables, telemetry presence, optional observation/action/free-energy trace previews, renderer/execution status, and artifact links.
- [x] **Release Readiness Ledger** — release gates passed for focused family/report tests, all-family strict acceptance, collect-only inventory, full suite evidence, maintained docs/verifier gates, and tracked `output/` cleanliness.

Release evidence: the all-family strict harness parses real pipeline summaries,
fails closed when summaries or required per-step records are missing, rejects
incomplete `--only-steps` acceptance profiles, clears stale per-family outputs
before each run, and requires concrete artifacts for selected evidence steps.
`continuous` and `hierarchical` profile Step 11/12 as unsupported and skip those
steps before execution; failed Step 11/12 summaries are rejected by strict
negative controls instead of converted to success.

### Acceptance
```bash
uv run --extra dev python -m pytest src/tests/pipeline/test_model_family_acceptance.py src/tests/analysis/test_interpretability_summary.py src/tests/report/test_model_family_report.py -q
uv run --extra dev python scripts/run_model_family_acceptance.py --manifest input/model_family_manifest.json --families basics,discrete,multiagent,structured --output-dir /tmp/gnn-family-acceptance --strict
uv run --extra dev python scripts/run_model_family_acceptance.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-family-acceptance-all --strict
uv run --extra dev python src/main.py --target-dir input/gnn_files/discrete --output-dir /tmp/gnn-v19-discrete-smoke --skip-steps "2" --skip-llm --verbose
```

---

## ✅ v2.0.0 — Semantic Fidelity & Cross-Framework Reliability (Released)

> **Scope**: Upgrade GNN from broad fixture acceptance to stronger semantic preservation, cross-format round trips, and cross-framework equivalence checks.
> **Released**: 2026-06-12 (tag: `v2.0.0`)

- [x] **Semantic Round-Trip Gates** — Require representative model families to preserve variables, edges, dimensions, parameter shapes, equations, time, and ontology mappings across the maintained strict JSON interchange path. `scripts/run_semantic_fidelity_gate.py` passed for all 9 manifest families and wrote `gnn_semantic_fidelity_ledger_v1` artifacts.
- [x] **Cross-Framework Result Comparisons** — Compare compatible backend outputs through `scripts/run_cross_framework_reliability.py`; required backends need Step 11/12 evidence, successful non-skipped Step 12 execution detail rows, current simulation payloads, matching seeds when present, trace lengths, and matrix-shape/provenance parity. The all-family gate passed for 9 families; GridWorld compared PyMDP, RxInfer, and ActiveInference.jl, while JAX, NumPyro, PyTorch, and DisCoPy remain explicit unsupported statuses unless profiled for a compatible family.

### Acceptance
```bash
uv run --extra dev python -m pytest src/tests/pipeline/test_semantic_fidelity_gate.py src/tests/pipeline/test_cross_framework_reliability.py -q
uv run --extra dev python scripts/run_semantic_fidelity_gate.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-semantic-fidelity --strict
uv run --extra dev python scripts/run_cross_framework_reliability.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-cross-framework --strict
uv run --extra dev python scripts/run_model_family_acceptance.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-family-acceptance-all --strict
```

---

## 📄 v2.1.0 — Auto-Injected Manuscript & Communication Surface (Landed locally; not yet tagged)

> **Scope**: A publishable manuscript describing the GNN language and the {{GNN_STEP_COUNT}}-step
> pipeline, where every quantitative value is auto-injected from the live repository — nothing
> hard-coded — rendered through the docxology/template manuscript pipeline.
> **Status**: Built and verified locally (uncommitted). Render: 26-page PDF, 0 unresolved tokens,
> 0 dangling citations/figures, cross-vendor (GPT-5.4 Forge) audited.

- [x] **Deterministic manuscript-variable producer** — `src/manuscript_variables.py` + thin
  `scripts/z_generate_manuscript_variables.py` introspect the live repo and emit 38 `{{TOKEN}}`
  values (steps, families, backends, MCP tools from the audit ledger, tests, corpora, source scale).
  Byte-identical across runs; 12 no-mock unit tests cross-check each count against its source surface.
- [x] **Evidence-bound manuscript** — `manuscript/` promoted from generic scaffold to 9 authored
  sections + glossary with real GNN content (StateSpaceBlock/A–B–C–D matrices, Triple Play, pipeline),
  7 verified `references.bib` entries, and zero hard-coded numbers (enforced by a new
  `scripts/check_manuscript_tokens.py` gate with a negative-control test).
- [x] **Generated figures** — 5 deterministic generators (`scripts/manuscript_fig_*.py`) → pipeline
  DAG, family×framework matrix, backend registry, repo-metrics, Triple Play; all embedded in the PDF.
- [ ] **Publication metadata** — assign a manuscript DOI / Zenodo record and fill `manuscript/config.yaml`
  `publication.doi`/`version_doi` before any public release.

### Acceptance
```bash
uv run --extra dev python -m pytest src/tests/test_manuscript_variables.py -q
uv run python scripts/z_generate_manuscript_variables.py   # regenerates output/data/manuscript_variables.json
uv run python scripts/check_manuscript_tokens.py            # 0 unknown tokens / 0 dangling cites / 0 hard-coded counts
# Render (from the sibling docxology/template checkout):
#   uv run python scripts/03_render_pdf.py --project GeneralizedNotationNotation
```

---

## 🌱 v3.0.0 — Long-Running Orchestration & Distributed Ecology Plans

> **Scope**: Prepare safe long-running orchestration, durable observation streams, and auditable container plans before any live infrastructure mutation.
> **Status**: Foundation contracts landed locally (uncommitted) as three safe-by-design `src/pipeline/`
> modules with no-mocks tests, a strict acceptance gate, and MCP tools. No live mutation is performed.

- [x] **Durable Observation Streams** — `src/pipeline/durable_streams.py`: `StreamManifest` (file/array) with
  content checksums, `ExecutionTrace` with `trace_integrity` (monotonic/contiguous/no-dup), atomic IO, and a
  deterministic `replay_trace`/`verify_replay`. Tampered streams and gapped traces are caught (negative controls).
- [x] **Long-Running Pipeline Sessions** — `src/pipeline/run_session.py`: `RunSession`/`WorkUnit` manifests,
  atomic `checkpoint`/`load_session`, `remaining_units`/`resume_plan`, `status_report`, and
  `cancel_safe_cleanup` (removes only non-DONE artifacts, path-escape-safe, idempotent).
- [x] **Auditable Container Plans** — `src/pipeline/container_plan.py`: `generate_container_plan` (hardened
  default: non-root, read-only rootfs, cap-drop ALL, pinned-digest image, resource limits), static
  `security_review` (CRITICAL/HIGH/MEDIUM/LOW findings), `RollbackDescriptor`, deterministic `plan_hash`.
  **No cluster/container is ever executed** (source-level safety test enforces this).

> Evidence: 40 no-mocks unit tests green (`src/tests/pipeline/test_{durable_streams,run_session,container_plan}.py`);
> full `src/tests/pipeline` suite 333 passed (no regression); MCP audit PASS with 3 new tools
> (`tools_total` 137→140); `just lint`/ruff clean.

### Acceptance
```bash
# Unit contracts (no mocks, with negative controls):
PYTHONPATH=src uv run python -m pytest src/tests/pipeline/test_durable_streams.py \
  src/tests/pipeline/test_run_session.py src/tests/pipeline/test_container_plan.py -q
# End-to-end acceptance gate (fails closed; --inject-defect exits non-zero):
PYTHONPATH=src uv run python scripts/run_v3_orchestration_acceptance.py --strict
```

> **Remaining for a tagged v3.0.0 release**: wire the session manifest into the live
> `model_family_acceptance` runner and the stream/trace contracts into Step 12 execution, then
> exercise an extended multi-family run end to end (the contracts above are the prerequisite foundation).

---

## 🚀 v4.0.0 — Bounded Autonomy & Self-Modifying Workflows

> **Scope**: Only after v1.9, v2.0, and v3.0 reliability gates are release-grade, promote bounded proposal artifacts toward reviewed self-editing workflows.

- [ ] **Autonomous Candidate Scoring** — Expand proposal-only candidate patch scoring using existing validators, model-family ledgers, execution summaries, and interpretability reports.
- [ ] **Reviewed Self-Editing GNN Files** — Add guarded workflows for proposing and applying GNN edits only after explicit user approval; no automatic source mutation.
- [ ] **Autonomous Ecology Controls** — Add policy, rollback, audit, and security controls before any self-editing or distributed mutation is permitted.

### Acceptance
```bash
uv run python src/main.py --autonomous --target-dir input/recursive_models/ --output-dir /tmp/gnn-autonomous-smoke
```

---

## Conventions

- Versions follow [SemVer](https://semver.org/) — `MAJOR.MINOR.PATCH`
- All releases require current verifier gates, real-implementation tests for changed surfaces, documentation integrity, and verifiable console acceptance metrics.
- Items marked `[x]` are verified complete against the codebase, not estimated.
