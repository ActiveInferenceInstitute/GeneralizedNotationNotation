# TO-DO — GNN Pipeline Roadmap

**Last Updated**: 2026-06-12
**Current Version**: 1.8.0
**Next Target**: v1.9.0 (model-family reliability and interpretability)

**Current Evidence (2026-06-12)**: Maintained verifier gates pass on the
roadmap-hardening branch. Current inventory is `2397` collected tests with the
documented Ollama ignores. Latest full local suite evidence with the same
Ollama ignores is `2379 passed, 17 skipped, 1 xfailed`. v1.8.0 focused release
smokes passed for `gnn templates list`, `gnn templates show
pomdp-gridworld-3x3`, dry-run `gnn pull` to `/tmp/gnn-pull`, and authenticated
MCP HTTP tests (`12 passed`; combined CLI/MCP/capability suite `32 passed`);
`just lint` passes.

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

## 🎯 v1.7.0 — Multi-Agent Topologies & Interactive Frontends

> **Scope**: Push the pipeline from single-agent generation to interactive, multi-agent architectures with real-time editing and streaming capabilities.
> **RC status**: Foundation contracts are implemented, but this release remains deferred until runtime-depth evidence catches up to the public claims.

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

## 🧭 v1.9.0 — Model-Family Reliability & Interpretability

> **Scope**: Make broader families of generative models reliably traverse the maintained pipeline with validation, execution-status, telemetry, interpretability, and report evidence.

- [ ] **Model-Family Acceptance Harness** — Maintain `input/model_family_manifest.json` and run representative basics, discrete, continuous, hierarchical, multi-agent, precision, structured, gridworld, and scaling-study fixtures through pipeline evidence steps with explicit passed/skipped/failed statuses.
- [ ] **Cross-Step Evidence Ledger** — Link Step 3/5/6/11/12/15/16/23 evidence for each accepted family: parsed model identity, matrix dimensions, renderer status, execution status, telemetry, analysis, visualization, and report artifacts.
- [ ] **Interpretability Summaries** — Emit per-family variable/edge inventories, matrix-shape tables, optional observation/action/free-energy trace previews, renderer/execution status, and artifact links.

Current RC foundation: the all-family strict harness parses real pipeline
summaries, fails closed when summaries or required per-step records are missing,
rejects incomplete `--only-steps` acceptance profiles, clears stale per-family
outputs before each run, and requires concrete artifacts for selected evidence
steps. `continuous` and `hierarchical` still expose raw failed Step 11/12
outcomes; the ledger may record them as explicitly allowed unsupported
renderer/executor skips with incompatibility and `no_executable_scripts`
reasons. This is honest traversal evidence, not full backend reliability.

### Acceptance
```bash
uv run --extra dev python -m pytest src/tests/pipeline/test_model_family_acceptance.py src/tests/analysis/test_interpretability_summary.py src/tests/report/test_model_family_report.py -q
uv run --extra dev python scripts/run_model_family_acceptance.py --manifest input/model_family_manifest.json --families basics,discrete,multiagent,structured --output-dir /tmp/gnn-family-acceptance --strict
uv run --extra dev python scripts/run_model_family_acceptance.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-family-acceptance-all --strict
uv run --extra dev python src/main.py --target-dir input/gnn_files/discrete --output-dir /tmp/gnn-v19-discrete-smoke --skip-steps "2" --skip-llm --verbose
```

---

## 🧪 v2.0.0 — Semantic Fidelity & Cross-Framework Reliability

> **Scope**: Upgrade GNN from broad fixture acceptance to stronger semantic preservation, cross-format round trips, and cross-framework equivalence checks.

- [ ] **Semantic Round-Trip Gates** — Require representative model families to preserve variables, edges, dimensions, and key matrix contracts across maintained formats.
- [ ] **Cross-Framework Result Comparisons** — Compare compatible PyMDP, RxInfer, JAX, NumPyro, PyTorch, ActiveInference.jl, and DisCoPy outputs with explicit skipped/failed states for unavailable frameworks.
- [ ] **Release Readiness Ledger** — Produce one release ledger tying docs, verifier gates, collect-only inventory, focused tests, and acceptance smokes to checked roadmap items.

---

## 🌱 v3.0.0 — Long-Running Orchestration & Distributed Ecology Plans

> **Scope**: Prepare safe long-running orchestration, durable observation streams, and auditable container plans before any live infrastructure mutation.

- [ ] **Durable Observation Streams** — Standardize file/array stream manifests and replayable execution traces before adding live sensors or device-backed streams.
- [ ] **Long-Running Pipeline Sessions** — Add resumable run manifests, status inspection, and cancellation-safe cleanup for extended model-family acceptance runs.
- [ ] **Auditable Container Plans** — Generate validated container plans with security review and rollback semantics; do not mutate real clusters.

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
