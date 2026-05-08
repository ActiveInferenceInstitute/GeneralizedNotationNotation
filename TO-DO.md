# TO-DO — GNN Pipeline Roadmap

**Last Updated**: 2026-05-08
**Current Version**: 1.6.0
**Next Target**: v1.7.0

---

## ✅ v1.6.0 — Zero-Mock Stabilization & Core Infrastructure (Released)

> **Released**: 2026-04-15 (tag: `v1.6.0`)
> **Scope**: Production hardening, MCP integration, renderer expansion, documentation integrity.

- [x] **NumPyro/Stan Renderer Integration** — Complete end-to-end render pathway for NumPyro (322-line renderer) and Stan (100-line renderer) alongside existing PyMDP, RxInfer, JAX, DisCoPy, PyTorch, ActiveInference.jl backends. All 8 renderers operational via `--frameworks` flag on Steps 11/12.
- [x] **MCP Full Module Exposure** — All 25 pipeline modules + infrastructure modules expose tools via 32 `mcp.py` files. 131 registered real tools (no placeholders) across 38+ domains. MCP deadlock in `discover_modules` resolved.
- [x] **PyMDP Scaling Study** — Automated scaling analysis pipeline (`scripts/run_pymdp_gnn_scaling_analysis.py`) with configurable N=[2,256] grids, exponential state-space sweeps, and 19-artifact visualization suite.
- [x] **Test Suite Hardening** — 1,906 tests passing, 30 skipped (Ollama-dependent). Zero-Mock compliance across all modules. Hypothesis tests refactored to deterministic parametric matrices.
- [x] **Documentation Integrity** — 105 `doc/gnn/` files, 34 `AGENTS.md` across `src/`, all version strings synchronized to `1.6.0`. Zero phantom file references.
- [x] **Enhanced Visual Logging** — Progress bars, color-coded output, structured summaries, correlation ID tracking, screen reader support across all 25 pipeline steps.
- [x] **LLM & ML Fixes** — LLM recursive glob fix, ML cross-validation fold logic hardened (`min(5, len(X), min_class_count)`).

---

## 🎯 v1.7.0 — Multi-Agent Topologies & Interactive Frontends

> **Scope**: Push the pipeline from single-agent generation to interactive, multi-agent architectures with real-time editing and streaming capabilities.

- [ ] **Multi-Agent Message Passing (RxInfer)** — Expand the `execute/` layer to handle clustered topologies (100+ agents) passing states asynchronously utilizing graph factorization in Julia via RxInfer.jl.
- [ ] **Categorical Symmetries (DisCoPy)** — Sync matrix permutations natively to string diagrams, allowing visual topology validation before simulation generation.
- [ ] **Reactive WebSocket GUI** — Overhaul the GUI stack (Step 22) into a WebSocket-powered frontend allowing users to adjust agent matrices on the fly without pipeline re-execution. Build on the oxdraw WebSocket architecture stub.
- [ ] **Audio Parameter Streaming** — Bridge Step 15 (Audio/Pedalboard/SAPF) to accept dynamic telemetry updates from long-running PyMDP agent simulations in real time. Extend the existing `process_realtime_chunk` pattern.
- [ ] **3D Matrix Visualization** — Upgrade the Matrix Visualization module into interactive Three.js canvas structures for explorable generative model inspection.

### Acceptance
```bash
uv run python src/main.py --only-steps "11,12" --frameworks "rxinfer,discopy" --target-dir input/multi_agent_models --verbose
uv run pytest src/tests/test_audio*.py src/tests/test_gui*.py
```

---

## 🌐 v1.8.0 — Developer Kit & Template Ecosystem

> **Scope**: Standardizing GNN as the definitive orchestration language with developer-grade tooling and reusable template packages.

- [ ] **GNN Template Library Engine** — Enable package-manager style downloads for specialized active-inference setups directly using `gnn pull [template_name]` via CLI (Step `src/cli/`).
- [ ] **Pre-commit Ecosystem** — Ship `.pre-commit-config.yaml`, `Justfile`, `.devcontainer/`, lint matrices, and auto-formatters to make repository contributions frictionless.
- [ ] **MCP Remote Orchestration** — Extend MCP server from local tool discovery to remote CI/CD agent manipulation with authenticated HTTP transport and rate limiting.

### Acceptance
```bash
gnn pull actinf-pomdp-2state     # Template library works
just lint                         # Developer tooling works
mcp-test-client ping gnn-server   # Remote MCP endpoint responds
```

---

## 🚀 v2.0.0 — Multimodal Autonomy & Self-Modifying Workflows

> **Scope**: Evolving the pipeline from a linear generator into a continuously-running autonomous ecology. Agents define, write, evaluate, and rewrite their own generative models.

- [ ] **Self-Modifying Active Inference** — Implement the capacity for the pipeline to self-recompile agent matrices based on failed execution evaluations, entering a recursive design loop.
- [ ] **Multimodal Agent Interfaces** — Integrate real-time vision processing into the `execute/` modules, allowing simulated agents to optimize policies based on dynamic streams natively defined in their notation.
- [ ] **Distributed Ecology Scaling** — Implement container orchestration allowing massive-scale distributed agent computing clusters triggered by GNN architecture definitions.

### Acceptance
```bash
uv run python src/main.py --autonomous --target-dir input/recursive_models/
```

---

## Conventions

- Versions follow [SemVer](https://semver.org/) — `MAJOR.MINOR.PATCH`
- All releases require 100% pipeline stability, Zero-Mock policy adherence, documentation integrity, and verifiable console acceptance metrics.
- Items marked `[x]` are verified complete against the codebase, not estimated.
