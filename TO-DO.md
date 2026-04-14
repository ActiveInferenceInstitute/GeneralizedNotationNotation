# TO-DO — GNN Pipeline Roadmap

**Last Updated**: 2026-04-14
**Current Version**: 1.5.0
**Next Target**: v1.6.0

---

## 🎯 v1.6.0 — Multi-Agent Ecology & RxInfer Scale

> **Scope**: Advanced topologies bridging DisCoPy categorical semantics with highly scalable multi-agent solvers via Julia (RxInfer/ActiveInference.jl). This moves the pipeline from single-agent generation to macroscopic swarm architectures.

- [ ] **Renderer Integration Depth** — Guarantee the complete end-to-end pathway from Markdown Specification -> NumPyro/Stan execution seamlessly without abstract layer data loss.
- [ ] **Multi-Agent Message Passing (RxInfer)** — Expand the `execute/` layer to handle clustered topologies (1,000+ agents) passing states asynchronously utilizing advanced graph factorization in Julia.
- [ ] **Categorical Symmetries (DisCoPy)** — Sync matrix permutations natively to string diagrams, allowing visual topology validation before simulation generation.

### Acceptance
```bash
uv run python src/main.py --frameworks "rxinfer,discopy,numpyro" --target-dir input/multi_agent_models
```

---

## 🧠 v1.7.0 — Sensorimotor Streams: Real-Time Audio & Advanced GUI

> **Scope**: Push the interactive boundaries of GNN from static configurations to real-time streams and live dynamic editing. Bridging outputs dynamically into external reality.

- [ ] **Reactive WebSocket Architectures** — Overhaul the local GUI stack (Step 22) into a cohesive websockets-powered frontend allowing users to adjust agent matrices on the fly without pipeline re-execution.
- [ ] **Audio Parameter Streaming** — Bridge Step 15 (Audio, Pedalboard/SAPF) to accept dynamic telemetry updates from long-running PyMDP agent simulations in real time.
- [ ] **3D Generative Matrix Plottings** — Upgrade the standard Matrix Visualization module into live, explorable Three.js canvas structures.

### Acceptance
```bash
uv run pytest src/tests/test_audio*.py src/tests/test_gui*.py
```

---

## 🌐 v1.8.0 — Enterprise Protocol Integration & Developer Kit (MCP)

> **Scope**: Standardizing GNN as the definitive orchestration language for external agents and enterprises via robust standard protocols.

- [ ] **MCP Capability Mapping** — Fully expose all 25 modules natively through Model Context Protocol (Step 21), allowing remote orchestration and CI/CD agent manipulation.
- [ ] **GNN Template Library Engine** — Enable package-manager style downloads for specialized active-inference setups directly using `gnn pull [template_name]`.
- [ ] **Pre-commit Ecosystem** — Ship robust developer velocity upgrades (`just`, lint matrices, auto-formatters, devcontainers) making repository contributions frictionless.

### Acceptance
```bash
mcp-test-client ping gnn-server  # 100% payload acceptance
```

---

## 🚀 v2.0.0 — Multimodal Autonomy & Self-Modifying Workflows

> **Scope**: Evolving the pipeline from a linear generator into a continuously-running autonomous ecology. Agents define, write, evaluate, and rewrite their own generative models.

- [ ] **Self-Modifying Active Inference** — Implement the capacity for the pipeline to self-recompile agent matrices based on failed execution evaluations, entering a recursive design loop.
- [ ] **Multimodal Agent Interfaces** — Integrate real-time vision processing directly into the `execute/` modules, allowing simulated agents to optimize policies based on dynamic camera or dataset streams natively defined in their notation.
- [ ] **Distributed Ecology Scaling** — Implement K8s orchestration allowing massive-scale distributed agent computing clusters directly triggered by GNN architecture definitions.

### Acceptance
```bash
uv run python src/autonomous_ecology_manager.py --target-dir input/recursive_models/
```

---

## Conventions

- Versions follow [SemVer](https://semver.org/) — `MAJOR.MINOR.PATCH`
- All releases require 100% pipeline stability, 0-mock policy adherence, documentation integrity, and verifiable console acceptance metrics.
