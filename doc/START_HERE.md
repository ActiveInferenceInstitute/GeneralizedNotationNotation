# GNN Generalized Notation Notation — START HERE

Welcome to the Generalized Notation Notation (GNN) documentation hub!

The `doc/` directory is vast, encompassing a 25-step script pipeline, theoretical foundations, deep-dive implementation guides, cognitive phenomena models, and integrations across 7 computational frameworks.

This document serves as your **Master Table of Contents**, helping you quickly locate exactly what you need.

---

## 🧭 Entry Points by Audience

If you are a...

- **Newcomer**: Start with the [GNN Overview](gnn/gnn_overview.md) and build your first model in 15 minutes with the [Quickstart Tutorial](gnn/tutorials/quickstart_tutorial.md).
- **Researcher**: Read the core theories in [About GNN](gnn/about_gnn.md) and explore [Cognitive Phenomena](cognitive_phenomena/README.md).
- **Developer/Architect**: Dive straight into the [Pipeline Architecture](../src/README.md) and learn about the "Thin Orchestrator" that governs the 25 processing steps.
- **System Integrator**: See the [Framework Integration Guide](gnn/integration/framework_integration_guide.md) and explore the 131 available tools in the [MCP Hub](mcp/README.md).

---

## 🗺️ Master Directory Map

Here is how the `doc/` directory is organized at a high level. Use these links to jump directly to specific domain areas.

### 1. The Core GNN Directory (`doc/gnn/`)

The `doc/gnn/` subdirectory contains the immediate specifications, guides, and internal workings of the GNN modeling language itself. Start at the **[GNN README](gnn/README.md)**.

- **[Tutorials (`doc/gnn/tutorials/`)](gnn/tutorials)**: Step-by-step guides and model progressions.
- **[Reference (`doc/gnn/reference/`)](gnn/reference)**: Strict technical specifications (syntax, architecture, DSL manuals, type systems).
- **[Integration (`doc/gnn/integration/`)](gnn/integration)**: Code generation, framework pipelines, and multi-format export.
- **[Advanced (`doc/gnn/advanced/`)](gnn/advanced)**: Multi-agent systems, LLM/Neurosymbolic combinations, and ontology processing.
- **[Operations (`doc/gnn/operations/`)](gnn/operations)**: Resource metrics, troubleshooting, tooling, and improvement analysis.

### 2. Pipeline Execution & Troubleshooting

Documents explaining how to run, scale, and fix the processing pipeline.

- **[Execution (`doc/execution/`)](execution/)**: Framework availability checks and multi-environment orchestration.
- **[Troubleshooting (`doc/troubleshooting/`)](troubleshooting/)**: Master FAQ, error taxonomy, and resolutions for specific script warnings.
- **[Dependencies (`doc/dependencies/`)](dependencies/)**: Core vs. optional libraries, and Julia/Python interoperability.

### 3. Computional Framework Integrations

Deep-dive implementations for how GNN translates to executable code in specific libraries.

- **[PyMDP (`doc/pymdp/`)](pymdp/)**: Python-based POMDP solvers.
- **[RxInfer (`doc/rxinfer/`)](rxinfer/)**: Julia-based reactive message passing.
- **[ActiveInference.jl (`doc/activeinference_jl/`)](activeinference_jl/)**: Julia-based Active Inference solvers.
- **[JAX (`src/render/jax/`)](../src/render/jax/README.md)**: GPU-accelerated render templates (see also [integration guide](gnn/integration/framework_integration_guide.md)).
- **[DisCoPy (`doc/discopy/`)](discopy/)**: Category theory and string diagrams.
- **[PyTorch (`src/render/pytorch/`)](../src/render/pytorch/README.md)**: Deep learning render templates.
- **[NumPyro (`src/render/numpyro/`)](../src/render/numpyro/README.md)**: Probabilistic programming render templates.
- **[CatColab (`doc/catcolab/`)](catcolab/)**: Categorical compositional intelligence.

### 4. Advanced Tooling & UI

- **[MCP Hub (`doc/mcp/`)](mcp/)**: 131 Model Context Protocol tools to interface with Claude/Cursor/LLMs.
- **[GUI & Visualization (`doc/visualization/`)](visualization/)**: Advanced multi-layer 3D network graphing, matrix heatmaps, and frontend interfaces.
- **[LLM Processing (`doc/llm/`)](llm/)**: Workflows detailing how GNN operates in a neurosymbolic pipeline.

### 5. Research & Domain Applications

- **[Cognitive Phenomena (`doc/cognitive_phenomena/`)](cognitive_phenomena/)**: Theoretical maps to concepts like attention, emotion, learning, depression, and meta-cognition.
- **[Templates (`doc/templates/`)](templates/)**: Quick-start boilerplate models for building your own GNN specifications.

---

## 💡 Top Recommended Reads

If you only read three documents in this entire repository, read these:

1. **[GNN Syntax Reference](gnn/reference/gnn_syntax.md)**: Master the domain-specific markdown to write expressive models.
2. **[Pipeline Orchestrator](../src/README.md)**: Understand how `src/main.py` processes your text file through 25 distinct scientific steps.
3. **[GNN Implementations](gnn/implementations/README.md)**: Explore how those 25 steps ultimately produce real Python/Julia files that evaluate mathematically rigorous Active Inference simulations.

Welcome to GNN.
