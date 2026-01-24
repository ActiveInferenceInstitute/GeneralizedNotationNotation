# GNN Documentation Index

This index provides navigation to all documentation in the GNN repository, organized by category.

## Quick Links

| Getting Started | Core Reference | Frameworks |
|-----------------|----------------|------------|
| [README](../README.md) | [GNN Syntax](gnn/gnn_syntax.md) | [PyMDP](pymdp/README.md) |
| [Setup Guide](../SETUP_GUIDE.md) | [Architecture](../ARCHITECTURE.md) | [RxInfer.jl](rxinfer/README.md) |
| [CLAUDE.md](../CLAUDE.md) | [Pipeline Steps](gnn/gnn_tools.md) | [ActiveInference.jl](activeinference_jl/README.md) |

---

## Core GNN Documentation

### GNN Language & Syntax
- [GNN Overview](gnn/gnn_overview.md) - High-level introduction to GNN
- [About GNN](gnn/about_gnn.md) - Background and motivation
- [GNN Syntax Reference](gnn/gnn_syntax.md) - Complete syntax specification
- [GNN DSL Manual](gnn/gnn_dsl_manual.md) - Domain-specific language guide
- [GNN File Structure](gnn/gnn_file_structure_doc.md) - File format specification
- [GNN Examples](gnn/gnn_examples_doc.md) - Example GNN files
- [GNN Implementation](gnn/gnn_implementation.md) - Implementation details

### Active Inference Theory
- [Active Inference README](active_inference/README.md) - Overview and navigation
- [FEP Foundations](active_inference/fep_foundations.md) - Free Energy Principle
- [Active Inference Theory](active_inference/active_inference_theory.md) - Core theory
- [Expected Free Energy](active_inference/expected_free_energy.md) - EFE computation
- [Generative Models](active_inference/generative_models.md) - Model structures
- [Computational Patterns](active_inference/computational_patterns.md) - Algorithm patterns
- [Glossary](active_inference/glossary.md) - Terminology reference

### Architecture & Design
- [Architecture Reference](gnn/architecture_reference.md) - System architecture
- [Advanced Modeling Patterns](gnn/advanced_modeling_patterns.md) - Design patterns
- [GNN Ontology](gnn/gnn_ontology.md) - Ontological foundations
- [LLM Neurosymbolic Integration](gnn/gnn_llm_neurosymbolic_active_inference.md) - AI integration

---

## Framework Documentation

### Python Frameworks

#### PyMDP
- [PyMDP README](pymdp/README.md) - Overview
- [PyMDP POMDP](pymdp/pymdp_pomdp/README.md) - POMDP implementation

#### JAX
- [JAX Integration](execution/README.md) - JAX execution documentation

#### DisCoPy
- [DisCoPy Integration](discopy/README.md) - Category theory diagrams

### Julia Frameworks

#### RxInfer.jl
- [RxInfer README](rxinfer/README.md) - Reactive inference
- [Multiagent Planning](rxinfer/multiagent_trajectory_planning/README.md) - Multi-agent systems

#### ActiveInference.jl
- [ActiveInference.jl README](activeinference_jl/README.md) - Julia implementation
- [Implementation Guide](active_inference/implementation_activeinference_jl.md) - Integration details

---

## Pipeline Documentation

### Pipeline Steps
- [Pipeline README](gnn/gnn_tools.md) - Pipeline overview
- Step 0: Template initialization
- Step 1: Environment setup
- Step 2: Test execution
- Step 3: GNN parsing
- Step 4: Model registry
- Step 5: Type checking
- Step 6: Validation
- Step 7: Export
- Step 8: Visualization
- Step 9: Advanced visualization
- Step 10: Ontology
- Step 11: Render (code generation)
- Step 12: Execute (simulation)
- Step 13: LLM analysis
- Step 14: ML integration
- Step 15: Audio
- Step 16: Analysis
- Step 17: Integration
- Step 18: Security
- Step 19: Research
- Step 20: Website
- Step 21: MCP
- Step 22: GUI
- Step 23: Report
- Step 24: Intelligent Analysis

### Integration Guides
- [Framework Integration Guide](gnn/framework_integration_guide.md) - Connecting frameworks
- [GNN Export](gnn/gnn_export.md) - Export formats

---

## Specialized Topics

### Visualization
- [Visualization README](visualization/README.md) - Visualization tools
- [D2 Diagrams](d2/README.md) - D2 diagram generation
- [GUI OxDraw](gui_oxdraw/README.md) - Interactive diagrams

### Advanced Features
- [LLM Integration](llm/README.md) - Language model integration
- [MCP Protocol](mcp/README.md) - Model Context Protocol
- [Audio Processing](audio/README.md) - Audio/SAPF integration
- [DSPy](dspy/README.md) - DSPy integration guide

### Development
- [Development Guide](development/README.md) - Contributing guide
- [Testing](testing/README.md) - Test documentation
- [Performance](performance/README.md) - Performance optimization
- [Troubleshooting](troubleshooting/README.md) - Common issues
- [Deployment](deployment/README.md) - Deployment guide

### Security
- [Security Guide](security/README.md) - Security documentation

---

## Research & Theory

### Cognitive Science
- [Cognitive Phenomena](cognitive_phenomena/) - Cognitive modeling
  - [Attention](cognitive_phenomena/attention/)
  - [Consciousness](cognitive_phenomena/consciousness/)
  - [Memory](cognitive_phenomena/memory/)
  - [Perception](cognitive_phenomena/perception/)
  - [Emotion/Affect](cognitive_phenomena/emotion_affect/)
  - [Executive Control](cognitive_phenomena/executive_control/)
  - [Language Processing](cognitive_phenomena/language_processing/)
  - [Learning/Adaptation](cognitive_phenomena/learning_adaptation/)
  - [Meta-Awareness](cognitive_phenomena/meta-awareness/)

### External Libraries
- [Axiom](axiom/README.md) - Axiom integration
- [CatColab](catcolab/README.md) - Category theory
- [Cerebrum](cerebrum/README.md) - Neural modeling
- [Petri Nets](petri_nets/README.md) - Petri net formalism
- [SymPy](sympy/README.md) - Symbolic computation
- [NTQR](ntqr/README.md) - Quantum-inspired reasoning
- [Quadray](quadray/README.md) - Coordinate systems

### Research Notes
- [Research](research/README.md) - Research documentation
- [SPM](spm/README.md) - Statistical Parametric Mapping
- [POMDP](pomdp/README.md) - POMDP theory
- [Type Inference](type-inference-zoo/README.md) - Type systems

---

## Configuration & Templates

- [Configuration](configuration/README.md) - Configuration options
- [Templates](templates/README.md) - Template files
- [Dependencies](dependencies/README.md) - Dependency management

---

## Archive & Legacy

- [Archive](archive/) - Archived documentation
- [Other](other/) - Miscellaneous documents
- [Releases](releases/README.md) - Release notes

---

## Directory Structure

```
doc/
├── active_inference/    # Active Inference theory
├── activeinference_jl/  # ActiveInference.jl framework
├── advanced_visualization/
├── api/                 # API documentation
├── arc-agi/             # ARC-AGI integration
├── archive/             # Archived docs
├── audio/               # Audio processing
├── axiom/               # Axiom integration
├── catcolab/            # Category theory
├── cerebrum/            # Neural modeling
├── cognitive_phenomena/ # Cognitive science models
├── configuration/       # Config docs
├── d2/                  # D2 diagrams
├── dependencies/        # Dependency info
├── deployment/          # Deployment guides
├── development/         # Dev guides
├── discopy/             # DisCoPy integration
├── dspy/                # DSPy integration
├── execution/           # Execution docs
├── export/              # Export formats
├── glowstick/           # Glowstick tool
├── gnn/                 # Core GNN docs
├── gui_oxdraw/          # GUI/OxDraw
├── iroh/                # Iroh integration
├── kit/                 # GNN Kit
├── klong/               # Klong language
├── llm/                 # LLM integration
├── mcp/                 # MCP protocol
├── muscle-mem/          # Muscle memory
├── nock/                # Nock/Nockchain
├── ntqr/                # NTQR framework
├── onefilellm/          # OneFileLLM
├── other/               # Miscellaneous
├── pedalboard/          # Audio effects
├── performance/         # Performance docs
├── petri_nets/          # Petri nets
├── pkl/                 # PKL format
├── poe-world/           # PoE World
├── pomdp/               # POMDP theory
├── pymdp/               # PyMDP framework
├── quadray/             # Coordinate systems
├── releases/            # Release notes
├── research/            # Research docs
├── rxinfer/             # RxInfer.jl
├── sapf/                # SAPF audio
├── security/            # Security docs
├── spm/                 # SPM integration
├── sympy/               # SymPy integration
├── templates/           # Templates
├── testing/             # Test docs
├── timep/               # Time processing
├── troubleshooting/     # Troubleshooting
├── tutorials/           # Tutorials
├── type-inference-zoo/  # Type inference
├── vec2text/            # Vector-to-text
├── visualization/       # Visualization
└── x402/                # X402 protocol
```

---

*Last updated: January 2026*
