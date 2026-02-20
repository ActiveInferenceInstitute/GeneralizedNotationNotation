# GNN Documentation Index

**Version**: v1.1.0  
**Last Updated**: February 9, 2026  
**Status**: ✅ Production Ready  
**Test Count**: 1,083 Tests Passing  

Complete navigation guide for all GNN (Generalized Notation Notation) documentation.

## 🚀 Quick Start

**New to GNN?** Start here:

1. [GNN Overview](gnn_overview.md) - Introduction to GNN and its purpose
2. [Quickstart Tutorial](quickstart_tutorial.md) - Build your first model in 15 minutes
3. [GNN File Structure](gnn_file_structure_doc.md) - Understanding GNN file organization

## 📋 Pipeline Documentation

**GNN Processing Pipeline** (25 Steps, 0-24):

**Core Documentation:**

- **[src/AGENTS.md](../../src/AGENTS.md)** - Master agent scaffolding and complete 25-step registry
- **[src/README.md](../../src/README.md)** - Pipeline architecture and thin orchestrator pattern
- **[src/main.py](../../src/main.py)** - Pipeline orchestrator implementation
- [Architecture Reference](architecture_reference.md) - Implementation patterns and cross-module data flow
- [GNN Tools and Resources](gnn_tools.md) - Complete pipeline usage and examples
- [Technical Reference](technical_reference.md) - Round-trip data flow and entry points

**Quick Start:**

```bash
# Run full pipeline
python src/main.py --target-dir input/gnn_files --verbose

# Run specific steps
python src/main.py --only-steps "3,5,8,11,12" --verbose
```

## 📖 Language Specification

**GNN Syntax and Structure:**

- [GNN Syntax Reference](gnn_syntax.md) - Complete syntax guide with examples
- [GNN DSL Manual](gnn_dsl_manual.md) - Domain-Specific Language specification
- [GNN Schema Specification](gnn_schema.md) - Parsing and validation schemas
- [GNN File Structure](gnn_file_structure_doc.md) - File organization and sections
- [GNN Standards](gnn_standards.md) - Domain knowledge and coding standards
- [About GNN](about_gnn.md) - Detailed GNN specification
- [GNN Troubleshooting](gnn_troubleshooting.md) - Common issues and solutions

## 🎯 Modeling and Examples

**Learning to Model:**

- [Quickstart Tutorial](quickstart_tutorial.md) - Step-by-step first model
- [GNN Examples](gnn_examples_doc.md) - Model progression from simple to complex
- [Advanced Modeling Patterns](advanced_modeling_patterns.md) - Hierarchical and sophisticated techniques
- [Multi-Agent Systems](gnn_multiagent.md) - Multi-agent modeling specification

**Cognitive Phenomena:**

- [Cognitive Phenomena Models](../cognitive_phenomena/) - Attention, memory, learning, emotion models

## 🔧 Implementation and Integration

**Framework Integration:**

- [Framework Integration Guide](framework_integration_guide.md) - PyMDP, RxInfer, ActiveInference.jl, DisCoPy, JAX
- [GNN Implementation Guide](gnn_implementation.md) - Implementation workflows and patterns
- [PyMDP Integration](../pymdp/gnn_pymdp.md) - Python POMDP framework
- [RxInfer Integration](../rxinfer/gnn_rxinfer.md) - Julia Bayesian inference
- [ActiveInference.jl Integration](../activeinference_jl/activeinference-jl.md) - Julia Active Inference
- [DisCoPy Integration](../discopy/gnn_discopy.md) - Category theory and string diagrams
- [CatColab Integration](../catcolab/catcolab_gnn.md) - Categorical compositional modeling

## 🧠 Advanced Topics

**AI and Analysis:**

- [LLM and Neurosymbolic Active Inference](gnn_llm_neurosymbolic_active_inference.md) - LLM integration (Step 13)
- [Ontology System](ontology_system.md) - Active Inference ontology annotations (Step 10)
- [Resource Metrics](resource_metrics.md) - Computational resource estimation (Step 5)

**Quality and Improvement:**

- [Improvement Analysis](improvement_analysis.md) - Pipeline improvement opportunities
- [REPO Coherence Check](REPO_COHERENCE_CHECK.md) - Quality standards and validation

## 📊 Complete Module Registry

### Core Processing (Steps 0-9)

| Step | Script | Module | Purpose |
| ---- | ------ | ------ | ------- |
| 0 | `0_template.py` | [template/](../../src/template/AGENTS.md) | Pipeline initialization |
| 1 | `1_setup.py` | [setup/](../../src/setup/AGENTS.md) | Environment setup |
| 2 | `2_tests.py` | [tests/](../../src/tests/AGENTS.md) | Test suite |
| 3 | `3_gnn.py` | [gnn/](../../src/gnn/AGENTS.md) | GNN parsing |
| 4 | `4_model_registry.py` | [model_registry/](../../src/model_registry/AGENTS.md) | Model versioning |
| 5 | `5_type_checker.py` | [type_checker/](../../src/type_checker/AGENTS.md) | Type validation |
| 6 | `6_validation.py` | [validation/](../../src/validation/AGENTS.md) | Consistency checking |
| 7 | `7_export.py` | [export/](../../src/export/AGENTS.md) | Multi-format export |
| 8 | `8_visualization.py` | [visualization/](../../src/visualization/AGENTS.md) | Graph visualization |
| 9 | `9_advanced_viz.py` | [advanced_visualization/](../../src/advanced_visualization/AGENTS.md) | Advanced plots |

### Simulation and Analysis (Steps 10-16)

| Step | Script | Module | Purpose |
| ---- | ------ | ------ | ------- |
| 10 | `10_ontology.py` | [ontology/](../../src/ontology/AGENTS.md) | Ontology processing |
| 11 | `11_render.py` | [render/](../../src/render/AGENTS.md) | Code generation |
| 12 | `12_execute.py` | [execute/](../../src/execute/AGENTS.md) | Simulation execution |
| 13 | `13_llm.py` | [llm/](../../src/llm/AGENTS.md) | LLM analysis |
| 14 | `14_ml_integration.py` | [ml_integration/](../../src/ml_integration/AGENTS.md) | ML integration |
| 15 | `15_audio.py` | [audio/](../../src/audio/AGENTS.md) | Audio generation |
| 16 | `16_analysis.py` | [analysis/](../../src/analysis/AGENTS.md) | Statistical analysis |

### Integration and Output (Steps 17-24)

| Step | Script | Module | Purpose |
| ---- | ------ | ------ | ------- |
| 17 | `17_integration.py` | [integration/](../../src/integration/AGENTS.md) | System integration |
| 18 | `18_security.py` | [security/](../../src/security/AGENTS.md) | Security validation |
| 19 | `19_research.py` | [research/](../../src/research/AGENTS.md) | Research tools |
| 20 | `20_website.py` | [website/](../../src/website/AGENTS.md) | Website generation |
| 21 | `21_mcp.py` | [mcp/](../../src/mcp/AGENTS.md) | MCP processing |
| 22 | `22_gui.py` | [gui/](../../src/gui/AGENTS.md) | GUI interface |
| 23 | `23_report.py` | [report/](../../src/report/AGENTS.md) | Report generation |
| 24 | `24_intelligent_analysis.py` | [intelligent_analysis/](../../src/intelligent_analysis/AGENTS.md) | Intelligent analysis |

## 🔍 Find What You Need

### By Task

**I want to...**

- **Learn GNN basics** → [GNN Overview](gnn_overview.md) → [Quickstart Tutorial](quickstart_tutorial.md)
- **Understand syntax** → [GNN Syntax](gnn_syntax.md) → [GNN DSL Manual](gnn_dsl_manual.md)
- **Build a model** → [Quickstart Tutorial](quickstart_tutorial.md) → [GNN Examples](gnn_examples_doc.md)
- **Use advanced patterns** → [Advanced Modeling Patterns](advanced_modeling_patterns.md)
- **Integrate frameworks** → [Framework Integration Guide](framework_integration_guide.md)
- **Process models** → [GNN Tools](gnn_tools.md) → [src/AGENTS.md](../../src/AGENTS.md)
- **Understand pipeline** → [src/README.md](../../src/README.md) → [Architecture Reference](architecture_reference.md)
- **Debug issues** → [Technical Reference](technical_reference.md) → [Improvement Analysis](improvement_analysis.md)

### By Audience

**Beginners:**

1. [GNN Overview](gnn_overview.md)
2. [Quickstart Tutorial](quickstart_tutorial.md)
3. [GNN Examples](gnn_examples_doc.md)
4. [GNN Syntax](gnn_syntax.md)

**Researchers:**

1. [Advanced Modeling Patterns](advanced_modeling_patterns.md)
2. [Multi-Agent Systems](gnn_multiagent.md)
3. [LLM and Neurosymbolic AI](gnn_llm_neurosymbolic_active_inference.md)
4. [Ontology System](ontology_system.md)

**Developers:**

1. [src/AGENTS.md](../../src/AGENTS.md)
2. [Architecture Reference](architecture_reference.md)
3. [Technical Reference](technical_reference.md)
4. [Framework Integration Guide](framework_integration_guide.md)
5. [GNN Implementation Guide](gnn_implementation.md)

**System Architects:**

1. [src/README.md](../../src/README.md)
2. [Architecture Reference](architecture_reference.md)
3. [GNN Standards](gnn_standards.md)
4. [Improvement Analysis](improvement_analysis.md)
5. [REPO Coherence Check](REPO_COHERENCE_CHECK.md)

## 🛠️ Additional Resources

**Related Documentation:**

- [GNN Troubleshooting](gnn_troubleshooting.md) - Common issues and solutions
- [API Reference](../api/) - API documentation
- [Configuration Guide](../configuration/) - Configuration options
- [Troubleshooting](../troubleshooting/) - Common issues and solutions
- [Development Guide](../development/) - Development standards

**External Links:**

- [GitHub Repository](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation)
- [Active Inference Institute](https://activeinference.org)
- [Community Discussions](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)

## 📝 Documentation Standards

All GNN documentation follows these principles:

- **Understated**: Concrete examples over promotional language
- **Show Not Tell**: Working code and real outputs
- **Evidence-Based**: Specific metrics and measurable results
- **Functional**: Emphasis on what the code actually does
- **Professional**: Clear, technical, and precise

## 📋 Documentation Status

- ✅ All 25 pipeline steps documented
- ✅ Complete module AGENTS.md coverage (28/28 modules)
- ✅ 100% pipeline success rate (~3 minutes / 172.7 seconds execution time)
- ✅ 1,083 tests passing
- ✅ Enhanced visual logging across all steps
- ✅ Comprehensive cross-referencing between documentation

## 📖 Document Index (Alphabetical)

- [About GNN](about_gnn.md)
- [Advanced Modeling Patterns](advanced_modeling_patterns.md)
- [Architecture Reference](architecture_reference.md)
- [Framework Integration Guide](framework_integration_guide.md)
- [GNN DSL Manual](gnn_dsl_manual.md)
- [GNN Examples](gnn_examples_doc.md)
- [GNN Export Guide](gnn_export.md)
- [GNN File Structure](gnn_file_structure_doc.md)
- [GNN Implementation Guide](gnn_implementation.md)
- [GNN LLM and Neurosymbolic AI](gnn_llm_neurosymbolic_active_inference.md)
- [GNN Multi-Agent](gnn_multiagent.md)
- [GNN Ontology Guide](gnn_ontology.md)
- [GNN Overview](gnn_overview.md)
- [GNN Paper](gnn_paper.md)
- [GNN Schema](gnn_schema.md)
- [GNN Standards](gnn_standards.md)
- [GNN Syntax](gnn_syntax.md)
- [GNN Tools and Resources](gnn_tools.md)
- [GNN Troubleshooting](gnn_troubleshooting.md)
- [GNN Type System](gnn_type_system.md)
- [GNN Visualization Guide](gnn_visualization.md)
- [Improvement Analysis](improvement_analysis.md)
- [Ontology System](ontology_system.md)
- [Quickstart Tutorial](quickstart_tutorial.md)
- [REPO Coherence Check](REPO_COHERENCE_CHECK.md)
- [Resource Metrics](resource_metrics.md)
- [Technical Reference](technical_reference.md)

---

**GNN Version**: v1.1.0
**Pipeline Version**: 1.1.0
**Total Pipeline Steps**: 25 (0-24)
**Last Updated**: 2026-02-09
**Status**: ✅ All Documentation Complete
