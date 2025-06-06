# GeneralizedNotationNotation (GNN) Documentation

Welcome to the comprehensive documentation for Generalized Notation Notation (GNN), a standardized text-based language for expressing Active Inference generative models.

> **Important**: For setup and installation instructions, please refer to the [GNN Project Setup Guide](SETUP.md).

## 🚀 Quick Start

New to GNN? Start here:

1. [**What is GNN?**](gnn/about_gnn.md) - Overview and motivation
2. [**Quickstart Tutorial**](gnn/quickstart_tutorial.md) - Comprehensive getting started guide
3. [**Your First GNN Model**](gnn/gnn_examples_doc.md#example-1-static-perception-model) - Simple example walkthrough
4. [**Basic Syntax Guide**](gnn/gnn_syntax.md) - Essential notation rules
5. [**Tools Setup**](gnn/gnn_tools.md#installation) - Get GNN tools running

## 📚 Documentation Structure

### For Beginners
- [GNN Overview](gnn/gnn_overview.md) - High-level concepts and ecosystem
- [About GNN](gnn/about_gnn.md) - Motivation, goals, and "triple play" approach
- [Basic Examples](gnn/gnn_examples_doc.md) - Step-by-step model development

### For Practitioners
- [GNN Syntax Reference](gnn/gnn_syntax.md) - Complete notation specification
- [File Structure Guide](gnn/gnn_file_structure_doc.md) - How to organize GNN files
- [Implementation Guide](gnn/gnn_implementation.md) - Best practices for creating models
- [Tools and Resources](gnn/gnn_tools.md) - Available software and utilities

### For Developers
- [Integration Guides](#framework-integrations) - Framework-specific documentation
- [Advanced Topics](#advanced-topics) - Complex modeling patterns
- [Tool Development](gnn/gnn_dsl_manual.md) - Creating GNN-compatible tools

### For Researchers
- [Academic Paper](gnn/gnn_paper.md) - Formal specification and theory
- [Multiagent Systems](gnn/gnn_multiagent.md) - Multi-agent modeling approaches
- [LLM Integration](gnn/gnn_llm_neurosymbolic_active_inference.md) - AI-assisted modeling

## 🔧 Framework Integrations

| Framework | Documentation | Description |
|-----------|---------------|-------------|
| **PyMDP** | [gnn_pymdp.md](pymdp/gnn_pymdp.md) | Python Active Inference framework |
| **RxInfer** | [gnn_rxinfer.md](rxinfer/gnn_rxinfer.md) | Julia Bayesian inference |
| **DisCoPy** | [gnn_discopy.md](discopy/gnn_discopy.md) | Category theory and quantum computing |
| **DSPy** | [gnn_dspy.md](dspy/gnn_dspy.md) | AI prompt programming |
| **AutoGenLib** | [gnn_autogenlib.md](autogenlib/gnn_autogenlib.md) | Dynamic code generation |
| **MCP** | [gnn_mcp.md](mcp/gnn_mcp_model_context_protocol.md) | Model Context Protocol |

## 📖 Example Gallery

Explore increasingly complex GNN models:

### Basic Examples
- [Static Perception](archive/gnn_example_dynamic_perception.md) - Simplest GNN model
- [Dynamic Perception](archive/gnn_example_dynamic_perception_policy.md) - Adding time dynamics

### Intermediate Examples  
- [Butterfly Agent](archive/gnn_example_butterfly_pheromone_agent.md) - POMDP navigation
- [Trading Agent](archive/gnn_airplane_trading_pomdp.md) - Decision making under uncertainty

### Advanced Examples
- [Language Model](archive/gnn_active_inference_language_model.md) - Multi-level linguistic processing
- [Learning Agent](archive/gnn_example_jax_pymdp_learning_agent.md) - Parameter learning in JAX
- [Poetic Muse](archive/gnn_poetic_muse_model.md) - Creative Bayesian network

## 🎯 Advanced Topics

- [Advanced Modeling Patterns](gnn/advanced_modeling_patterns.md) - Sophisticated modeling techniques
- [Ontology System](gnn/ontology_system.md) - Active Inference Ontology integration
- [Resource Metrics](gnn/resource_metrics.md) - Computational resource estimation
- [GNN Kit](kit/gnn_kit.md) - Comprehensive toolkit documentation
- [Cerebrum Integration](cerebrum/gnn_cerebrum.md) - Advanced cognitive architectures

## 🔍 Quick Reference

### Common Tasks
- **Creating your first model**: Start with [Static Perception Example](gnn/gnn_examples_doc.md#example-1-static-perception-model)
- **Understanding syntax**: Check [GNN Syntax Reference](gnn/gnn_syntax.md)
- **Validating models**: Use [Type Checker Guide](gnn/gnn_tools.md#validation-tools)
- **Converting to code**: See [Rendering Documentation](gnn/gnn_tools.md#conversion-tools)
- **Visualizing models**: Follow [Visualization Guide](gnn/gnn_tools.md#visualization-tools)

### File Templates
- [Basic GNN Template](templates/basic_gnn_template.md) *(to be created)*
- [POMDP Template](templates/pomdp_template.md) *(to be created)*
- [Multi-agent Template](templates/multiagent_template.md) *(to be created)*

### Troubleshooting
- [Common Errors](troubleshooting/common_errors.md) *(to be created)*
- [FAQ](troubleshooting/faq.md) *(to be created)*
- [Performance Tips](troubleshooting/performance.md) *(to be created)*

## 🤝 Contributing to Documentation

We welcome contributions! See our [Documentation Style Guide](contributing/documentation_style_guide.md) *(to be created)* for:

- Writing standards and conventions
- Documentation templates and examples  
- Review process and quality criteria
- How to add new examples and tutorials

## 📞 Getting Help

- **Issues**: Report problems on [GitHub Issues](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)
- **Community**: Connect with the [Active Inference Institute](https://activeinference.org)

---

**Last Updated**: 2023-11-15  
**Documentation Version**: Compatible with GNN v1.x 