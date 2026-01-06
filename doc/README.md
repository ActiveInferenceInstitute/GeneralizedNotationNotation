# GeneralizedNotationNotation (GNN) Documentation

> **📋 Document Metadata**  
> **Type**: Navigation Hub | **Audience**: All Users | **Complexity**: Beginner  
> **Status**: Production-Ready  
> **Cross-References**: [Setup Guide](SETUP.md) | [Contributing](../CONTRIBUTING.md)

Welcome to the comprehensive documentation for Generalized Notation Notation (GNN), a standardized text-based language for expressing Active Inference generative models.

> **⚠️ Important**: For setup and installation instructions, please refer to the [GNN Project Setup Guide](SETUP.md).

## 🚀 Quick Start

> **🎯 Learning Path**: Beginner → Intermediate → Advanced (Estimated Time: 2-8 hours total)

**New to GNN?** Follow the **Beginner Path** in our [Learning Paths Guide](learning_paths.md):

1. **[What is GNN?](gnn/about_gnn.md)** - Overview and motivation
   - *Cross-refs*: [GNN Overview](gnn/gnn_overview.md), [Academic Paper](gnn/gnn_paper.md)
2. **[Quickstart Tutorial](quickstart.md)** - Comprehensive getting started guide  
   - *Cross-refs*: [Basic Examples](gnn/gnn_examples_doc.md), [Template System](templates/README.md)
3. **[Your First GNN Model](gnn/gnn_examples_doc.md#example-1-static-perception-model)** - Simple example walkthrough
   - *Cross-refs*: [Basic Template](templates/basic_gnn_template.md), [Syntax Reference](gnn/gnn_syntax.md)
4. **[Basic Syntax Guide](gnn/gnn_syntax.md)** - Essential notation rules
   - *Cross-refs*: [File Structure](gnn/gnn_file_structure_doc.md), [Implementation Guide](gnn/gnn_implementation.md)
5. **[Tools Setup](gnn/gnn_tools.md#installation)** - Get GNN tools running
   - *Cross-refs*: [Pipeline Guide](pipeline/README.md), [API Documentation](api/README.md)

**📚 Complete Learning Paths**: See [Learning Paths Guide](learning_paths.md) for structured beginner → intermediate → advanced progression

## 📚 Documentation Structure

> **🧭 Navigation by User Type** | **🔗 Related**: Comprehensive cross-reference system

### For Beginners
> **📖 Learning Path**: Concepts → Syntax → Examples → Practice ([Full Beginner Path](learning_paths.md#beginner-path))

- **[GNN Overview](gnn/gnn_overview.md)** - High-level concepts and ecosystem
  - *See Also*: [About GNN](gnn/about_gnn.md), [Academic Paper](gnn/gnn_paper.md)
  - *Next Steps*: [Quickstart Tutorial](quickstart.md)
- **[About GNN](gnn/about_gnn.md)** - Motivation, goals, and "triple play" approach
  - *See Also*: [GNN Overview](gnn/gnn_overview.md), [Ontology System](gnn/ontology_system.md)
  - *Next Steps*: [Basic Examples](gnn/gnn_examples_doc.md)
- **[Basic Examples](gnn/gnn_examples_doc.md)** - Step-by-step model development
  - *See Also*: [Template System](templates/README.md), [Quickstart Tutorial](quickstart.md)
  - *Next Steps*: [Syntax Reference](gnn/gnn_syntax.md)

### For Intermediate Users
> **🛠️ Learning Path**: Syntax → Structure → Implementation → Tools ([Full Intermediate Path](learning_paths.md#intermediate-path))

- **[GNN Syntax Reference](gnn/gnn_syntax.md)** - Complete notation specification
  - *See Also*: [File Structure](gnn/gnn_file_structure_doc.md), [Examples](gnn/gnn_examples_doc.md)
  - *Related Tools*: [Type Checker](gnn/gnn_tools.md#validation-tools), [Templates](templates/README.md)
- **[File Structure Guide](gnn/gnn_file_structure_doc.md)** - How to organize GNN files
  - *See Also*: [Syntax Reference](gnn/gnn_syntax.md), [Implementation Guide](gnn/gnn_implementation.md)
  - *Related Tools*: [Pipeline Documentation](pipeline/README.md)
- **[Implementation Guide](gnn/gnn_implementation.md)** - Best practices for creating models
  - *See Also*: [Advanced Patterns](gnn/advanced_modeling_patterns.md), [Framework Integration](#framework-integrations)
  - *Related Tools*: [Testing Guide](testing/README.md), [API Documentation](api/README.md)
- **[Tools and Resources](gnn/gnn_tools.md)** - Available software and utilities
  - *See Also*: [Pipeline Guide](pipeline/README.md), [API Documentation](api/README.md)
  - *Related*: [Configuration Guide](configuration/README.md), [Deployment Guide](deployment/README.md)

### For Developers
> **⚙️ Learning Path**: APIs → Architecture → Integration → Development

- **[Integration Guides](#framework-integrations)** - Framework-specific documentation
  - *Featured*: [PyMDP](pymdp/gnn_pymdp.md), [RxInfer](rxinfer/gnn_rxinfer.md), [DisCoPy](discopy/gnn_discopy.md)
  - *See Also*: [API Documentation](api/README.md), [Development Guide](development/README.md)
- **[Advanced Topics](#advanced-topics)** - Complex modeling patterns
  - *Featured*: [Advanced Patterns](gnn/advanced_modeling_patterns.md), [Multi-agent](gnn/gnn_multiagent.md)
  - *See Also*: [Cognitive Phenomena](cognitive_phenomena/README.md), [LLM Integration](gnn/gnn_llm_neurosymbolic_active_inference.md)
- **[Tool Development](gnn/gnn_dsl_manual.md)** - Creating GNN-compatible tools
  - *See Also*: [API Documentation](api/README.md), [Pipeline Architecture](pipeline/PIPELINE_ARCHITECTURE.md)
  - *Related*: [MCP Integration](mcp/README.md), [Testing Framework](testing/README.md)

### For Researchers
> **🔬 Learning Path**: Theory → Specification → Applications → Research

- **[Academic Paper](gnn/gnn_paper.md)** - Formal specification and theory
  - *See Also*: [About GNN](gnn/about_gnn.md), [Ontology System](gnn/ontology_system.md)
  - *Related Research*: [Advanced Patterns](gnn/advanced_modeling_patterns.md), [Multi-agent](gnn/gnn_multiagent.md)
- **[Multiagent Systems](gnn/gnn_multiagent.md)** - Multi-agent modeling approaches
  - *See Also*: [Multi-agent Template](templates/multiagent_template.md), [Advanced Patterns](gnn/advanced_modeling_patterns.md)
  - *Framework Integration*: [RxInfer Examples](rxinfer/gnn_rxinfer.md#multi-agent-examples)
- **[LLM Integration](gnn/gnn_llm_neurosymbolic_active_inference.md)** - AI-assisted modeling
  - *See Also*: [DSPy Integration](dspy/gnn_dspy.md), [AutoGenLib](autogenlib/gnn_autogenlib.md)
  - *Related Tools*: [MCP Protocol](mcp/README.md), [Pipeline Step 11](pipeline/README.md#step-11-llm-enhanced-analysis)
- **[PoE-World Integration](poe-world/poe-world.md)** - Compositional world modeling research
  - *See Also*: [PoE-World GNN Integration](poe-world/poe-world_gnn.md), [Program Synthesis](dspy/gnn_dspy.md)
  - *Research Applications*: [Hierarchical Template](templates/hierarchical_template.md), [Advanced Patterns](gnn/advanced_modeling_patterns.md)

## 🔧 Framework Integrations

> **🔗 Cross-Platform Compatibility** | **📊 Coverage**: Complete framework integration guides

| Framework | Documentation | Description | Template Compatibility | Examples |
|-----------|---------------|-------------|----------------------|----------|
| **PyMDP** | [gnn_pymdp.md](pymdp/gnn_pymdp.md) | Python Active Inference framework | ✅ All templates | [POMDP](templates/pomdp_template.md), [Multi-agent](templates/multiagent_template.md) |
| **RxInfer** | [gnn_rxinfer.md](rxinfer/gnn_rxinfer.md) | Julia Bayesian inference | ✅ All templates | [Hierarchical](templates/hierarchical_template.md), [Multi-agent](rxinfer/multiagent_trajectory_planning/) |
| **DisCoPy** | [gnn_discopy.md](discopy/gnn_discopy.md) | Category theory and quantum computing | ✅ Advanced templates | [Category Theory Models](discopy/gnn_discopy.md) |
| **DSPy** | [gnn_dspy.md](dspy/gnn_dspy.md) | AI prompt programming | 🔄 LLM integration | [LLM Pipeline](gnn/gnn_llm_neurosymbolic_active_inference.md) |
| **AutoGenLib** | [gnn_autogenlib.md](autogenlib/gnn_autogenlib.md) | Dynamic code generation | 🔄 Code generation | [Tool Development](gnn/gnn_dsl_manual.md) |
| **MCP** | [gnn_mcp.md](mcp/gnn_mcp_model_context_protocol.md) | Model Context Protocol | ✅ API integration | [MCP Guide](mcp/README.md), [FastMCP](mcp/fastmcp.md) |
| **PoE-World** | [poe-world_gnn.md](poe-world/poe-world_gnn.md) | Compositional world modeling | 🔄 Research integration | [PoE-World Overview](poe-world/poe-world.md), [Program Synthesis](dspy/gnn_dspy.md) |

> **🔗 Cross-References**: [API Documentation](api/README.md) | [Pipeline Integration](pipeline/README.md) | [Performance Comparison](troubleshooting/performance.md)

### Additional Framework Categories

#### Audio and Sonification
- **[SAPF](sapf/sapf_gnn.md)** - Structured Audio Processing Framework for sonification
- **[Pedalboard](pedalboard/pedalboard_gnn.md)** - Audio effects framework for model representation

#### Formal Methods and Verification
- **[Axiom](axiom/axiom_gnn.md)** - Formal verification and theorem proving
- **[Petri Nets](petri_nets/README.md)** - Workflow modeling and process analysis
- **[Nock](nock/nock-gnn.md)** - Formal specification language integration

#### Distributed Systems
- **[Iroh](iroh/iroh.md)** - Peer-to-peer networking for decentralized agents
- **[X402](x402/gnn_x402.md)** - Distributed inference protocol

#### Specialized Tools
- **[GUI-Oxdraw](gui_oxdraw/gnn_oxdraw.md)** - Visual model construction interface
- **[OneFileLLM](onefilellm/onefilellm_gnn.md)** - Single-file LLM wrapper for analysis
- **[Vec2Text](vec2text/vec2text_gnn.md)** - Vector-to-text model interpretation

#### Research and Benchmarking
- **[ARC-AGI](arc-agi/arc-agi-gnn.md)** - Abstract reasoning benchmark integration
- **[D2](d2/gnn_d2.md)** - Scriptable diagram generation
- **[Glowstick](glowstick/glowstick_gnn.md)** - Interactive visualization framework
- **[Klong](klong/klong.md)** - Array programming language integration

#### Temporal and Analytical
- **[TimEP](timep/timep_gnn.md)** - Temporal modeling and time series analysis
- **[POMDP](pomdp/pomdp_overall.md)** - POMDP analytical framework
- **[SPM](spm/spm_gnn.md)** - Statistical Parametric Mapping for neuroscience

## 📖 Example Gallery

> **📈 Progressive Complexity** | **🎯 Learning Path**: Basic → Intermediate → Advanced

### Basic Examples
> **⏱️ Time to Complete**: 30 minutes | **Prerequisites**: [Syntax Guide](gnn/gnn_syntax.md)

- **[Static Perception](archive/gnn_example_dynamic_perception.md)** - Simplest GNN model
  - *Template*: [Basic GNN Template](templates/basic_gnn_template.md)
  - *Frameworks*: [PyMDP Tutorial](pymdp/gnn_pymdp.md#basic-examples), [RxInfer Basics](rxinfer/gnn_rxinfer.md#getting-started)
- **[Dynamic Perception](archive/gnn_example_dynamic_perception_policy.md)** - Adding time dynamics
  - *See Also*: [Time Modeling](gnn/gnn_file_structure_doc.md#time-section), [POMDP Template](templates/pomdp_template.md)

### Intermediate Examples  
> **⏱️ Time to Complete**: 1-2 hours | **Prerequisites**: Basic examples + [Implementation Guide](gnn/gnn_implementation.md)

- **[Butterfly Agent](archive/gnn_example_butterfly_pheromone_agent.md)** - POMDP navigation
  - *Template*: [POMDP Template](templates/pomdp_template.md)
  - *Frameworks*: [PyMDP POMDP](pymdp/gnn_pymdp.md#pomdp-examples), [RxInfer Navigation](rxinfer/multiagent_trajectory_planning/)
- **[Trading Agent](archive/gnn_airplane_trading_pomdp.md)** - Decision making under uncertainty
  - *See Also*: [Decision Theory](gnn/advanced_modeling_patterns.md#decision-theory), [Economic Models](cognitive_phenomena/README.md)

### Advanced Examples
> **⏱️ Time to Complete**: 2-4 hours | **Prerequisites**: Intermediate examples + [Advanced Patterns](gnn/advanced_modeling_patterns.md)

- **[Language Model](archive/gnn_active_inference_language_model.md)** - Multi-level linguistic processing
  - *Template*: [Hierarchical Template](templates/hierarchical_template.md)
  - *See Also*: [LLM Integration](gnn/gnn_llm_neurosymbolic_active_inference.md), [Cognitive Phenomena](cognitive_phenomena/README.md)
- **[Learning Agent](archive/gnn_example_jax_pymdp_learning_agent.md)** - Parameter learning in JAX
  - *See Also*: [Advanced Patterns](gnn/advanced_modeling_patterns.md#learning-algorithms), [Performance Guide](troubleshooting/performance.md)
- **[Poetic Muse](archive/gnn_poetic_muse_model.md)** - Creative Bayesian network
  - *See Also*: [Creative AI](cognitive_phenomena/imagination/), [Multi-modal Models](gnn/advanced_modeling_patterns.md)

## 🎯 Advanced Topics

> **🔬 Learning Path**: Theory → Specification → Applications → Research ([Full Advanced Path](learning_paths.md#advanced-path))  
> **🧠 Specialized Applications** | **🔗 Related**: [Cognitive Phenomena](cognitive_phenomena/README.md)

- **[Advanced Modeling Patterns](gnn/advanced_modeling_patterns.md)** - Sophisticated modeling techniques
  - *Cross-refs*: [Implementation Guide](gnn/gnn_implementation.md), [Templates](templates/README.md)
  - *Applications*: [Cognitive Phenomena](cognitive_phenomena/README.md), [Multi-agent](gnn/gnn_multiagent.md)
- **[Ontology System](gnn/ontology_system.md)** - Active Inference Ontology integration
  - *Cross-refs*: [About GNN](gnn/about_gnn.md), [Academic Paper](gnn/gnn_paper.md)
  - *Related*: [Pipeline Step 8](pipeline/README.md#step-8-ontology-processing)
- **[Resource Metrics](gnn/resource_metrics.md)** - Computational resource estimation
  - *Cross-refs*: [Performance Guide](troubleshooting/performance.md), [Type Checker](gnn/gnn_tools.md#validation-tools)
  - *Related*: [Pipeline Step 4](pipeline/README.md#step-4-gnn-type-checker)
- **[GNN Kit](kit/gnn_kit.md)** - Comprehensive toolkit documentation
  - *Cross-refs*: [Tools Guide](gnn/gnn_tools.md), [API Documentation](api/README.md)
- **[Cerebrum Integration](cerebrum/gnn_cerebrum.md)** - Advanced cognitive architectures
  - *Cross-refs*: [Cognitive Phenomena](cognitive_phenomena/README.md), [Hierarchical Template](templates/hierarchical_template.md)
|- **[Audio Sonification](sapf/sapf_gnn.md)** - Auditory representation of model dynamics
  - *Cross-refs*: [SAPF](sapf/README.md), [Pedalboard](pedalboard/pedalboard_gnn.md)
|- **[Formal Verification](axiom/axiom_gnn.md)** - Provably correct model specification
  - *Cross-refs*: [Petri Nets](petri_nets/README.md), [Nock](nock/nock-gnn.md)
|- **[Visual Model Construction](gui_oxdraw/gnn_oxdraw.md)** - Interactive GUI for model building
  - *Cross-refs*: [Oxdraw](gui_oxdraw/README.md), [Glowstick](glowstick/glowstick_gnn.md)

## 🔍 Quick Reference

> **⚡ Fast Access** | **🎯 Common Tasks** | **🔗 Related**: [Troubleshooting](troubleshooting/README.md)

### Common Tasks
- **Creating your first model**: Start with [Static Perception Example](gnn/gnn_examples_doc.md#example-1-static-perception-model)
  - *Tools*: [Basic Template](templates/basic_gnn_template.md) → [Type Checker](gnn/gnn_tools.md#validation-tools) → [PyMDP Rendering](pymdp/gnn_pymdp.md)
- **Understanding syntax**: Check [GNN Syntax Reference](gnn/gnn_syntax.md)
  - *Practice*: [Examples](gnn/gnn_examples_doc.md) → [Templates](templates/README.md) → [Implementation](gnn/gnn_implementation.md)
- **Validating models**: Use [Type Checker Guide](gnn/gnn_tools.md#validation-tools)
  - *Troubleshooting*: [Common Errors](troubleshooting/common_errors.md) → [FAQ](troubleshooting/faq.md)
- **Converting to code**: See [Rendering Documentation](gnn/gnn_tools.md#conversion-tools)
  - *Frameworks*: [PyMDP](pymdp/gnn_pymdp.md) | [RxInfer](rxinfer/gnn_rxinfer.md) | [DisCoPy](discopy/gnn_discopy.md)
- **Visualizing models**: Follow [Visualization Guide](gnn/gnn_tools.md#visualization-tools)
  - *Pipeline*: [Step 6 Visualization](pipeline/README.md#step-6-visualization) → [Step 12 Website](pipeline/README.md#step-12-website-generation)

### File Templates
> **📋 Production-Ready Templates** | **📊 Total**: 4 comprehensive templates (49KB)

- **[Template System Overview](templates/README.md)** - Complete template documentation
  - *Cross-refs*: [Examples](gnn/gnn_examples_doc.md), [Implementation](gnn/gnn_implementation.md)
- **[Basic GNN Template](templates/basic_gnn_template.md)** - Simple model starting point
  - *Use Cases*: Learning, prototyping, static models
  - *Next Steps*: [POMDP Template](templates/pomdp_template.md) or [Examples](gnn/gnn_examples_doc.md)
- **[POMDP Template](templates/pomdp_template.md)** - Comprehensive POMDP modeling template
  - *Use Cases*: Navigation, perception, decision-making
  - *Frameworks*: [PyMDP POMDP](pymdp/gnn_pymdp.md#pomdp-examples), [RxInfer POMDP](rxinfer/gnn_rxinfer.md#pomdp-models)
- **[Multi-agent Template](templates/multiagent_template.md)** - Multi-agent systems template
  - *Use Cases*: Coordination, communication, social modeling
  - *Examples*: [Multi-agent Systems](gnn/gnn_multiagent.md), [RxInfer Multi-agent](rxinfer/multiagent_trajectory_planning/)
- **[Hierarchical Template](templates/hierarchical_template.md)** - Hierarchical architectures template
  - *Use Cases*: Cognitive architectures, multi-scale modeling, complex systems
  - *Related*: [Advanced Patterns](gnn/advanced_modeling_patterns.md), [Cerebrum](cerebrum/gnn_cerebrum.md)

### Pipeline Documentation
> **⚙️ 23-Step Processing Pipeline** | **📈 Complete Workflow Coverage**

- **[Complete Pipeline Guide](pipeline/README.md)** - All 24 steps explained
  - *Architecture*: [Pipeline Architecture](pipeline/PIPELINE_ARCHITECTURE.md)
  - *Configuration*: [Configuration Guide](configuration/README.md)
- **[Pipeline Architecture](pipeline/PIPELINE_ARCHITECTURE.md)** - Technical architecture
  - *Development*: [Development Guide](development/README.md)
  - *API Integration*: [API Documentation](api/README.md)

### API Reference & Integration
> **🔌 Programming Interfaces** | **📚 36KB Documentation** | **🎯 103 Functions**

- **[Complete API Documentation](api/README.md)** - All classes, functions, and interfaces
  - *Development*: [Development Guide](development/README.md)
  - *Examples*: [Tool Development](gnn/gnn_dsl_manual.md)
- **[MCP Integration Guide](mcp/README.md)** - Model Context Protocol APIs
  - *FastMCP*: [FastMCP Guide](mcp/fastmcp.md)
  - *Pipeline*: [Step 7 MCP](pipeline/README.md#step-7-model-context-protocol)
- **[Tool Development](mcp/README.md#development-guidelines)** - Creating new MCP tools
  - *Cross-refs*: [API Documentation](api/README.md), [DSL Manual](gnn/gnn_dsl_manual.md)
- **[Interactive GUI Tools](../src/gui/README.md)** - Visual model construction interfaces
  - *GUI 1*: Form-based constructor (localhost:7860)
  - *GUI 2*: Visual matrix editor (localhost:7861) 
  - *GUI 3*: State space design studio (localhost:7862)
  - *Pipeline*: [Step 22 GUI Processing](pipeline/README.md#step-22-gui-processing)

### Learning Resources
> **📖 Progressive Learning System** | **🎯 Beginner to Expert**

- **[Tutorial System](tutorials/README.md)** - Step-by-step learning guides from beginner to expert
  - *Start Here*: [Quickstart](gnn/quickstart_tutorial.md) → [Examples](gnn/gnn_examples_doc.md)
- **[Configuration Guide](configuration/README.md)** - Complete configuration reference
  - *Examples*: [Configuration Examples](configuration/examples.md)
  - *Deployment*: [Deployment Guide](deployment/README.md)
- **[Testing Guide](testing/README.md)** - Testing strategies and best practices
  - *Pipeline*: [Step 3 Testing](pipeline/README.md#step-3-test-execution)
  - *Quality*: [Style Guide](style_guide.md) and [Testing Guide](testing/README.md)

### Security & Compliance
> **🔒 Enterprise Security** | **📊 Production-Ready**

- **[Security Guide](security/README.md)** - Comprehensive security documentation
  - *LLM Security*: Prompt injection prevention and API security
  - *MCP Security*: Model Context Protocol security measures
  - *Production Security*: Deployment and infrastructure security
  - *Development Security*: Secure coding practices and testing

### Release Management
> **🚀 Professional Release Process** | **📋 Version Control**

- **[Release Management](releases/README.md)** - Complete release process documentation
  - *Versioning*: Semantic versioning strategy and guidelines
  - *Release Cycles*: Regular and emergency release procedures
  - *Quality Assurance*: Testing and validation requirements
  - *Security Releases*: Critical vulnerability response process
- **[Changelog](../CHANGELOG.md)** - Complete project change history
  - *Current Version*: v1.1.0 with comprehensive feature additions
  - *Version History*: Detailed change tracking since project inception
  - *Upgrade Guides*: Migration assistance between major versions

### Documentation Standards
> **📝 Contribution Guidelines** | **✅ Quality Assurance**

- **[Documentation Style Guide](style_guide.md)** - Comprehensive writing and formatting standards
  - *Writing Standards*: Voice, tone, and clarity guidelines
  - *Technical Guidelines*: Code examples, mathematical notation, GNN syntax
  - *Quality Assurance*: Review processes and automated validation
  - *Content Guidelines*: Structure, cross-references, and accessibility

### Deployment & Operations
> **🚀 Production Deployment** | **📊 46KB Operational Documentation**

- **[Deployment Guide](deployment/README.md)** - Local development to enterprise deployment
  - *Configuration*: [Configuration Guide](configuration/README.md)
  - *Testing*: [Testing Guide](testing/README.md)
- **[Performance Guide](performance/README.md)** - Optimization strategies and benchmarking
  - *Troubleshooting*: [Performance Issues](troubleshooting/performance.md)
  - *Resource Metrics*: [Resource Estimation](gnn/resource_metrics.md)
- **[Development Guide](development/README.md)** - Contributing and extending GNN
  - *Contributing*: [Contributing Guide](../CONTRIBUTING.md)
  - *API Development*: [API Documentation](api/README.md)

### Troubleshooting & Support
> **🆘 Comprehensive Problem Solving** | **📊 56KB Support Documentation**

- **[Troubleshooting Guide](troubleshooting/README.md)** - Comprehensive problem-solving guide
  - *Common Issues*: [Common Errors](troubleshooting/common_errors.md)
  - *Performance*: [Performance Issues](troubleshooting/performance.md)
- **[Common Errors](troubleshooting/common_errors.md)** - Detailed error scenarios and solutions
  - *FAQ*: [Frequently Asked Questions](troubleshooting/faq.md)
  - *Validation*: [Type Checker](gnn/gnn_tools.md#validation-tools)
- **[FAQ](troubleshooting/faq.md)** - Extensive frequently asked questions
  - *Learning*: [Learning Resources](#learning-resources)
  - *Community*: [Getting Help](#-getting-help)

## 🤝 Contributing to Documentation

> **📝 Community Contributions Welcome** | **📋 Standards & Guidelines**

We welcome contributions! See our **[Contributing Guide](../CONTRIBUTING.md)** and **[Documentation Style Guide](style_guide.md)** for:

- **Writing standards and conventions**
      - *Style*: [Style Guide](style_guide.md)
  - *Quality*: [Style Guide](style_guide.md) and [Testing Guide](testing/README.md)
- **Documentation templates and examples**  
  - *Templates*: [Template System](templates/README.md)
  - *Examples*: [Example Gallery](#-example-gallery)
- **Review process and quality criteria**
  - *Process*: [Contributing Guide](../CONTRIBUTING.md#submitting-changes)
  - *Quality*: [Testing Guide](testing/README.md)
- **How to add new examples and tutorials**
  - *Development*: [Development Guide](development/README.md)
  - *Templates*: [Template Creation](templates/README.md#creating-new-templates)

## 📞 Getting Help

> **🌐 Community Support Channels**

- **Issues**: Report problems on [GitHub Issues](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/issues)
  - *Before Posting*: Check [Common Errors](troubleshooting/common_errors.md) and [FAQ](troubleshooting/faq.md)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)
  - *Topics*: Research, development, applications, best practices
- **Community**: Connect with the [Active Inference Institute](https://activeinference.org)
  - *Resources*: Papers, workshops, collaborations

> **🔗 Related Support**: [Support Guide](../SUPPORT.md) | [Troubleshooting](troubleshooting/README.md) | [Contributing](../CONTRIBUTING.md)

---

## 📊 Documentation Metadata

> **🏷️ Machine-Readable Navigation Data**

```yaml
document_type: navigation_hub
primary_audience: [beginners, practitioners, developers, researchers]
learning_paths:
  beginner: [about_gnn.md, quickstart_tutorial.md, gnn_examples_doc.md, gnn_syntax.md]
  practitioner: [gnn_syntax.md, gnn_file_structure_doc.md, gnn_implementation.md, templates/README.md]
  developer: [api/README.md, pipeline/PIPELINE_ARCHITECTURE.md, development/README.md]
  researcher: [gnn/gnn_paper.md, advanced_modeling_patterns.md, cognitive_phenomena/README.md, poe-world/poe-world.md]
cross_references:
  setup: [SETUP.md, configuration/README.md, deployment/README.md]
  frameworks: [pymdp/gnn_pymdp.md, rxinfer/gnn_rxinfer.md, discopy/gnn_discopy.md, poe-world/poe-world_gnn.md]
  templates: [templates/README.md, templates/basic_gnn_template.md, templates/pomdp_template.md, templates/multiagent_template.md, templates/hierarchical_template.md]
  support: [troubleshooting/README.md, troubleshooting/common_errors.md, troubleshooting/faq.md]
  research_integration: [poe-world/poe-world.md, gnn/gnn_llm_neurosymbolic_active_inference.md, dspy/gnn_dspy.md]
coverage_metrics:
  total_documents: 70+
  total_content: 1.3MB+
  template_count: 4
  framework_integrations: 12+
  research_integrations: 5+
  pipeline_steps: 24
  cognitive_phenomena: 22+
status: production_ready
quality_level: gold_standard
recent_additions: [poe-world_integration, enhanced_cross_references, improved_research_pathways, 100%_signposting_coverage]
```

---

**Status**: Comprehensive and Production-Ready  
**Documentation Version**: Compatible with GNN v1.x  
**Cross-Reference Network**: ✅ [Fully Integrated](CROSS_REFERENCE_INDEX.md) 