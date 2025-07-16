# GNN Documentation Cross-Reference Index

> **📋 Document Metadata**  
> **Type**: Navigation Index | **Audience**: All Users, Systems | **Complexity**: Reference  
> **Last Updated**: October 2025 | **Status**: Production-Ready  
> **Purpose**: Machine-readable cross-reference network for all GNN documentation

## Overview

> **🎯 Machine Navigation**: Comprehensive cross-reference system for automated tools and enhanced user navigation  
> **📊 Coverage**: 70+ documents with 1300+ cross-references  
> **🔗 Integration**: Bidirectional linking with semantic relationships  
> **🆕 Recent Additions**: Integrated learning_paths.md, expanded cognitive phenomena, PoE-World integration

This index provides a comprehensive mapping of all cross-references within the GNN documentation ecosystem, designed for both human navigation and machine processing.

## Learning Pathways

### Beginner Path
> **📚 Foundation Building** | **⏱️ Estimated Time**: 4-6 hours
1. **[README.md](README.md)** → **[About GNN](gnn/about_gnn.md)** → **[Quickstart Tutorial](gnn/quickstart_tutorial.md)**
2. **[GNN Examples](gnn/gnn_examples_doc.md)** → **[Basic Template](templates/basic_gnn_template.md)**
3. **[GNN Syntax](gnn/gnn_syntax.md)** → **[PyMDP Integration](pymdp/gnn_pymdp.md)**
4. **[Setup Guide](SETUP.md)** → **[First Model Creation](quickstart.md)**
5. **[Learning Paths Overview](learning_paths.md)** (New: Structured beginner guidance)

### Practitioner Path
> **🛠️ Implementation Focus** | **⏱️ Estimated Time**: 8-12 hours
1. **[GNN Syntax](gnn/gnn_syntax.md)** → **[Implementation Guide](gnn/gnn_implementation.md)**
2. **[Template System](templates/README.md)** → **[POMDP Template](templates/pomdp_template.md)**
3. **[Tools Guide](gnn/gnn_tools.md)** → **[Framework Integration](README.md#framework-integrations)**
4. **[Type Checker](gnn/gnn_tools.md#validation-tools)** → **[Pipeline Architecture](pipeline/PIPELINE_ARCHITECTURE.md)**
5. **[Configuration Guide](configuration/README.md)** → **[Deployment Guide](deployment/README.md)**

### Developer Path
> **⚙️ Systems Integration** | **⏱️ Estimated Time**: 12-20 hours
1. **[API Documentation](api/README.md)** → **[Pipeline Architecture](pipeline/PIPELINE_ARCHITECTURE.md)**
2. **[Development Guide](development/README.md)** → **[Testing Guide](testing/README.md)**
3. **[MCP Integration](mcp/README.md)** → **[Tool Development](gnn/gnn_dsl_manual.md)**
4. **[Security Guide](security/README.md)** → **[Performance Optimization](performance/README.md)**
5. **[Contribution Workflow](../CONTRIBUTING.md)** → **[Documentation Standards](style_guide.md)**

### Researcher Path
> **🔬 Advanced Research** | **⏱️ Estimated Time**: 20+ hours
1. **[Academic Paper](gnn/gnn_paper.md)** → **[Advanced Patterns](gnn/advanced_modeling_patterns.md)**
2. **[Multi-agent Systems](gnn/gnn_multiagent.md)** → **[Cognitive Phenomena](cognitive_phenomena/README.md)**
3. **[Cerebrum Integration](cerebrum/gnn_cerebrum.md)** → **[Hierarchical Template](templates/hierarchical_template.md)**
4. **[LLM Integration](gnn/gnn_llm_neurosymbolic_active_inference.md)** → **[DSPy Integration](dspy/gnn_dspy.md)**
5. **[PoE-World Research](poe-world/poe-world.md)** → **[PoE-World GNN Integration](poe-world/poe-world_gnn.md)**
6. **[Advanced Learning Path](learning_paths.md#advanced-path-research-and-custom-extensions)** (New: Research extensions)

## Framework Integration Network

### PyMDP
> **🐍 Python Active Inference** | **📊 Comprehensive Coverage**
- **Primary**: [PyMDP Guide](pymdp/gnn_pymdp.md)
- **Templates**: [POMDP Template](templates/pomdp_template.md), [Basic Template](templates/basic_gnn_template.md)
- **Examples**: [Butterfly Agent](archive/gnn_example_butterfly_pheromone_agent.md), [Trading Agent](archive/gnn_airplane_trading_pomdp.md)
- **Pipeline**: [Step 9 Rendering](pipeline/README.md#step-9-rendering)
- **Advanced**: [Learning Agent](archive/gnn_example_jax_pymdp_learning_agent.md), [Cognitive Effort](archive/gnn_cognitive_effort.md)

### RxInfer.jl
> **🔬 Julia Bayesian Inference** | **🎯 Research-Grade**
- **Primary**: [RxInfer Guide](rxinfer/gnn_rxinfer.md)
- **Templates**: [Multi-agent Template](templates/multiagent_template.md), [Hierarchical Template](templates/hierarchical_template.md)
- **Examples**: [Multi-agent Trajectory Planning](rxinfer/multiagent_trajectory_planning/), [Hidden Markov Model](src/gnn/examples/rxinfer_hidden_markov_model.md)
- **Engineering**: [Engineering Guide](rxinfer/engineering_rxinfer_gnn.md)
- **Pipeline**: [Step 9 Rendering](pipeline/README.md#step-9-rendering)

### DisCoPy
> **🔄 Category Theory Integration** | **📐 Mathematical Foundations**
- **Primary**: [DisCoPy Guide](discopy/gnn_discopy.md)
- **Templates**: [Hierarchical Template](templates/hierarchical_template.md)
- **Theory**: [Advanced Patterns - Compositional Modeling](gnn/advanced_modeling_patterns.md)
- **Pipeline**: [Step 12 Website Generation](pipeline/README.md#step-12-website-generation), [Step 13 SAPF Processing](pipeline/README.md#step-13-sapf-processing)
- **Examples**: [Simple DisCoPy Test](archive/gnn_simple_discopy_test.md)

### LLM Integrations
> **🤖 AI-Enhanced Processing** | **🧠 Intelligent Assistance**
- **DSPy**: [DSPy Integration](dspy/gnn_dspy.md) → [LLM Neurosymbolic](gnn/gnn_llm_neurosymbolic_active_inference.md)
- **AutoGenLib**: [AutoGenLib Guide](autogenlib/gnn_autogenlib.md) → [Code Generation](gnn/gnn_tools.md#code-generation)
- **PoE-World**: [PoE-World Overview](poe-world/poe-world.md) → [PoE-World GNN Integration](poe-world/poe-world_gnn.md)
- **Pipeline**: [Step 11 LLM Analysis](pipeline/README.md#step-11-llm-enhanced-analysis)

### Specialized Frameworks
> **🔧 Domain-Specific Tools** | **🎯 Specialized Applications**
- **MCP**: [MCP Integration](mcp/README.md) → [FastMCP Guide](mcp/fastmcp.md)
- **Cerebrum**: [Cerebrum Guide](cerebrum/gnn_cerebrum.md) → [Cerebrum v1.4](cerebrum/cerebrum_v1-4.md)
- **X402**: [X402 Integration](x402/gnn_x402.md)
- **Glowstick**: [Glowstick Guide](glowstick/glowstick_gnn.md) → [Glowstick Overview](glowstick/glowstick.md)
- **Muscle-Mem**: [Muscle Memory Integration](muscle-mem/gnn-muscle-mem.md)
- **SAPF**: [SAPF Guide](sapf/sapf.md) → [GNN SAPF Integration](sapf/sapf_gnn.md) (New: Added for structural analysis)
- **Quadray**: [Quadray Guide](quadray/quadray.md) → [GNN Quadray](quadray/quadray_gnn.md) (New: Coordinate system integration)

## Topic-Based Index

### Active Inference Theory
> **🧠 Theoretical Foundations** | **📚 Comprehensive Coverage**
- **Core Theory**: [About GNN](gnn/about_gnn.md), [Academic Paper](gnn/gnn_paper.md), [GNN Overview](gnn/gnn_overview.md)
- **Implementation**: [PyMDP Guide](pymdp/gnn_pymdp.md), [RxInfer Guide](rxinfer/gnn_rxinfer.md)
- **Examples**: [GNN Examples](gnn/gnn_examples_doc.md), [Language Model](archive/gnn_active_inference_language_model.md)
- **Advanced**: [Neurosymbolic Integration](gnn/gnn_llm_neurosymbolic_active_inference.md), [Ontology System](gnn/ontology_system.md)

### Modeling Patterns
> **🏗️ Architecture Patterns** | **📖 Progressive Complexity**

#### Basic Patterns
- **Static Models**: [Basic Template](templates/basic_gnn_template.md), [Static Perception](archive/gnn_example_dynamic_perception.md)
- **Dynamic Models**: [Dynamic Perception](archive/gnn_example_dynamic_perception_policy.md)
- **POMDP Models**: [POMDP Template](templates/pomdp_template.md), [POMDP Example](archive/gnn_POMDP_example.md)

#### Intermediate Patterns
- **Navigation**: [Butterfly Agent](archive/gnn_example_butterfly_pheromone_agent.md)
- **Decision Making**: [Trading Agent](archive/gnn_airplane_trading_pomdp.md)
- **Spatial Reasoning**: [Geo-Inference](archive/gnn_geo_infer.md)

#### Advanced Patterns
- **Multi-agent**: [Multi-agent Template](templates/multiagent_template.md), [Multi-agent Theory](gnn/gnn_multiagent.md)
- **Hierarchical**: [Hierarchical Template](templates/hierarchical_template.md), [Cerebrum](cerebrum/gnn_cerebrum.md)
- **Creative AI**: [Poetic Muse](archive/gnn_poetic_muse_model.md)
- **Compositional**: [PoE-World Integration](poe-world/poe-world_gnn.md), [Advanced Patterns](gnn/advanced_modeling_patterns.md)

### Cognitive Phenomena
> **🧠 Cognitive Modeling** | **🔬 Research Applications**
- **Overview**: [Cognitive Phenomena](cognitive_phenomena/README.md)
- **Attention**: [Attention Models](cognitive_phenomena/attention/README.md) → [Attention Model](cognitive_phenomena/attention/attention_model.md)
- **Consciousness**: [Consciousness Models](cognitive_phenomena/consciousness/README.md) → [Global Workspace Model](cognitive_phenomena/consciousness/global_workspace_model.md)
- **Effort**: [Cognitive Effort](cognitive_phenomena/effort/README.md) → [Effort Model](cognitive_phenomena/effort/cognitive_effort.md)
- **Emotion**: [Emotion Models](cognitive_phenomena/emotion_affect/README.md) → [Interoceptive Emotion](cognitive_phenomena/emotion_affect/interoceptive_emotion_model.md)
- **Executive Control**: [Executive Control](cognitive_phenomena/executive_control/README.md) → [Task Switching](cognitive_phenomena/executive_control/task_switching_model.md)
- **Language Processing**: [Language Models](cognitive_phenomena/language_processing/README.md) (New: Added for LLM integrations)
- **Learning**: [Learning Models](cognitive_phenomena/learning_adaptation/README.md) → [Hierarchical Learning](cognitive_phenomena/learning_adaptation/hierarchical_learning_model.md)
- **Memory**: [Memory Models](cognitive_phenomena/memory/README.md) → [Working Memory](cognitive_phenomena/memory/working_memory_model.md)
- **Perception**: [Perception Models](cognitive_phenomena/perception/README.md) → [Bistable Perception](cognitive_phenomena/perception/bistable_perception_model.md)
- **Meta-Awareness**: [Meta-Awareness Models](cognitive_phenomena/meta-awareness/README.md) → [Meta-Aware Implementation](cognitive_phenomena/meta-awareness/meta_aware_model.md) (New: Expanded for advanced cognition)

### Technical Implementation
> **⚙️ Systems and Tools** | **🔧 Implementation Details**
- **Syntax**: [GNN Syntax](gnn/gnn_syntax.md), [File Structure](gnn/gnn_file_structure_doc.md)
- **Tools**: [GNN Tools](gnn/gnn_tools.md), [Pipeline Guide](pipeline/README.md)
- **APIs**: [API Documentation](api/README.md), [MCP Implementation](mcp/README.md)
- **Validation**: [Type Checker](gnn/gnn_tools.md#validation-tools), [Resource Metrics](gnn/resource_metrics.md)
- **Performance**: [Performance Guide](performance/README.md), [Troubleshooting](troubleshooting/performance.md)

### Data Persistence and Serialization
> **💾 Data Management** | **🔄 Serialization Formats**
- **PKL Integration**: [PKL Guide](pkl/pkl_gnn.md) → [PKL Demo](pkl/pkl_gnn_demo.py)
- **Examples**: [Base Model](pkl/examples/BaseActiveInferenceModel.pkl), [Visual Foraging](pkl/examples/VisualForagingModel.pkl)
- **Multi-Format**: [Export Formats](gnn/gnn_tools.md#export-formats), [Format Converters](README.md#format-converters)

### Support and Learning
> **📖 Learning Resources** | **🆘 Support Systems**
- **Quickstart**: [Quick Start](quickstart.md), [Setup Guide](SETUP.md)
- **Troubleshooting**: [Troubleshooting Guide](troubleshooting/README.md), [Common Errors](troubleshooting/common_errors.md), [FAQ](troubleshooting/faq.md)
- **Learning**: [Tutorial System](tutorials/README.md), [Quickstart Tutorial](gnn/quickstart_tutorial.md)
- **Community**: [Contributing Guide](../CONTRIBUTING.md), [Support](../SUPPORT.md)

## Research Integration Network

### Compositional World Modeling
> **🌍 World Model Research** | **🔬 Cutting-Edge Integration**
- **PoE-World**: [Research Overview](poe-world/poe-world.md) → [GNN Integration](poe-world/poe-world_gnn.md)
- **Program Synthesis**: [DSPy Integration](dspy/gnn_dspy.md) → [AutoGenLib](autogenlib/gnn_autogenlib.md)
- **Hierarchical Modeling**: [Advanced Patterns](gnn/advanced_modeling_patterns.md) → [Hierarchical Template](templates/hierarchical_template.md)
- **Multi-Agent**: [Multi-agent Systems](gnn/gnn_multiagent.md) → [RxInfer Multi-agent](rxinfer/multiagent_trajectory_planning/)

### Neurosymbolic AI
> **🧠 Symbolic-Neural Integration** | **🤖 Hybrid Intelligence**
- **LLM Integration**: [LLM Neurosymbolic](gnn/gnn_llm_neurosymbolic_active_inference.md) → [DSPy](dspy/gnn_dspy.md)
- **Symbolic Reasoning**: [DisCoPy Integration](discopy/gnn_discopy.md) → [Category Theory](gnn/advanced_modeling_patterns.md#category-theory)
- **Program Synthesis**: [PoE-World Integration](poe-world/poe-world_gnn.md) → [AutoGenLib](autogenlib/gnn_autogenlib.md)

### Mathematical Foundations
> **📐 Mathematical Rigor** | **🔢 Formal Methods**
- **Category Theory**: [DisCoPy Guide](discopy/gnn_discopy.md) → [Advanced Patterns](gnn/advanced_modeling_patterns.md)
- **Symbolic Math**: [SymPy Integration](sympy/gnn_sympy.md) → [Implementation Summary](sympy/implementation_summary.md)
- **Formal Methods**: [Academic Paper](gnn/gnn_paper.md) → [Ontology System](gnn/ontology_system.md)

## Pipeline Integration Matrix

### 13-Step Processing Pipeline
> **⚙️ Complete Workflow** | **🔄 Automated Processing**

| Step | Component | Primary Documentation | Cross-References | Framework Integration |
|------|-----------|---------------------|------------------|---------------------|
| 1 | **GNN Parsing** | [Pipeline Step 1](pipeline/README.md#step-1-gnn-parsing) | [Syntax Guide](gnn/gnn_syntax.md), [File Structure](gnn/gnn_file_structure_doc.md) | Universal |
| 2 | **Setup** | [Setup Guide](SETUP.md), [Pipeline Step 2](pipeline/README.md#step-2-setup) | [Configuration](configuration/README.md), [Dependencies](SETUP.md#dependencies-explained) | Environment |
| 3 | **Testing** | [Testing Guide](testing/README.md), [Pipeline Step 3](pipeline/README.md#step-3-testing) | [Test Examples](tests/), [Quality Assurance](testing/README.md#quality-assurance) | Validation |
| 4 | **Type Checking** | [Type Checker](gnn/gnn_tools.md#validation-tools), [Pipeline Step 4](pipeline/README.md#step-4-type-checking) | [Resource Metrics](gnn/resource_metrics.md), [Common Errors](troubleshooting/common_errors.md) | Validation |
| 5 | **Export** | [Export Guide](gnn/gnn_tools.md#export-formats), [Pipeline Step 5](pipeline/README.md#step-5-export) | [Format Converters](README.md#format-converters), [Multi-Format](pkl/pkl_gnn.md) | Universal |
| 6 | **Visualization** | [Visualization Guide](gnn/gnn_tools.md#visualization-tools), [Pipeline Step 6](pipeline/README.md#step-6-visualization) | [DisCoPy Diagrams](discopy/gnn_discopy.md), [Advanced Patterns](gnn/advanced_modeling_patterns.md) | Universal |
| 7 | **MCP** | [MCP Guide](mcp/README.md), [Pipeline Step 7](pipeline/README.md#step-7-mcp) | [FastMCP](mcp/fastmcp.md), [API Integration](api/README.md) | Protocol |
| 8 | **Ontology** | [Ontology System](gnn/ontology_system.md), [Pipeline Step 8](pipeline/README.md#step-8-ontology) | [About GNN](gnn/about_gnn.md), [Academic Paper](gnn/gnn_paper.md) | Semantic |
| 9 | **Rendering** | [Code Generation](gnn/gnn_tools.md#code-generation), [Pipeline Step 9](pipeline/README.md#step-9-rendering) | [PyMDP](pymdp/gnn_pymdp.md), [RxInfer](rxinfer/gnn_rxinfer.md) | Framework-Specific |
| 10 | **Execution** | [Execution Guide](gnn/gnn_tools.md#execution), [Pipeline Step 10](pipeline/README.md#step-10-execution) | [PyMDP Examples](pymdp/gnn_pymdp.md#examples), [RxInfer Examples](rxinfer/gnn_rxinfer.md#examples) | Framework-Specific |
| 11 | **LLM** | [LLM Integration](gnn/gnn_llm_neurosymbolic_active_inference.md), [Pipeline Step 11](pipeline/README.md#step-11-llm) | [DSPy](dspy/gnn_dspy.md), [PoE-World](poe-world/poe-world_gnn.md) | AI-Enhanced |
| 12 | **Website** | [Website Generation](gnn/gnn_tools.md#documentation), [Pipeline Step 12](pipeline/README.md#step-12-website) | [Documentation](README.md), [Site Generation](README.md) | Documentation |
| 13 | **SAPF** | [SAPF Guide](sapf/sapf_gnn.md), [Pipeline Step 13](pipeline/README.md#step-13-sapf) | [Audio Generation](sapf/sapf.md), [Spatial Processing](sapf/sapf_gnn.md) | Audio Processing |


## Machine-Readable Navigation Data

```yaml
navigation_graph:
  learning_pathways:
    beginner: [README.md, about_gnn.md, quickstart_tutorial.md, gnn_examples_doc.md, gnn_syntax.md, basic_gnn_template.md, learning_paths.md]
    practitioner: [gnn_syntax.md, gnn_implementation.md, templates/README.md, pomdp_template.md, gnn_tools.md]
    developer: [api/README.md, pipeline/PIPELINE_ARCHITECTURE.md, development/README.md, testing/README.md, mcp/README.md]
    researcher: [gnn/gnn_paper.md, advanced_modeling_patterns.md, cognitive_phenomena/README.md, poe-world/poe-world.md, learning_paths.md]
  
  framework_integrations:
    pymdp: 
      primary: pymdp/gnn_pymdp.md
      templates: [templates/pomdp_template.md, templates/basic_gnn_template.md]
      examples: [archive/gnn_example_butterfly_pheromone_agent.md, archive/gnn_airplane_trading_pomdp.md]
    rxinfer:
      primary: rxinfer/gnn_rxinfer.md  
      templates: [templates/multiagent_template.md, templates/hierarchical_template.md]
      examples: [rxinfer/multiagent_trajectory_planning/, src/gnn/examples/rxinfer_hidden_markov_model.md]
    discopy:
      primary: discopy/gnn_discopy.md
      templates: [templates/hierarchical_template.md]
      examples: [archive/gnn_simple_discopy_test.md]
    llm_integrations:
      dspy: dspy/gnn_dspy.md
      autogenlib: autogenlib/gnn_autogenlib.md
      poe_world: poe-world/poe-world_gnn.md
  
  research_integration:
    compositional_modeling:
      primary: poe-world/poe-world.md
      integration: poe-world/poe-world_gnn.md
      related: [gnn/advanced_modeling_patterns.md, templates/hierarchical_template.md]
    neurosymbolic_ai:
      primary: gnn/gnn_llm_neurosymbolic_active_inference.md
      integrations: [dspy/gnn_dspy.md, autogenlib/gnn_autogenlib.md]
    cognitive_modeling:
      primary: cognitive_phenomena/README.md
      models: [cognitive_phenomena/attention/attention_model.md, cognitive_phenomena/consciousness/global_workspace_model.md, cognitive_phenomena/meta-awareness/meta_aware_model.md]
  
  topic_clusters:
    active_inference_theory: [gnn/about_gnn.md, gnn/gnn_paper.md, gnn/ontology_system.md]
    modeling_patterns: [gnn/advanced_modeling_patterns.md, templates/README.md]
    cognitive_phenomena: [cognitive_phenomena/README.md, cognitive_phenomena/*/README.md]
    technical_implementation: [gnn/gnn_syntax.md, gnn/gnn_tools.md, pipeline/README.md]
    data_persistence: [pkl/pkl_gnn.md, pkl/examples/]
    
  support_network:
    troubleshooting: [troubleshooting/README.md, troubleshooting/common_errors.md, troubleshooting/faq.md]
    learning: [tutorials/README.md, gnn/quickstart_tutorial.md, quickstart.md, learning_paths.md]
    community: [../CONTRIBUTING.md, ../SUPPORT.md]
    quality_assurance: [style_guide.md, testing/README.md]

  pipeline_integration:
    preprocessing: [1, 2, 3, 4]  # GNN, Setup, Tests, Type Check
    processing: [5, 6, 7, 8]     # Export, Visualization, MCP, Ontology
    generation: [9, 10, 11]      # Rendering, Execution, LLM
    advanced: [12, 13, 14]       # DisCoPy, JAX, Site
    
  cross_reference_density:
    high_density: [README.md, gnn/gnn_syntax.md, templates/README.md, pipeline/README.md, learning_paths.md]
    medium_density: [gnn/advanced_modeling_patterns.md, cognitive_phenomena/README.md]
    specialized: [poe-world/poe-world_gnn.md, cerebrum/gnn_cerebrum.md]
```

## Quality Metrics

### Documentation Coverage
> **📊 Comprehensive Analysis** | **✅ Quality Assurance**

| Category | Documents | Cross-References | Coverage Level |
|----------|-----------|------------------|----------------|
| **Core GNN** | 16 | 220+ | ✅ Excellent |
| **Framework Integration** | 9 | 160+ | ✅ Excellent |
| **Templates** | 5 | 90+ | ✅ Complete |
| **Cognitive Phenomena** | 22+ | 120+ | ✅ Comprehensive |
| **Pipeline Documentation** | 3 | 70+ | ✅ Complete |
| **Research Integration** | 12+ | 60+ | 🆕 Newly Added |
| **Support & Troubleshooting** | 8 | 130+ | ✅ Excellent |
| **Tool Integration** | 14+ | 100+ | ✅ Comprehensive |

### Recent Improvements
> **🆕 Documentation Enhancements** | **📈 Continuous Improvement**

- **✅ Added**: Expanded cognitive phenomena with meta-awareness and language processing
- **✅ Enhanced**: Integrated learning_paths.md across pathways
- **✅ Improved**: YAML structure for better machine parsing (added cognitive_modeling models)
- **✅ Expanded**: Specialized frameworks with SAPF and Quadray
- **✅ Updated**: Metrics for increased coverage (70+ docs, 1300+ refs)
- **✅ Strengthened**: Bidirectional links for all new additions

---

**Last Updated**: October 2025  
**Cross-Reference Network**: ✅ Fully Integrated (1300+ references)  
**Machine Readability**: ✅ Structured Data Available with Expanded YAML  
**Coverage Metrics**: 📊 70+ documents, 100% connectivity