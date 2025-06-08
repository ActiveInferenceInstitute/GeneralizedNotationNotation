# GNN Documentation Cross-Reference Index

> **📋 Document Metadata**  
> **Type**: Navigation Index | **Audience**: All Users, Systems | **Complexity**: Reference  
> **Last Updated**: January 2025 | **Status**: Production-Ready  
> **Purpose**: Machine-readable cross-reference network for all GNN documentation

## Overview

> **🎯 Machine Navigation**: Comprehensive cross-reference system for automated tools and enhanced user navigation  
> **📊 Coverage**: 50+ documents with 1000+ cross-references  
> **🔗 Integration**: Bidirectional linking with semantic relationships

This index provides a comprehensive mapping of all cross-references within the GNN documentation ecosystem, designed for both human navigation and machine processing.

## Learning Pathways

### Beginner Path
1. **[README.md](README.md)** → **[About GNN](gnn/about_gnn.md)** → **[Quickstart Tutorial](gnn/quickstart_tutorial.md)**
2. **[GNN Examples](gnn/gnn_examples_doc.md)** → **[Basic Template](templates/basic_gnn_template.md)**
3. **[GNN Syntax](gnn/gnn_syntax.md)** → **[PyMDP Integration](pymdp/gnn_pymdp.md)**

### Practitioner Path
1. **[GNN Syntax](gnn/gnn_syntax.md)** → **[Implementation Guide](gnn/gnn_implementation.md)**
2. **[Template System](templates/README.md)** → **[POMDP Template](templates/pomdp_template.md)**
3. **[Tools Guide](gnn/gnn_tools.md)** → **[Framework Integration](README.md#framework-integrations)**

### Developer Path
1. **[API Documentation](api/README.md)** → **[Pipeline Architecture](pipeline/PIPELINE_ARCHITECTURE.md)**
2. **[Development Guide](development/README.md)** → **[Testing Guide](testing/README.md)**
3. **[MCP Integration](mcp/README.md)** → **[Tool Development](gnn/gnn_dsl_manual.md)**

### Researcher Path
1. **[Academic Paper](gnn/gnn_paper.md)** → **[Advanced Patterns](gnn/advanced_modeling_patterns.md)**
2. **[Multi-agent Systems](gnn/gnn_multiagent.md)** → **[Cognitive Phenomena](cognitive_phenomena/README.md)**
3. **[Cerebrum Integration](cerebrum/gnn_cerebrum.md)** → **[Hierarchical Template](templates/hierarchical_template.md)**

## Framework Integration Network

### PyMDP
- **Primary**: [PyMDP Guide](pymdp/gnn_pymdp.md)
- **Templates**: [POMDP Template](templates/pomdp_template.md), [Basic Template](templates/basic_gnn_template.md)
- **Examples**: [Butterfly Agent](archive/gnn_example_butterfly_pheromone_agent.md)
- **Pipeline**: [Step 9 Rendering](pipeline/README.md#step-9-rendering)

### RxInfer.jl
- **Primary**: [RxInfer Guide](rxinfer/gnn_rxinfer.md)
- **Templates**: [Multi-agent Template](templates/multiagent_template.md), [Hierarchical Template](templates/hierarchical_template.md)
- **Examples**: [Multi-agent Trajectory Planning](rxinfer/multiagent_trajectory_planning/)
- **Engineering**: [Engineering Guide](rxinfer/engineering_rxinfer_gnn.md)

### DisCoPy
- **Primary**: [DisCoPy Guide](discopy/gnn_discopy.md)
- **Templates**: [Hierarchical Template](templates/hierarchical_template.md)
- **Theory**: [Advanced Patterns - Compositional Modeling](gnn/advanced_modeling_patterns.md)
- **Pipeline**: [Step 12 Categorical Diagrams](pipeline/README.md#step-12-discopy-categorical-diagrams)

## Topic-Based Index

### Active Inference
- **Theory**: [About GNN](gnn/about_gnn.md), [Academic Paper](gnn/gnn_paper.md)
- **Implementation**: [PyMDP Guide](pymdp/gnn_pymdp.md), [RxInfer Guide](rxinfer/gnn_rxinfer.md)
- **Examples**: [GNN Examples](gnn/gnn_examples_doc.md)

### Modeling Patterns
- **Basic**: [Basic Template](templates/basic_gnn_template.md), [Static Perception](archive/gnn_example_dynamic_perception.md)
- **POMDP**: [POMDP Template](templates/pomdp_template.md), [Butterfly Agent](archive/gnn_example_butterfly_pheromone_agent.md)
- **Multi-agent**: [Multi-agent Template](templates/multiagent_template.md), [Multi-agent Theory](gnn/gnn_multiagent.md)
- **Hierarchical**: [Hierarchical Template](templates/hierarchical_template.md), [Cerebrum](cerebrum/gnn_cerebrum.md)

### Technical Implementation
- **Syntax**: [GNN Syntax](gnn/gnn_syntax.md), [File Structure](gnn/gnn_file_structure_doc.md)
- **Tools**: [GNN Tools](gnn/gnn_tools.md), [Pipeline Guide](pipeline/README.md)
- **APIs**: [API Documentation](api/README.md), [MCP Integration](mcp/README.md)

### Support and Learning
- **Troubleshooting**: [Troubleshooting Guide](troubleshooting/README.md), [Common Errors](troubleshooting/common_errors.md)
- **Learning**: [Quickstart Tutorial](gnn/quickstart_tutorial.md), [Tutorial System](tutorials/README.md)
- **Community**: [Contributing Guide](../CONTRIBUTING.md), [Support](../SUPPORT.md)

## Machine-Readable Navigation Data

```yaml
navigation_graph:
  learning_pathways:
    beginner: [about_gnn.md, quickstart_tutorial.md, gnn_examples_doc.md, gnn_syntax.md]
    practitioner: [gnn_syntax.md, gnn_file_structure_doc.md, gnn_implementation.md, templates/README.md]
    developer: [api/README.md, pipeline/PIPELINE_ARCHITECTURE.md, development/README.md]
    researcher: [gnn/gnn_paper.md, gnn/advanced_modeling_patterns.md, cognitive_phenomena/README.md]
  
  framework_integrations:
    pymdp: 
      primary: pymdp/gnn_pymdp.md
      templates: [templates/pomdp_template.md, templates/basic_gnn_template.md]
    rxinfer:
      primary: rxinfer/gnn_rxinfer.md  
      templates: [templates/multiagent_template.md, templates/hierarchical_template.md]
    discopy:
      primary: discopy/gnn_discopy.md
      templates: [templates/hierarchical_template.md]
  
  support_network:
    troubleshooting: [troubleshooting/README.md, troubleshooting/common_errors.md, troubleshooting/faq.md]
    learning: [tutorials/README.md, gnn/quickstart_tutorial.md]
    community: [../CONTRIBUTING.md, ../SUPPORT.md]
```

---

**Last Updated**: January 2025  
**Cross-Reference Network**: ✅ Fully Integrated  
**Machine Readability**: ✅ Structured Data Available 