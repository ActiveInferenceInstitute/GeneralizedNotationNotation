# GNN Learning Paths

This document outlines structured learning paths for users of varying expertise levels in Generalized Notation Notation (GNN) and Active Inference. Each path includes key resources, prerequisites, and progression steps. Paths are designed to be modular, example-driven, and tied to the project's pipeline for reproducibility.

## Beginner Path: Getting Started with GNN
**Target Audience**: New users with basic programming knowledge (Python/Julia) but no Active Inference experience.

1. **Introduction to GNN**:
   - Read [GNN Overview](/doc/gnn/gnn_overview.md) for core concepts.
   - Review [About GNN](/doc/gnn/about_gnn.md) for project context.

2. **Quickstart Tutorial**:
   - Follow [Quickstart](/doc/quickstart.md) to set up your environment (see [SETUP.md](/doc/SETUP.md)).
   - Build your first model using [Basic GNN Template](/doc/templates/basic_gnn_template.md).

3. **Basic Syntax and Examples**:
   - Study [GNN Syntax](/doc/gnn/gnn_syntax.md).
   - Explore simple examples in [GNN Examples](/doc/gnn/gnn_examples_doc.md).

4. **Run Your First Pipeline**:
   - Execute via `src/main.py` (details in [Pipeline Architecture](/doc/pipeline/PIPELINE_ARCHITECTURE.md)).
   - Visualize results (see [Visualization Docs](/doc/visualization/)).

**Next Steps**: Move to Intermediate Path once comfortable with basic models.

### **🎓 Beginner Skill Checkpoints**
- [ ] Can you explain the "triple play" approach?
- [ ] Have you successfully generated visualization output for a basic model?
- [ ] Can you identify the difference between `s_f0` and `o_m0` in a GNN file?

## Intermediate Path: Building and Integrating Models
**Target Audience**: Users familiar with GNN basics, seeking integrations and advanced patterns.

**Prerequisites**: Complete Beginner Path.

1. **Advanced Modeling**:
   - Dive into [Advanced Modeling Patterns](/doc/gnn/advanced_modeling_patterns.md).
   - Learn multi-agent systems in [GNN Multiagent](/doc/gnn/gnn_multiagent.md).

2. **Integrations**:
   - PyMDP: [GNN PyMDP](/doc/pymdp/gnn_pymdp.md).
   - RxInfer: [GNN RxInfer](/doc/rxinfer/gnn_rxinfer.md).
   - DisCoPy: [GNN DisCoPy](/doc/discopy/gnn_discopy.md).

3. **Tools and APIs**:
   - Use MCP tools ([MCP Docs](/doc/mcp/gnn_mcp_model_context_protocol.md)).
   - Explore LLM enhancements ([LLM Integration](/doc/llm/README.md)).
   - Visualization tools: [GUI Oxdraw](/doc/gui_oxdraw/gnn_oxdraw.md) for visual model construction.
   - Analysis tools: [OneFileLLM](/doc/onefilellm/onefilellm_gnn.md), [Vec2Text](/doc/vec2text/vec2text_gnn.md).

4. **Testing and Troubleshooting**:
   - Run tests ([Testing Docs](/doc/testing/README.md)).
   - Handle errors ([Common Errors](/doc/troubleshooting/common_errors.md)).

**Next Steps**: Proceed to Advanced Path for research-level applications.

### **🛠️ Intermediate Skill Checkpoints**
- [ ] Have you modified a POMDP template for a custom domain?
- [ ] Can you run the same GNN model across two different frameworks (e.g., PyMDP and RxInfer)?
- [ ] Do you understand how to use the MCP tools to query your model structure?

## Advanced Path: Research and Custom Extensions
**Target Audience**: Experienced researchers extending GNN for novel Active Inference applications.

**Prerequisites**: Complete Intermediate Path.

1. **Domain-Specific Applications**:
   - Cognitive Phenomena: Explore subdirs in [/doc/cognitive_phenomena/](/doc/cognitive_phenomena/README.md) (e.g., [Meta-Awareness](/doc/cognitive_phenomena/meta-awareness/)).
   - Ontology: [Ontology System](/doc/gnn/ontology_system.md).

2. **Custom Development**:
   - Extend pipeline ([Development Docs](/doc/development/README.md)).
   - Integrate with external tools (e.g., [SymPy](/doc/sympy/gnn_sympy.md), [DSPy](/doc/dspy/gnn_dspy.md)).
   - Formal methods: [Axiom](/doc/axiom/axiom_gnn.md), [Petri Nets](/doc/petri_nets/README.md), [Nock](/doc/nock/nock-gnn.md).
   - Audio processing: [SAPF](/doc/sapf/sapf_gnn.md), [Pedalboard](/doc/pedalboard/pedalboard_gnn.md).
   - Visualization: [D2 Diagramming](/doc/d2/gnn_d2.md), [Glowstick](/doc/glowstick/glowstick_gnn.md), [GUI Oxdraw](/doc/gui_oxdraw/gnn_oxdraw.md).

3. **Performance and Optimization**:
   - Metrics: [Resource Metrics](/doc/gnn/resource_metrics.md).
   - Optimization: [Performance Guide](/doc/pymdp/pymdp_performance_guide.md).

4. **Contribution and Research**:
   - Read [GNN Paper](/doc/gnn/gnn_paper.md).
74:    - Contribute via [Releases](/doc/releases/README.md) and [Security Framework](/doc/security/security_framework.md).
75: 
76: ### **🔬 Advanced Skill Checkpoints**
77: - [ ] Have you implemented a custom cognitive phenomenon model using GNN?
78: - [ ] Can you explain the categorical foundations of your model using the DisCoPy output?
79: - [ ] Have you integrated a new scientific library into the GNN pipeline?
80: 
81: **Additional Resources**:
- Full index: [Cross-Reference Index](/doc/CROSS_REFERENCE_INDEX.md).
- Style Guide: [Style Guide](/doc/style_guide.md) for contributions.
- For updates, check project [README](/README.md) and [CHANGELOG](/CHANGELOG.md).

These paths emphasize hands-on examples and reproducibility, aligning with GNN's scientific standards. 