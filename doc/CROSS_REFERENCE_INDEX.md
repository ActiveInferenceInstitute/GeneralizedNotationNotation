# GNN Documentation Cross-Reference Index

> **Role**: Topic and framework cross-links (graph-style). **Hub**: [README.md](README.md). **Flat index**: [INDEX.md](INDEX.md). **Canonical step numbering**: [../CLAUDE.md](../CLAUDE.md) and [PIPELINE_SCRIPTS.md](PIPELINE_SCRIPTS.md).

## Overview

- **[Learning paths](learning_paths.md)** — structured curricula
- **Top-level `doc/` folders**: [expected_dirs.txt](expected_dirs.txt) lists **61** topic directories (see [INDEX.md](INDEX.md) tree)

This index maps relationships between frameworks, theory pages, and tooling. For numbered pipeline steps, use [../CLAUDE.md](../CLAUDE.md) (steps 0–24) as the source of truth; older narrative in this file may predate current step ordering.

## Learning Pathways

### Beginner Path
>
> **📚 Foundation Building** | **⏱️ Estimated Time**: 4-6 hours

1. **[README.md](README.md)** → **[About GNN](gnn/about_gnn.md)** → **[Quickstart Tutorial](gnn/tutorials/quickstart_tutorial.md)**
2. **[GNN Examples](gnn/tutorials/gnn_examples_doc.md)** → **[Basic Template](templates/basic_gnn_template.md)**
3. **[GNN Syntax](gnn/reference/gnn_syntax.md)** → **[PyMDP Integration](pymdp/gnn_pymdp.md)**
4. **[Setup Guide](SETUP.md)** → **[First Model Creation](quickstart.md)**
5. **[Learning Paths Overview](learning_paths.md)** - Structured beginner guidance

### Practitioner Path
>
> **🛠️ Implementation Focus** | **⏱️ Estimated Time**: 8-12 hours

1. **[GNN Syntax](gnn/reference/gnn_syntax.md)** → **[Implementation Guide](gnn/integration/gnn_implementation.md)**
2. **[Template System](templates/README.md)** → **[POMDP Template](templates/pomdp_template.md)**
3. **[Tools Guide](gnn/operations/gnn_tools.md)** → **[Framework Integration](README.md#framework-integrations)** (see [README.md](README.md) if the anchor does not resolve in your viewer)
4. **[Type Checker](gnn/operations/gnn_tools.md#validation-tools)** → **[Pipeline architecture](gnn/reference/architecture_reference.md)** (see also [PIPELINE_SCRIPTS.md](PIPELINE_SCRIPTS.md))
5. **[Configuration Guide](configuration/README.md)** → **[Deployment Guide](deployment/README.md)**

### Developer Path
>
> **⚙️ Systems Integration** | **⏱️ Estimated Time**: 12-20 hours

1. **[API Documentation](api/README.md)** → **[Pipeline Architecture](gnn/reference/architecture_reference.md)**
2. **[Development Guide](development/README.md)** → **[Testing Guide](testing/README.md)**
3. **[MCP Integration](mcp/README.md)** → **[Tool Development](gnn/reference/gnn_dsl_manual.md)**
4. **[Security Guide](security/README.md)** → **[Performance Optimization](performance/README.md)**
5. **[Contribution Workflow](../CONTRIBUTING.md)** → **[Documentation Standards](style_guide.md)**

### Researcher Path
>
> **🔬 Advanced Research** | **⏱️ Estimated Time**: 20+ hours

1. **[Academic Paper](gnn/gnn_paper.md)** → **[Advanced Patterns](gnn/advanced/advanced_modeling_patterns.md)**
2. **[Multi-agent Systems](gnn/advanced/gnn_multiagent.md)** → **[Cognitive Phenomena](cognitive_phenomena/README.md)**
3. **[Cerebrum Integration](cerebrum/gnn_cerebrum.md)** → **[Hierarchical Template](templates/hierarchical_template.md)**
4. **[LLM Integration](gnn/advanced/gnn_llm_neurosymbolic_active_inference.md)** → **[DSPy Integration](dspy/gnn_dspy.md)**
5. **[PoE-World Research](poe-world/poe-world.md)** → **[PoE-World GNN Integration](poe-world/poe-world_gnn.md)**
6. **[Advanced learning path](learning_paths.md#advanced-path-research-and-custom-extensions)** — research extensions

## Framework Integration Network

### PyMDP
>
> **🐍 Python Active Inference** | **📊 Comprehensive Coverage**

- **Primary**: [PyMDP Guide](pymdp/gnn_pymdp.md)
- **Templates**: [POMDP Template](templates/pomdp_template.md), [Basic Template](templates/basic_gnn_template.md)
- **Examples**: [Butterfly Agent](archive/gnn_example_butterfly_pheromone_agent.md), [Trading Agent](archive/gnn_airplane_trading_pomdp.md)
- **Pipeline**: code generation and execution ([Step 11 render / Step 12 execute](../CLAUDE.md); [gnn_tools pipeline section](gnn/operations/gnn_tools.md#complete-pipeline-stages-25-steps))
- **Advanced**: [Learning Agent](archive/gnn_example_jax_pymdp_learning_agent.md), [Cognitive Effort](archive/gnn_cognitive_effort.md)

### RxInfer.jl
>
> **🔬 Julia Bayesian Inference** | **🎯 Research-Grade**

- **Primary**: [RxInfer Guide](rxinfer/gnn_rxinfer.md)
- **Templates**: [Multi-agent Template](templates/multiagent_template.md), [Hierarchical Template](templates/hierarchical_template.md)
- **Examples**: [Multi-agent Trajectory Planning](rxinfer/multiagent_trajectory_planning/), [Hidden Markov Model](archive/rxinfer_hidden_markov_model.md)
- **Engineering**: [Engineering Guide](rxinfer/engineering_rxinfer_gnn.md)
- **Pipeline**: [Step 11 render / Step 12 execute](../CLAUDE.md); [gnn_tools pipeline section](gnn/operations/gnn_tools.md#complete-pipeline-stages-25-steps)

### DisCoPy
>
> **🔄 Category Theory Integration** | **📐 Mathematical Foundations**

- **Primary**: [DisCoPy Guide](discopy/gnn_discopy.md)
- **Templates**: [Hierarchical Template](templates/hierarchical_template.md)
- **Theory**: [Advanced Patterns - Compositional Modeling](gnn/advanced/advanced_modeling_patterns.md)
- **Pipeline**: DisCoPy backends participate in **Step 11 (render)** and **Step 12 (execute)** ([CLAUDE.md](../CLAUDE.md)); see [execute/discopy](../../src/execute/discopy_translator_module/README.md) and [render/discopy](../../src/render/discopy/README.md)
- **Examples**: [Simple DisCoPy Test](archive/gnn_simple_discopy_test.md)

### LLM Integrations
>
> **🤖 AI-Enhanced Processing** | **🧠 Intelligent Assistance**

- **DSPy**: [DSPy Integration](dspy/gnn_dspy.md) → [LLM Neurosymbolic](gnn/advanced/gnn_llm_neurosymbolic_active_inference.md)
- **AutoGenLib**: [AutoGenLib Guide](autogenlib/gnn_autogenlib.md) → [Conversion tools](gnn/operations/gnn_tools.md#conversion-tools)
- **PoE-World**: [PoE-World Overview](poe-world/poe-world.md) → [PoE-World GNN Integration](poe-world/poe-world_gnn.md)
- **Pipeline**: **Step 13** — LLM analysis ([CLAUDE.md](../CLAUDE.md); script `13_llm.py`)

### Specialized Frameworks
>
> **🔧 Domain-Specific Tools** | **🎯 Specialized Applications**

- **MCP**: [MCP Integration](mcp/README.md) → [FastMCP Guide](mcp/fastmcp.md)
- **Cerebrum**: [Cerebrum Guide](cerebrum/gnn_cerebrum.md) → [Cerebrum v1.4](cerebrum/cerebrum_v1-4.md)
- **X402**: [X402 Integration](x402/gnn_x402.md)
- **Glowstick**: [Glowstick Guide](glowstick/glowstick_gnn.md) → [Glowstick Overview](glowstick/glowstick.md)
- **Muscle-Mem**: [Muscle Memory Integration](muscle-mem/gnn-muscle-mem.md)
- **SAPF**: [SAPF Guide](sapf/sapf.md) → [GNN SAPF Integration](sapf/sapf_gnn.md)
- **Quadray**: [Quadray Guide](quadray/quadray.md) → [GNN Quadray](quadray/quadray_gnn.md)
- **Axiom**: [Axiom Framework](axiom/axiom_gnn.md) → [Formal Verification](axiom/axiom.md)
- **ARC-AGI**: [ARC-AGI Integration](arc-agi/arc-agi-gnn.md) → [ARC Benchmark](arc-agi/README.md)
- **D2**: [D2 Diagramming](d2/gnn_d2.md) → [D2 Integration](d2/d2.md)
- **Petri Nets**: [Petri Net Modeling](petri_nets/pnml.pnml) → [Workflow Analysis](petri_nets/README.md)
- **OneFileLLM**: [Single-File LLM](onefilellm/onefilellm_gnn.md) → [Integration Guide](onefilellm/README.md)
- **Vec2Text**: [Vector-to-Text](vec2text/vec2text_gnn.md) → [Implementation](vec2text/README.md)
- **Iroh**: [Iroh P2P](iroh/iroh.md) → [Distributed Models](iroh/README.md)
- **Nock**: [Nock Formal Spec](nock/nock-gnn.md) → [Formal Methods](nock/cognitive-security-framework.md)
- **Pedalboard**: [Audio Effects](pedalboard/pedalboard_gnn.md) → [Sonification](pedalboard/README.md)
- **Pkl**: [Configuration Language](pkl/pkl_gnn.md) → [Configuration Management](pkl/README.md)
- **POMDP**: [Theoretical Framework](pomdp/pomdp_overall.md) → [POMDP Analysis](pomdp/README.md)
- **SPM**: [Statistical Mapping](spm/spm_gnn.md) → [Neuroimaging Analysis](spm/README.md)
- **SymPy**: [Symbolic Math](sympy/gnn_sympy.md) → [MCP Integration](sympy/README.md)
- **TimEP**: [Performance Profiling](timep/timep_gnn.md) → [Profiling Tools](timep/README.md)
- **Kit**: [Code Intelligence](kit/gnn_kit.md) → [Developer Tools](kit/README.md)
- **Klong**: [Array Language](klong/klong.md) → [Array Programming](klong/README.md)
- **ActiveInference.jl**: [Julia Framework](activeinference_jl/activeinference-jl.md) → [High-Performance](activeinference_jl/README.md)

## Topic-Based Index

### Active Inference Theory
>
> **🧠 Theoretical Foundations** | **📚 Comprehensive Coverage**

- **Core Theory**: [About GNN](gnn/about_gnn.md), [Academic Paper](gnn/gnn_paper.md), [GNN Overview](gnn/gnn_overview.md)
- **Implementation**: [PyMDP Guide](pymdp/gnn_pymdp.md), [RxInfer Guide](rxinfer/gnn_rxinfer.md)
- **Examples**: [GNN Examples](gnn/tutorials/gnn_examples_doc.md), [Language Model](archive/gnn_active_inference_language_model.md)
- **Advanced**: [Neurosymbolic Integration](gnn/advanced/gnn_llm_neurosymbolic_active_inference.md), [Ontology System](gnn/advanced/ontology_system.md)

### Modeling Patterns
>
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

- **Multi-agent**: [Multi-agent Template](templates/multiagent_template.md), [Multi-agent Theory](gnn/advanced/gnn_multiagent.md)
- **Hierarchical**: [Hierarchical Template](templates/hierarchical_template.md), [Cerebrum](cerebrum/gnn_cerebrum.md)
- **Creative AI**: [Poetic Muse](archive/gnn_poetic_muse_model.md)
- **Compositional**: [PoE-World Integration](poe-world/poe-world_gnn.md), [Advanced Patterns](gnn/advanced/advanced_modeling_patterns.md)

### Cognitive Phenomena
>
> **🧠 Cognitive Modeling** | **🔬 Research Applications**

- **Overview**: [Cognitive Phenomena](cognitive_phenomena/README.md)
- **Attention**: [Attention Models](cognitive_phenomena/attention/README.md) → [Attention Model](cognitive_phenomena/attention/attention_model.md)
- **Consciousness**: [Consciousness Models](cognitive_phenomena/consciousness/README.md) → [Global Workspace Model](cognitive_phenomena/consciousness/global_workspace_model.md)
- **Effort**: [Cognitive Effort](cognitive_phenomena/effort/README.md) → [Effort Model](cognitive_phenomena/effort/cognitive_effort.md)
- **Emotion**: [Emotion Models](cognitive_phenomena/emotion_affect/README.md) → [Interoceptive Emotion](cognitive_phenomena/emotion_affect/interoceptive_emotion_model.md)
- **Executive Control**: [Executive Control](cognitive_phenomena/executive_control/README.md) → [Task Switching](cognitive_phenomena/executive_control/task_switching_model.md)
- **Language Processing**: [Language Models](cognitive_phenomena/language_processing/README.md)
- **Learning**: [Learning Models](cognitive_phenomena/learning_adaptation/README.md) → [Hierarchical Learning](cognitive_phenomena/learning_adaptation/hierarchical_learning_model.md)
- **Memory**: [Memory Models](cognitive_phenomena/memory/README.md) → [Working Memory](cognitive_phenomena/memory/working_memory_model.md)
- **Perception**: [Perception Models](cognitive_phenomena/perception/README.md) → [Bistable Perception](cognitive_phenomena/perception/bistable_perception_model.md)
- **Meta-Awareness**: [Meta-Awareness Models](cognitive_phenomena/meta-awareness/README.md) → [Meta-Aware Implementation](cognitive_phenomena/meta-awareness/meta_aware_model.md)

### Technical Implementation
>
> **⚙️ Systems and Tools** | **🔧 Implementation Details**

- **Syntax**: [GNN Syntax](gnn/reference/gnn_syntax.md), [File Structure](gnn/reference/gnn_file_structure_doc.md)
- **Tools**: [GNN Tools](gnn/operations/gnn_tools.md), [Pipeline Guide](gnn/operations/gnn_tools.md)
- **APIs**: [API Documentation](api/README.md), [MCP Implementation](mcp/README.md)
- **Validation**: [Type Checker](gnn/operations/gnn_tools.md#validation-tools), [Resource Metrics](gnn/operations/resource_metrics.md)
- **Performance**: [Performance Guide](performance/README.md), [Troubleshooting](troubleshooting/performance.md)

### Data Persistence and Serialization
>
> **💾 Data Management** | **🔄 Serialization Formats**

- **PKL Integration**: [PKL Guide](pkl/pkl_gnn.md) → [Configuration Language](pkl/README.md)
- **Examples**: [Base Model](pkl/examples/BaseActiveInferenceModel.pkl), [Visual Foraging](pkl/examples/VisualForagingModel.pkl)
- **Multi-Format**: [GNN tools / pipeline](gnn/operations/gnn_tools.md), [Framework integrations in README](README.md#framework-integrations)
- **Pickle Format**: [Pickle Serialization](pkl/pkl_gnn.md) → [Serialization Tools](pkl/README.md)

### Support and Learning
>
> **📖 Learning Resources** | **🆘 Support Systems**

- **Quickstart**: [Quick Start](quickstart.md), [Setup Guide](SETUP.md)
- **Troubleshooting**: [Troubleshooting Guide](troubleshooting/README.md), [Common Errors](troubleshooting/common_errors.md), [FAQ](troubleshooting/faq.md)
- **Learning**: [Tutorial System](tutorials/README.md), [Quickstart Tutorial](gnn/tutorials/quickstart_tutorial.md)
- **Community**: [Contributing Guide](../CONTRIBUTING.md), [Support](../SUPPORT.md)

## Research Integration Network

### Compositional World Modeling
>
> **🌍 World Model Research** | **🔬 Cutting-Edge Integration**

- **PoE-World**: [Research Overview](poe-world/poe-world.md) → [GNN Integration](poe-world/poe-world_gnn.md)
- **Program Synthesis**: [DSPy Integration](dspy/gnn_dspy.md) → [AutoGenLib](autogenlib/gnn_autogenlib.md)
- **Hierarchical Modeling**: [Advanced Patterns](gnn/advanced/advanced_modeling_patterns.md) → [Hierarchical Template](templates/hierarchical_template.md)
- **Multi-Agent**: [Multi-agent Systems](gnn/advanced/gnn_multiagent.md) → [RxInfer Multi-agent](rxinfer/multiagent_trajectory_planning/)

### Neurosymbolic AI
>
> **🧠 Symbolic-Neural Integration** | **🤖 Hybrid Intelligence**

- **LLM Integration**: [LLM Neurosymbolic](gnn/advanced/gnn_llm_neurosymbolic_active_inference.md) → [DSPy](dspy/gnn_dspy.md)
- **Symbolic Reasoning**: [DisCoPy Integration](discopy/gnn_discopy.md) → [Advanced modeling patterns](gnn/advanced/advanced_modeling_patterns.md)
- **Program Synthesis**: [PoE-World Integration](poe-world/poe-world_gnn.md) → [AutoGenLib](autogenlib/gnn_autogenlib.md)

### Mathematical Foundations
>
> **📐 Mathematical Rigor** | **🔢 Formal Methods**

- **Category Theory**: [DisCoPy Guide](discopy/gnn_discopy.md) → [Advanced Patterns](gnn/advanced/advanced_modeling_patterns.md)
- **Symbolic Math**: [SymPy Integration](sympy/gnn_sympy.md) → [MCP Integration](sympy/README.md)
- **Formal Methods**: [Academic Paper](gnn/gnn_paper.md) → [Ontology System](gnn/advanced/ontology_system.md)
- **Formal Verification**: [Axiom Framework](axiom/axiom_gnn.md) → [Theorem Proving](axiom/axiom.md)
- **Petri Nets**: [Workflow Modeling](petri_nets/pnml.pnml) → [Process Analysis](petri_nets/README.md)
- **NTQR**: [Quantum Reasoning](ntqr/gnn_ntqr.md) → [Hybrid Approaches](ntqr/README.md)
- **POMDP**: [Theoretical Framework](pomdp/pomdp_overall.md) → [POMDP Analysis](pomdp/README.md)
- **Quadray**: [Geometric Coordinates](quadray/quadray_gnn.md) → [Spatial Modeling](quadray/README.md)
- **Type Inference Zoo**: [Type Systems](type-inference-zoo/type-inference-zoo.md) → [Type Inference](type-inference-zoo/README.md)

### Audio and Sonification
>
> **🎵 Auditory Representation** | **🔊 Sensory Modalities**

- **SAPF**: [Audio Framework](sapf/sapf_gnn.md) → [Structured Processing](sapf/README.md)
- **Pedalboard**: [Effects Processing](pedalboard/pedalboard_gnn.md) → [Audio Effects](pedalboard/README.md)

### Temporal and Dynamical Systems
>
> **⏰ Time Series Analysis** | **🔄 Continuous Dynamics**

- **TimEP**: [Temporal Modeling](timep/timep_gnn.md) → [Time Series](timep/README.md)
- **POMDP**: [Analytical Framework](pomdp/pomdp_overall.md) → [Belief State Analysis](pomdp/README.md)

### Distributed Systems and Networking
>
> **🌐 Decentralized Processing** | **🔗 Network Integration**

- **Iroh**: [P2P Networking](iroh/iroh.md) → [Distributed Models](iroh/README.md)
- **X402**: [Protocol Integration](x402/gnn_x402.md) → [Distributed Inference](x402/README.md)

### Neuroscience Integration
>
> **🧠 Brain Science** | **🔬 Neuroscientific Methods**

- **SPM**: [Statistical Mapping](spm/spm_gnn.md) → [Neuroimaging Analysis](spm/README.md)
- **Cognitive Phenomena**: [Cognitive Modeling](cognitive_phenomena/README.md) → [Phenomenon Models](cognitive_phenomena/README.md)

### Setup and Infrastructure
>
> **⚙️ Environment Configuration** | **🛠️ System Management**

- **Dependencies**: [Package Management](dependencies/OPTIONAL_DEPENDENCIES.md) → [Dependency Guide](dependencies/README.md)
- **Execution**: [Framework Management](execution/FRAMEWORK_AVAILABILITY.md) → [Execution Strategy](execution/README.md)

## Pipeline integration matrix

The pipeline has **25 steps (0–24)**. Do not rely on older step-order tables in archived prose; use:

- **[../CLAUDE.md](../CLAUDE.md)** — command-line entry points and step table
- **[PIPELINE_SCRIPTS.md](PIPELINE_SCRIPTS.md)** — numbered `src/N_*.py` orchestrators
- **[gnn/operations/gnn_tools.md — GNN Processing Pipeline](gnn/operations/gnn_tools.md#gnn-processing-pipeline-srcmainpy)** — tooling context and [complete pipeline stages](gnn/operations/gnn_tools.md#complete-pipeline-stages-25-steps)

| Step (excerpt) | Script (repo) | Doc |
|----------------|---------------|-----|
| 11 Render | `11_render.py` | [PIPELINE_SCRIPTS.md](PIPELINE_SCRIPTS.md), [Framework integration](gnn/integration/framework_integration_guide.md) |
| 12 Execute | `12_execute.py` | [execution/README.md](execution/README.md), [src/README.md](../src/README.md) |
| 13 LLM | `13_llm.py` | [llm/README.md](llm/README.md) |
| 15 Audio | `15_audio.py` | [audio/README.md](audio/README.md) |
| 20 Website | `20_website.py` | [website docs](../src/website/README.md) |
| 21 MCP | `21_mcp.py` | [mcp/README.md](mcp/README.md) |
| 23 Report | `23_report.py` | [report](../src/report/README.md) |

## Machine-Readable Navigation Data

```yaml
navigation_graph:
  learning_pathways:
    beginner: [README.md, about_gnn.md, quickstart_tutorial.md, gnn_examples_doc.md, gnn_syntax.md, basic_gnn_template.md, learning_paths.md]
    practitioner: [gnn_syntax.md, gnn_implementation.md, templates/README.md, pomdp_template.md, gnn_tools.md]
    developer: [api/README.md, gnn/reference/architecture_reference.md, development/README.md, testing/README.md, mcp/README.md]
    researcher: [gnn/gnn_paper.md, advanced_modeling_patterns.md, cognitive_phenomena/README.md, poe-world/poe-world.md, learning_paths.md]
  
  framework_integrations:
    pymdp: 
      primary: pymdp/gnn_pymdp.md
      templates: [templates/pomdp_template.md, templates/basic_gnn_template.md]
      examples: [archive/gnn_example_butterfly_pheromone_agent.md, archive/gnn_airplane_trading_pomdp.md]
    rxinfer:
      primary: rxinfer/gnn_rxinfer.md  
      templates: [templates/multiagent_template.md, templates/hierarchical_template.md]
      examples: [rxinfer/multiagent_trajectory_planning/, archive/rxinfer_hidden_markov_model.md]
    discopy:
      primary: discopy/gnn_discopy.md
      templates: [templates/hierarchical_template.md]
      examples: [archive/gnn_simple_discopy_test.md]
    llm_integrations:
      dspy: dspy/gnn_dspy.md
      autogenlib: autogenlib/gnn_autogenlib.md
      onefilellm: onefilellm/onefilellm_gnn.md
      poe_world: poe-world/poe-world_gnn.md
    audio_processing:
      sapf: sapf/sapf_gnn.md
      pedalboard: pedalboard/pedalboard_gnn.md
    formal_methods:
      axiom: axiom/axiom_gnn.md
      petri_nets: petri_nets/pnml.pnml
      nock: nock/nock-gnn.md
    distributed_systems:
      iroh: iroh/iroh.md
      x402: x402/gnn_x402.md
    specialized_tools:
      gui_oxdraw: gui_oxdraw/gnn_oxdraw.md
      vec2text: vec2text/vec2text_gnn.md
      klong: klong/klong.md
      arc_agi: arc-agi/arc-agi-gnn.md
  
  research_integration:
    compositional_modeling:
      primary: poe-world/poe-world.md
      integration: poe-world/poe-world_gnn.md
      related: [gnn/advanced/advanced_modeling_patterns.md, templates/hierarchical_template.md]
    neurosymbolic_ai:
      primary: gnn/advanced/gnn_llm_neurosymbolic_active_inference.md
      integrations: [dspy/gnn_dspy.md, autogenlib/gnn_autogenlib.md]
    cognitive_modeling:
      primary: cognitive_phenomena/README.md
      models: [cognitive_phenomena/attention/attention_model.md, cognitive_phenomena/consciousness/global_workspace_model.md, cognitive_phenomena/meta-awareness/meta_aware_model.md]
  
  topic_clusters:
    active_inference_theory: [gnn/about_gnn.md, gnn/gnn_paper.md, gnn/advanced/ontology_system.md]
    modeling_patterns: [gnn/advanced/advanced_modeling_patterns.md, templates/README.md]
    cognitive_phenomena: [cognitive_phenomena/README.md, cognitive_phenomena/*/README.md]
    technical_implementation: [gnn/reference/gnn_syntax.md, gnn/operations/gnn_tools.md, gnn/operations/gnn_tools.md]
    data_persistence: [pkl/pkl_gnn.md, pkl/examples/]
    audio_sonification: [sapf/sapf_gnn.md, pedalboard/pedalboard_gnn.md]
    formal_methods: [axiom/axiom_gnn.md, petri_nets/pnml.pnml, nock/nock-gnn.md]
    distributed_systems: [iroh/iroh.md, x402/gnn_x402.md]
    interactive_tools: [gui_oxdraw/gnn_oxdraw.md, glowstick/glowstick_gnn.md]
    
  support_network:
    troubleshooting: [troubleshooting/README.md, troubleshooting/common_errors.md, troubleshooting/faq.md]
    learning: [tutorials/README.md, gnn/tutorials/quickstart_tutorial.md, quickstart.md, learning_paths.md]
    community: [../CONTRIBUTING.md, ../SUPPORT.md]
    quality_assurance: [style_guide.md, testing/README.md]

  pipeline_integration:
    note: "Step indices are 0-24; see CLAUDE.md for authoritative ordering"
    core_examples: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    simulation_examples: [10, 11, 12, 13, 14, 15, 16]
    output_examples: [17, 18, 19, 20, 21, 22, 23, 24]
    
  cross_reference_density:
    high_density: [README.md, gnn/reference/gnn_syntax.md, templates/README.md, gnn/operations/gnn_tools.md, learning_paths.md]
    medium_density: [gnn/advanced/advanced_modeling_patterns.md, cognitive_phenomena/README.md, CROSS_REFERENCE_INDEX.md]
    specialized: [poe-world/poe-world_gnn.md, cerebrum/gnn_cerebrum.md, axiom/axiom_gnn.md]
    emerging: [onefilellm/onefilellm_gnn.md, vec2text/vec2text_gnn.md, arc-agi/arc-agi-gnn.md]
```

## Topic README anchor registry

These headings exist so deep links from `doc/<topic>/README.md` resolve to a stable fragment. Prefer narrative sections above when browsing.

### activeinference_jl

See **ActiveInference.jl** under [Specialized Frameworks](#specialized-frameworks).

### Advanced visualization

See [Technical Implementation](#technical-implementation) and [visualization/README.md](visualization/README.md).

### API reference integration

See [API Documentation](api/README.md) and the doc hub [API Reference & Integration](README.md#api-reference-integration).

### Arc-agi

See [ARC-AGI Integration](arc-agi/arc-agi-gnn.md) under [Specialized Frameworks](#specialized-frameworks).

### Archive

Historical examples: [archive/README.md](archive/README.md).

### Autogenlib

See [AutoGenLib](autogenlib/gnn_autogenlib.md) under [LLM Integrations](#llm-integrations).

### Axiom

See [Axiom](axiom/axiom_gnn.md) under [Specialized Frameworks](#specialized-frameworks) and [Mathematical Foundations](#mathematical-foundations).

### Cerebrum

See [Cerebrum](cerebrum/gnn_cerebrum.md) in [Modeling Patterns](#modeling-patterns).

### D2

See [D2](d2/gnn_d2.md) under [Specialized Frameworks](#specialized-frameworks).

### Data processing

See [OneFileLLM](onefilellm/onefilellm_gnn.md) and [Neurosymbolic AI](#neurosymbolic-ai).

### Dspy

See [DSPy](dspy/gnn_dspy.md) under [LLM Integrations](#llm-integrations).

### Embedding systems

See [Vec2Text](vec2text/vec2text_gnn.md) and [Neurosymbolic AI](#neurosymbolic-ai).

### Export

See [export/README.md](export/README.md) and [Technical Implementation](#technical-implementation).

### Formal methods and verification

See [Mathematical Foundations](#mathematical-foundations) and formal tooling under [Specialized Frameworks](#specialized-frameworks).

### Glowstick

See [Glowstick](glowstick/glowstick_gnn.md) under [Specialized Frameworks](#specialized-frameworks).

### gui_oxdraw

See [GUI Oxdraw](gui_oxdraw/gnn_oxdraw.md) under [Specialized Frameworks](#specialized-frameworks).

### Iroh

See [Iroh](iroh/iroh.md) under [Distributed Systems and Networking](#distributed-systems-and-networking).

### Kit

See [Kit](kit/gnn_kit.md) under [Specialized Frameworks](#specialized-frameworks).

### Klong

See [Klong](klong/klong.md) under [Specialized Frameworks](#specialized-frameworks).

### muscle-mem

See [Muscle-Mem](muscle-mem/gnn-muscle-mem.md) under [Specialized Frameworks](#specialized-frameworks).

### Neuroscience

See [Neuroscience Integration](#neuroscience-integration) and [SPM](spm/spm_gnn.md).

### Nock

See [Nock](nock/nock-gnn.md) under [Mathematical Foundations](#mathematical-foundations).

### Onefilellm

See [OneFileLLM](onefilellm/onefilellm_gnn.md) under [LLM Integrations](#llm-integrations).

### Other

Miscellaneous topics: [other/README.md](other/README.md).

### Pedalboard

See [Pedalboard](pedalboard/pedalboard_gnn.md) under [Audio and Sonification](#audio-and-sonification).

### Performance optimization

See [Performance Guide](performance/README.md), [TimEP](timep/timep_gnn.md), and [Muscle-Mem](muscle-mem/gnn-muscle-mem.md).

### Pkl

See [Pkl](pkl/pkl_gnn.md) under [Data Persistence and Serialization](#data-persistence-and-serialization).

### Poe-world

See [PoE-World](poe-world/poe-world.md) under [Compositional World Modeling](#compositional-world-modeling).

### Pomdp

See [POMDP](pomdp/pomdp_overall.md) under [Mathematical Foundations](#mathematical-foundations) and [Temporal and Dynamical Systems](#temporal-and-dynamical-systems).

### Privacy and interpretability

See [Vec2Text](vec2text/vec2text_gnn.md) and privacy-related notes in [Neurosymbolic AI](#neurosymbolic-ai).

### Quadray

See [Quadray](quadray/quadray_gnn.md) under [Mathematical Foundations](#mathematical-foundations).

### Release management

See [releases/README.md](releases/README.md) and [Support and Learning](#support-and-learning).

### Research

See [Research Integration Network](#research-integration-network) and [Researcher Path](#researcher-path).

### Sapf

See [SAPF](sapf/sapf_gnn.md) under [Audio and Sonification](#audio-and-sonification).

### Spatial modeling

See [Quadray](quadray/quadray_gnn.md) and spatial patterns in [Modeling Patterns](#modeling-patterns).

### Spm

See [SPM](spm/spm_gnn.md) under [Neuroscience Integration](#neuroscience-integration).

### Sympy

See [SymPy](sympy/gnn_sympy.md) under [Mathematical Foundations](#mathematical-foundations).

### Timep

See [TimEP](timep/timep_gnn.md) under [Temporal and Dynamical Systems](#temporal-and-dynamical-systems).

### Type inference zoo

See [type-inference-zoo/README.md](type-inference-zoo/README.md) under [Mathematical Foundations](#mathematical-foundations).

### Vec2text

See [Vec2Text](vec2text/vec2text_gnn.md) under [LLM Integrations](#llm-integrations).

### Visualization

See [visualization/README.md](visualization/README.md) under [Technical Implementation](#technical-implementation).

### X402

See [X402](x402/gnn_x402.md) under [Distributed Systems and Networking](#distributed-systems-and-networking).

## Quality Metrics

### Documentation Coverage
>
> **📊 Comprehensive Analysis** | **✅ Quality Assurance**

| Category | Documents | Cross-References | Coverage Level |
|----------|-----------|------------------|----------------|
| **Core GNN** | 16 | 220+ | ✅ Excellent |
| **Framework Integration** | 12+ | 190+ | ✅ Excellent |
| **Templates** | 5 | 90+ | ✅ Complete |
| **Cognitive Phenomena** | 22+ | 120+ | ✅ Comprehensive |
| **Pipeline Documentation** | 3 | 70+ | ✅ Complete |
| **Research Integration** | 12+ | 60+ | ✅ Excellent |
| **Support & Troubleshooting** | 8 | 130+ | ✅ Excellent |
| **Tool Integration** | 24+ | 150+ | ✅ Comprehensive |
| **Audio & Sonification** | 2 | 40+ | ✅ Complete |
| **Formal Methods** | 3 | 45+ | ✅ Complete |
| **Distributed Systems** | 2 | 35+ | ✅ Complete |
| **Infrastructure** | 2 | 30+ | ✅ Complete |

### Documentation Completeness
>
> **✅ Comprehensive Integration** | **📊 Complete Coverage**

- **Coverage**: [expected_dirs.txt](expected_dirs.txt) lists 61 top-level `doc/` folders; pairing of README/AGENTS is enforced by [docs_audit.py](development/docs_audit.py) for maintained trees
- **Sections**: Audio processing, formal methods, distributed systems, infrastructure
- **Audio**: Sonification frameworks (SAPF, Pedalboard) and audio representation
- **Formal Methods**: Verification frameworks (Axiom, Petri Nets, Nock) and formal specification
- **Distributed Systems**: Networking (Iroh, X402) and decentralized coordination
- **Utilities**: Analysis tools (OneFileLLM, Vec2Text, ARC-AGI, D2, GUI-Oxdraw)
- **Temporal**: Temporal modeling (TimEP) and time series analysis
- **Neuroscience**: SPM integration and neuroscientific methods
- **YAML Integration**: Framework integrations with tool categories
- **Topic Clusters**: Comprehensive topic organization and emerging categories
- **Metrics**: Cross-reference density varies by subtree; run `docs_audit.py --strict` for mechanical checks

---

**Status**: Maintained — prefer [CLAUDE.md](../CLAUDE.md) and [PIPELINE_SCRIPTS.md](PIPELINE_SCRIPTS.md) for pipeline truth  
**Top-level doc folders**: 61 ([expected_dirs.txt](expected_dirs.txt))
