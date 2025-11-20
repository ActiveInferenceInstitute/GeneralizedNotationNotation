# GNN Template System

> **📋 Document Metadata**  
> **Type**: Template Library | **Audience**: Practitioners, Developers | **Complexity**: Intermediate  
> **Cross-References**: [Implementation Guide](../gnn/gnn_implementation.md) | [Examples](../gnn/gnn_examples_doc.md) | [Syntax Reference](../gnn/gnn_syntax.md)

## Overview

> **🎯 Purpose**: Standardized starting points for GNN model development  
> **📊 Total Templates**: 4 comprehensive templates (49KB documentation)  
> **✅ Coverage**: All major use cases (basic, POMDP, multi-agent, hierarchical)

This directory contains template files for different types of GNN models, providing standardized starting points for model development.

## Available Templates

> **📈 Progressive Complexity**: Basic → POMDP → Multi-agent → Hierarchical  
> **🎯 Learning Integration**: Templates align with [Learning Paths Guide](../learning_paths.md) progression

### Basic GNN Template
> **📋 Metadata**: Complexity: Beginner | Domain: General | Features: Static  
> **🎯 Learning Path**: Part of [Beginner Path](../learning_paths.md#beginner-path)

**File**: `basic_gnn_template.md` (4.4KB, 141 lines)  
**Use Case**: Simple static perception models, learning GNN syntax  
**Features**: 
- Minimal StateSpaceBlock with 2-3 variables
- Basic Connections section
- Example InitialParameterization
- Clear annotations

**Cross-References**:
- *Learning Path*: [Quickstart Tutorial](../quickstart.md) → [Basic Examples](../gnn/gnn_examples_doc.md)
- *Framework Integration*: [PyMDP Basics](../pymdp/gnn_pymdp.md#basic-examples) | [RxInfer Getting Started](../rxinfer/gnn_rxinfer.md#getting-started)
- *Validation*: [Type Checker](../gnn/gnn_tools.md#validation-tools)

### POMDP Template
> **📋 Metadata**: Complexity: Intermediate | Domain: Navigation, Perception | Features: Temporal, Stochastic  
> **🎯 Learning Path**: Part of [Intermediate Path](../learning_paths.md#intermediate-path)

**File**: `pomdp_template.md` (9.5KB, 240 lines)  
**Use Case**: Partially Observable Markov Decision Processes  
**Features**:
- Hidden states and observations
- Action/policy variables
- Belief state representations
- Temporal dynamics with detailed examples

**Cross-References**:
- *Example Applications*: [Butterfly Agent](../archive/gnn_example_butterfly_pheromone_agent.md) | [Trading Agent](../archive/gnn_airplane_trading_pomdp.md)
- *Framework Integration*: [PyMDP POMDP](../pymdp/gnn_pymdp.md#pomdp-examples) | [RxInfer POMDP](../rxinfer/gnn_rxinfer.md#pomdp-models)
- *Advanced Patterns*: [Decision Theory](../gnn/advanced_modeling_patterns.md#decision-theory)

### Multi-agent Template  
> **📋 Metadata**: Complexity: Advanced | Domain: Multi-agent, Robotics | Features: Multimodal, Communication

**File**: `multiagent_template.md` (17KB, 393 lines)  
**Use Case**: Multiple interacting agents  
**Features**:
- Agent-specific state spaces
- Communication variables
- Shared environment representations
- Coordination mechanisms with extensive examples

**Cross-References**:
- *Theory*: [Multi-agent Systems](../gnn/gnn_multiagent.md)
- *Framework Integration*: [RxInfer Multi-agent](../rxinfer/multiagent_trajectory_planning/) | [PyMDP Coordination](../pymdp/gnn_pymdp.md#multi-agent-examples)
- *Advanced Applications*: [Cognitive Phenomena](../cognitive_phenomena/README.md) | [Social Modeling](../gnn/advanced_modeling_patterns.md#social-cognition)

### Hierarchical Template
> **📋 Metadata**: Complexity: Expert | Domain: Cognitive Architecture | Features: Hierarchical, Multi-scale

**File**: `hierarchical_template.md` (19KB, 412 lines)  
**Use Case**: Multi-level cognitive architectures  
**Features**:
- Multiple temporal scales
- Hierarchical state representations
- Top-down and bottom-up connections
- Cross-level interactions with comprehensive modeling patterns

**Cross-References**:
- *Applications*: [Cerebrum Integration](../cerebrum/gnn_cerebrum.md) | [Language Model](../archive/gnn_active_inference_language_model.md)
- *Advanced Patterns*: [Hierarchical Modeling](../gnn/advanced_modeling_patterns.md#hierarchical-architectures)
- *Framework Integration*: [RxInfer Hierarchical](../rxinfer/gnn_rxinfer.md#hierarchical-models) | [DisCoPy Composition](../discopy/gnn_discopy.md)

## Using Templates

> **🛠️ Multiple Approaches**: Copy & Edit | Pipeline Generation | Interactive Creation

### Method 1: Copy and Modify
```bash
cp doc/templates/basic_gnn_template.md my_model.md
# Edit my_model.md with your specific model details
```

**Cross-References**: [File Structure Guide](../gnn/gnn_file_structure_doc.md) | [Syntax Reference](../gnn/gnn_syntax.md)

### Method 2: Pipeline Generation
```bash
# Generate from template with pipeline
python src/main.py --template basic --output my_new_model.md
```

**Cross-References**: [Pipeline Guide](../pipeline/README.md) | [Configuration](../configuration/README.md)

### Method 3: Interactive Creation
```bash
# Use interactive template wizard
python src/utils/template_wizard.py
```

**Cross-References**: [API Documentation](../api/README.md) | [Tool Development](../gnn/gnn_dsl_manual.md)

## Template Structure

> **📋 Standardized Format**: All templates follow GNN specification structure

All templates follow this standardized structure:

```markdown
# GNNVersionAndFlags
GNN-1.0

# ModelName
[Template Name] - [Your Model Name]

# ModelAnnotation
[Description of what this model represents and its intended use]

# StateSpaceBlock
[Variable definitions with dimensions and types]

# Connections
[Relationships between variables]

# InitialParameterization
[Starting parameter values and matrices]

# Equations
[Mathematical relationships in LaTeX]

# Time
[Temporal settings]

# ActInfOntologyAnnotation
[Ontology mappings]

# Footer
[Additional notes]

# Signature
[Author and creation information]
```

**Cross-References**: 
- *Specification*: [File Structure Guide](../gnn/gnn_file_structure_doc.md)
- *Syntax*: [GNN Syntax Reference](../gnn/gnn_syntax.md)
- *Ontology*: [Ontology System](../gnn/ontology_system.md)

## Customization Guidelines

> **🎯 Best Practices**: Naming → Structure → Validation → Integration

### State Space Variables
- **Use descriptive names**: `perception_f0`, `action_policy_c0`
- **Include dimensions**: `[3,1,type=int]`
- **Add comments for clarity**: `### Represents visual attention`

**Cross-References**: [Variable Naming](../gnn/gnn_syntax.md#variable-naming-conventions) | [Implementation Guide](../gnn/gnn_implementation.md#variable-design)

### Connections
- **Use consistent directional notation**: `A > B` (A influences B)
- **Group related connections** for readability
- **Document connection semantics**

**Cross-References**: [Connection Syntax](../gnn/gnn_syntax.md#connection-notation) | [Causal Modeling](../gnn/advanced_modeling_patterns.md#causal-inference)

### Parameters
- **Provide sensible defaults**
- **Include parameter ranges** where applicable
- **Reference relevant literature** for parameter choices

**Cross-References**: [Parameter Design](../gnn/gnn_implementation.md#parameter-specification) | [Performance Guide](../troubleshooting/performance.md)

### Equations
- **Use LaTeX** for mathematical expressions
- **Include probability distributions** explicitly
- **Document variable meanings**

**Cross-References**: [Equation Syntax](../gnn/gnn_syntax.md#mathematical-expressions) | [Mathematical Modeling](../gnn/advanced_modeling_patterns.md#mathematical-foundations)

## Validation

> **✅ Quality Assurance**: All templates validated through type checker

All templates are validated using the type checker:

```bash
python src/main.py --only-steps 4 --target-dir doc/templates/
```

Templates should pass basic syntax validation and resource estimation.

**Cross-References**: 
- *Type Checker*: [Validation Tools](../gnn/gnn_tools.md#validation-tools)
- *Pipeline*: [Step 4 Type Checker](../pipeline/README.md#step-4-gnn-type-checker)
- *Troubleshooting*: [Common Errors](../troubleshooting/common_errors.md)

## Contributing Templates

> **🤝 Community Contributions**: Standards → Review → Integration

### Creating New Templates
1. **Follow the standard structure** above
2. **Include comprehensive annotations**
3. **Test with the type checker**
4. **Add example use cases**
5. **Update this README**

**Cross-References**: [Contributing Guide](../../CONTRIBUTING.md) | [Development Guide](../development/README.md)

### Template Naming Convention
- **Use descriptive names**: `multimodal_perception_template.md`
- **Include complexity level**: `advanced_hierarchical_template.md`
- **Specify domain if applicable**: `robotics_navigation_template.md`

### Review Process
1. **Submit template** via pull request
2. **Include example model** based on template
3. **Documentation review** for clarity
4. **Technical review** for GNN compliance
5. **Integration** into template system

**Cross-References**: [Contributing Process](../../CONTRIBUTING.md#submitting-changes) | [Quality Assurance](../style_guide.md)

## Template Metadata

> **🏷️ Machine-Readable Classification System**

Each template includes metadata for automatic categorization:

```yaml
# Template Metadata (in comments)
# Complexity: basic|intermediate|advanced|expert
# Domain: general|robotics|nlp|vision|multiagent|cognitive
# Features: temporal|hierarchical|multimodal|stochastic|communication
# Prerequisites: basic_gnn|pomdp_fundamentals|category_theory|multi_agent_systems
# Framework_Compatibility: pymdp|rxinfer|discopy|all
# Use_Cases: [learning, navigation, perception, decision_making, coordination]
```

## Framework Integration Matrix

> **🔗 Cross-Platform Template Compatibility**

| Template | PyMDP | RxInfer | DisCoPy | Notes |
|----------|-------|---------|---------|-------|
| **Basic** | ✅ Full | ✅ Full | ✅ Full | All frameworks support basic models |
| **POMDP** | ✅ Native | ✅ Full | ✅ Partial | PyMDP specializes in POMDP |
| **Multi-agent** | ✅ Full | ✅ Native | ✅ Compositional | RxInfer excels at coordination |
| **Hierarchical** | ✅ Partial | ✅ Full | ✅ Native | DisCoPy ideal for composition |

**Cross-References**: 
- *PyMDP Integration*: [PyMDP Guide](../pymdp/gnn_pymdp.md)
- *RxInfer Integration*: [RxInfer Guide](../rxinfer/gnn_rxinfer.md)
- *DisCoPy Integration*: [DisCoPy Guide](../discopy/gnn_discopy.md)

## Learning Pathways

> **📚 Structured Progression**: Template-Based Learning

### Beginner Path
1. **[Basic Template](basic_gnn_template.md)** → **[Static Perception Example](../archive/gnn_example_dynamic_perception.md)**
2. **[Syntax Reference](../gnn/gnn_syntax.md)** → **[Type Checker](../gnn/gnn_tools.md#validation-tools)**
3. **[PyMDP Integration](../pymdp/gnn_pymdp.md#basic-examples)**

### Intermediate Path
1. **[POMDP Template](pomdp_template.md)** → **[Butterfly Agent](../archive/gnn_example_butterfly_pheromone_agent.md)**
2. **[Implementation Guide](../gnn/gnn_implementation.md)** → **[Advanced Patterns](../gnn/advanced_modeling_patterns.md)**
3. **Framework Choice**: [PyMDP POMDP](../pymdp/gnn_pymdp.md#pomdp-examples) or [RxInfer Navigation](../rxinfer/multiagent_trajectory_planning/)

### Advanced Path
1. **[Multi-agent Template](multiagent_template.md)** → **[Multi-agent Theory](../gnn/gnn_multiagent.md)**
2. **[Hierarchical Template](hierarchical_template.md)** → **[Cognitive Architectures](../cerebrum/gnn_cerebrum.md)**
3. **[Research Applications](../gnn/gnn_llm_neurosymbolic_active_inference.md)**

## Related Documentation

> **🔗 Comprehensive Cross-Reference Network**

### Core GNN Documentation
- **[GNN Syntax Reference](../gnn/gnn_syntax.md)** - Complete notation specification
- **[Model Examples](../gnn/gnn_examples_doc.md)** - Step-by-step examples using templates
- **[Implementation Guide](../gnn/gnn_implementation.md)** - Best practices for template customization
- **[Type Checker Documentation](../gnn/gnn_tools.md#validation-tools)** - Template validation

### Framework Integration
- **[PyMDP Integration](../pymdp/gnn_pymdp.md)** - Python Active Inference framework
- **[RxInfer Integration](../rxinfer/gnn_rxinfer.md)** - Julia Bayesian inference
- **[DisCoPy Integration](../discopy/gnn_discopy.md)** - Category theory and composition

### Advanced Topics
- **[Advanced Modeling Patterns](../gnn/advanced_modeling_patterns.md)** - Sophisticated techniques
- **[Multi-agent Systems](../gnn/gnn_multiagent.md)** - Multi-agent modeling theory
- **[Cognitive Phenomena](../cognitive_phenomena/README.md)** - Specialized applications

### Development & Support
- **[API Documentation](../api/README.md)** - Programming interfaces
- **[Pipeline Guide](../pipeline/README.md)** - Processing workflow
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving
- **[Contributing](../../CONTRIBUTING.md)** - Community contributions

---

## 📊 Template System Metadata

> **🏷️ Machine-Readable System Data**

```yaml
template_system:
  version: "1.0"
  total_templates: 4
  total_documentation: "49KB"
  coverage:
    complexity_levels: [basic, intermediate, advanced, expert]
    domains: [general, navigation, multiagent, cognitive]
    features: [static, temporal, hierarchical, multimodal, stochastic]
  framework_compatibility:
    pymdp: "100%"
    rxinfer: "100%"
    discopy: "95%"
  validation_status: "fully_tested"
  learning_paths:
    beginner: ["basic_gnn_template.md"]
    intermediate: ["pomdp_template.md"]
    advanced: ["multiagent_template.md", "hierarchical_template.md"]
  cross_references:
    syntax: "../gnn/gnn_syntax.md"
    examples: "../gnn/gnn_examples_doc.md"
    implementation: "../gnn/gnn_implementation.md"
    validation: "../gnn/gnn_tools.md#validation-tools"
```

---

**Template System Version**: 1.0  
**Status**: Production-Ready  
**Cross-Reference Network**: ✅ Fully Integrated 