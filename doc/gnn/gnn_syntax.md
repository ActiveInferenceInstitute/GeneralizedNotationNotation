# GNN Syntax and Punctuation Reference

> **📋 Document Metadata**  
> **Type**: Specification | **Audience**: All Users | **Complexity**: Core Reference  
> **Last Updated**: January 2025 | **Status**: Production-Ready  
> **Cross-References**: [File Structure](gnn_file_structure_doc.md) | [Examples](gnn_examples_doc.md) | [Implementation Guide](gnn_implementation.md)

## Overview

> **🎯 Purpose**: Complete notation specification for GNN language  
> **📊 Scope**: All syntax elements, punctuation, and naming conventions  
> **✅ Status**: Comprehensive and validated reference

This document provides a comprehensive reference for the syntax and punctuation used in Generalized Notation Notation (GNN) files.

## Punctuation and Operators

> **⚡ Quick Reference**: Core symbols and their meanings

| Symbol | Name                | Purpose                             | Example       | Description                              |
|--------|---------------------|-------------------------------------|---------------|------------------------------------------|
| _      | Underscore          | Subscript notation                  | s_f0          | State factor 0                           |
| ^      | Caret               | Superscript notation                | X^T           | X transpose                              |
| >      | Greater than        | Directed causal edge                | X>Y           | X influences Y                           |
| -      | Hyphen              | Undirected causal edge              | X-Y           | Undirected relation between X and Y      |
| ( )    | Parentheses         | Grouping                            | (X+Y)         | Parenthesized expression                 |
| { }    | Curly braces        | Exact value specification           | X{1}          | X equals 1 exactly                       |
| [ ]    | Square brackets     | Dimensionality or indexing          | X[2,3]        | X is a 2×3 matrix                        |
| #      | Hash/Pound          | Title header                        | # Title       | Top-level section header                 |
| ##     | Double hash         | Section header                      | ## Section    | Section header                           |
| ###    | Triple hash         | Comment line                        | ### Comment   | Comment line                             |
| +      | Plus                | Addition                            | X+Y           | Sum of X and Y                           |
| *      | Asterisk            | Multiplication                      | X*Y           | Product of X and Y                       |
| /      | Forward slash       | Division                            | X/Y           | X divided by Y                           |
| \|     | Vertical bar        | Conditional probability             | P(X\|Y)       | Probability of X given Y                 |
| →      | Arrow (ASCII: ->)   | Direction or sequence               | X→Y           | X leads to Y                             |
| ∈      | Element of (ASCII: in) | Set membership                   | x∈X           | x is an element of set X                 |
| ∧      | Logical AND (ASCII: &) | Logical conjunction              | A∧B           | A and B                                  |
| ∨      | Logical OR (ASCII: \|) | Logical disjunction              | A∨B           | A or B                                   |

**Cross-References**: 
- *Mathematical Expressions*: [Advanced Patterns - Mathematical Foundations](advanced_modeling_patterns.md#mathematical-foundations)
- *Practical Usage*: [Examples](gnn_examples_doc.md) | [Templates](../templates/README.md)

## Variable Naming Conventions

> **📝 Standardized Naming**: Active Inference concepts to GNN variables

### State Variables (s)
> **🧠 Hidden States**: Internal model representations

State factors represent hidden states in Active Inference models:

- **Format**: `s_f{factor_index}[dimensions,type]`
- **Examples**: 
  - `s_f0[3,1,type=int]` - 3-dimensional discrete state factor 0
  - `s_f1[2,2,type=float]` - 2×2 continuous state factor 1

**Cross-References**: 
- *Active Inference Theory*: [About GNN](about_gnn.md) | [Academic Paper](gnn_paper.md)
- *Implementation*: [Templates - State Variables](../templates/README.md#state-space-variables)
- *Framework Integration*: [PyMDP States](../pymdp/gnn_pymdp.md#state-representation) | [RxInfer Factors](../rxinfer/gnn_rxinfer.md#factor-graphs)

### Observation Variables (o)
> **👁️ Observable Outcomes**: Sensory modalities

Observation modalities represent observable outcomes:

- **Format**: `o_m{modality_index}[dimensions,type]`
- **Examples**:
  - `o_m0[5,1,type=int]` - 5-dimensional discrete observation modality 0
  - `o_m1[64,64,type=float]` - 64×64 continuous observation (e.g., image)

**Cross-References**: 
- *Perceptual Models*: [POMDP Template](../templates/pomdp_template.md)
- *Multi-modal*: [Advanced Patterns - Multi-modal Processing](advanced_modeling_patterns.md#multi-modal-processing)

### Action/Control Variables (u, π)
> **⚙️ Control Systems**: Actions and policies

Control factors and policies for action selection:

- **Actions**: `u_c{control_index}[dimensions,type]`
- **Policies**: `π_c{control_index}[dimensions,type]`
- **Examples**:
  - `u_c0[3,1,type=int]` - 3-action discrete control
  - `π_c0[3,T,type=float]` - Policy over 3 actions and T time steps

**Cross-References**: 
- *Decision Theory*: [Advanced Patterns - Decision Theory](advanced_modeling_patterns.md#decision-theory)
- *Multi-agent*: [Multi-agent Template](../templates/multiagent_template.md)

### Matrix Variables (A, B, C, D)
> **🎛️ Model Parameters**: Core Active Inference matrices

Standard Active Inference parameter matrices:

- **A matrices**: `A_m{modality}[obs_dim,state_dim,type=float]` - Likelihood matrices
- **B matrices**: `B_f{factor}[next_state,current_state,action,type=float]` - Transition matrices  
- **C matrices**: `C_m{modality}[obs_dim,type=float]` - Preference vectors
- **D matrices**: `D_f{factor}[state_dim,type=float]` - Prior beliefs

**Cross-References**: 
- *Parameter Design*: [Implementation Guide - Parameter Specification](gnn_implementation.md#parameter-specification)
- *Framework Integration*: [PyMDP Matrices](../pymdp/gnn_pymdp.md#matrix-specification) | [RxInfer Parameters](../rxinfer/gnn_rxinfer.md#parameter-specification)

### Precision Parameters (γ, α, β)
> **🎯 Uncertainty Control**: Precision and learning rates

Parameters controlling uncertainty and learning:

- **γ (gamma)**: Action precision
- **α (alpha)**: Learning rate
- **β (beta)**: Inverse temperature
- **Examples**: `γ_action[1,type=float]`, `α_learn[1,type=float]`

**Cross-References**: 
- *Advanced Learning*: [Advanced Patterns - Learning Algorithms](advanced_modeling_patterns.md#learning-algorithms)
- *Parameter Tuning*: [Performance Guide](../troubleshooting/performance.md)

## Dimensionality Specification

> **📐 Array Dimensions**: Matrix and tensor specifications

### Basic Format
```
variable_name[dim1,dim2,...,dimN,type=data_type]
```

### Dimension Types
- **Scalar**: `[1,type=float]` - Single value
- **Vector**: `[n,type=int]` - 1D array of length n
- **Matrix**: `[m,n,type=float]` - 2D array m×n
- **Tensor**: `[m,n,p,type=float]` - 3D array m×n×p

**Cross-References**: 
- *Type System*: [Type Checker](gnn_tools.md#validation-tools) | [Resource Estimation](resource_metrics.md)
- *Framework Mapping*: [PyMDP Arrays](../pymdp/gnn_pymdp.md#array-handling) | [RxInfer Tensors](../rxinfer/gnn_rxinfer.md#tensor-operations)

### Data Types
- **type=int**: Discrete/categorical variables
- **type=float**: Continuous variables  
- **type=bool**: Boolean variables
- **type=complex**: Complex numbers (advanced)

**Cross-References**: 
- *Validation*: [Common Errors - Type Mismatches](../troubleshooting/common_errors.md#type-errors)
- *Framework Compatibility*: [Framework Integration Matrix](../templates/README.md#framework-integration-matrix)

## Connection Notation

> **🔗 Causal Relationships**: Variable dependencies and influences

### Directed Connections
**Syntax**: `X > Y` (X influences Y)

**Examples**:
```
s_f0 > o_m0    ### State factor 0 generates observation modality 0
u_c0 > s_f1    ### Action influences state factor 1
```

**Cross-References**: 
- *Causal Modeling*: [Advanced Patterns - Causal Inference](advanced_modeling_patterns.md#causal-inference)
- *Graphical Models*: [Visualization Guide](gnn_tools.md#visualization-tools)

### Undirected Connections  
**Syntax**: `X - Y` (X and Y are related)

**Examples**:
```
s_f0 - s_f1    ### State factors are correlated
o_m0 - o_m1    ### Observation modalities share information
```

### Complex Connection Patterns
**Multi-target**: `X > Y, Z` (X influences both Y and Z)
**Chain**: `X > Y > Z` (Causal chain)
**Conditional**: `X > Y | Z` (X influences Y given Z)

**Cross-References**: 
- *Complex Models*: [Hierarchical Template](../templates/hierarchical_template.md)
- *DisCoPy Integration*: [Categorical Diagrams](../discopy/gnn_discopy.md)

## Mathematical Expressions

> **📐 LaTeX Integration**: Mathematical notation in GNN

### Probability Notation
```latex
P(o_m0|s_f0)           # Likelihood
P(s_f0^{t+1}|s_f0^t,u_c0)  # Transition probability
```

### Equations Section Format
```markdown
## Equations
\[
P(o_t|s_t) = \text{Cat}(A_{m0} s_t)
\]
\[
P(s_{t+1}|s_t,u_t) = \text{Cat}(B_{f0}[:,:,u_t] s_t)
\]
```

**Cross-References**: 
- *Mathematical Foundations*: [Advanced Patterns - Mathematical Foundations](advanced_modeling_patterns.md#mathematical-foundations)
- *LaTeX Rendering*: [Site Generation](../pipeline/README.md#step-14-static-site-generation)

### Common Mathematical Constructs
- **Categorical Distribution**: `Cat(parameters)`
- **Normal Distribution**: `𝒩(μ,σ²)`
- **Dirichlet Distribution**: `Dir(α)`
- **Expected Value**: `𝔼[X]`
- **KL Divergence**: `D_{KL}(P||Q)`

**Cross-References**: 
- *Distributions*: [PyMDP Distributions](../pymdp/gnn_pymdp.md#probability-distributions) | [RxInfer Distributions](../rxinfer/gnn_rxinfer.md#distribution-specification)

## Comments and Annotations

> **📝 Documentation**: Inline comments and explanations

### Comment Types
- **Line Comments**: `### This is a comment`
- **Section Comments**: `### --- Section Divider ---`
- **Variable Annotations**: `s_f0[3,1,type=int] ### Visual attention state`

### Annotation Best Practices
1. **Describe variable purpose**: What cognitive process does this represent?
2. **Explain connections**: Why does X influence Y?
3. **Document parameters**: What do these values represent?
4. **Reference literature**: Cite relevant papers

**Cross-References**: 
- *Documentation Standards*: [Contributing Guide](../../CONTRIBUTING.md) | [Style Guide](../style_guide.md)

## File Organization

> **📁 Structure**: How syntax elements fit together

### Section Headers
```markdown
# GNNVersionAndFlags
## ModelName  
## StateSpaceBlock
## Connections
## InitialParameterization
## Equations
## Time
```

**Cross-References**: 
- *Complete Structure*: [File Structure Guide](gnn_file_structure_doc.md)
- *Templates*: [Template System](../templates/README.md)

### Content Organization
1. **Variables first**: Define all variables before connections
2. **Group by type**: States, observations, actions, parameters
3. **Logical flow**: Match cognitive process sequence
4. **Clear separation**: Use comments to divide sections

**Cross-References**: 
- *Best Practices*: [Implementation Guide](gnn_implementation.md)
- *Examples*: [Model Examples](gnn_examples_doc.md)

## Validation and Error Checking

> **✅ Quality Assurance**: Syntax validation and error prevention

### Common Syntax Errors
1. **Undefined variables**: Using variables not declared in StateSpaceBlock
2. **Dimension mismatches**: Incompatible matrix dimensions
3. **Type errors**: Mixing int and float inappropriately
4. **Missing connections**: Variables without clear relationships

**Cross-References**: 
- *Error Resolution*: [Common Errors](../troubleshooting/common_errors.md)
- *Validation Tools*: [Type Checker](gnn_tools.md#validation-tools)

### Validation Tools
```bash
# Validate GNN syntax
python src/main.py --only-steps 4 --target-dir my_model.md

# Check specific syntax elements
python src/gnn_type_checker/validator.py --syntax-only my_model.md
```

**Cross-References**: 
- *Pipeline Validation*: [Pipeline Step 4](../pipeline/README.md#step-4-gnn-type-checker)
- *Development Workflow*: [Development Guide](../development/README.md)

## Advanced Syntax Features

> **🚀 Extended Capabilities**: Advanced notation patterns

### Temporal Indexing
```
s_f0^t      # State at time t
s_f0^{t+1}  # State at next time step
o_m0_{1:T}  # Observations from time 1 to T
```

### Conditional Dependencies
```
X > Y | Z   # X influences Y given Z
P(Y|X,Z)    # Conditional probability notation
```

### Hierarchical Notation
```
s_f0_level1 > s_f0_level2  # Hierarchical state relationship
```

**Cross-References**: 
- *Temporal Models*: [Advanced Patterns - Temporal Dynamics](advanced_modeling_patterns.md#temporal-dynamics)
- *Hierarchical Models*: [Hierarchical Template](../templates/hierarchical_template.md)
- *Cognitive Architectures*: [Cerebrum Integration](../cerebrum/gnn_cerebrum.md)

## Framework-Specific Considerations

> **🔗 Cross-Platform Compatibility**: Framework-specific syntax notes

### PyMDP Compatibility
- Use discrete types for POMDP variables
- Matrix dimensions must match PyMDP conventions
- Action indexing starts from 0

**Cross-References**: [PyMDP Integration Guide](../pymdp/gnn_pymdp.md)

### RxInfer.jl Compatibility  
- Supports both discrete and continuous variables
- Factor graph notation maps directly
- Julia-style indexing considerations

**Cross-References**: [RxInfer Integration Guide](../rxinfer/gnn_rxinfer.md)

### DisCoPy Compatibility
- Variables become types in category theory
- Connections become morphisms
- Compositional structure preserved

**Cross-References**: [DisCoPy Integration Guide](../discopy/gnn_discopy.md)

## Related Documentation

> **🔗 Comprehensive Cross-Reference Network**

### Core GNN Documentation
- **[File Structure Guide](gnn_file_structure_doc.md)** - How syntax fits into file organization
- **[Examples](gnn_examples_doc.md)** - Syntax usage in practice  
- **[Implementation Guide](gnn_implementation.md)** - Best practices for syntax usage
- **[Tools and Resources](gnn_tools.md)** - Validation and processing tools

### Templates and Patterns
- **[Template System](../templates/README.md)** - Syntax examples in templates
- **[Advanced Patterns](advanced_modeling_patterns.md)** - Complex syntax usage
- **[Basic Template](../templates/basic_gnn_template.md)** - Simple syntax examples

### Framework Integration
- **[PyMDP Guide](../pymdp/gnn_pymdp.md)** - PyMDP-specific syntax considerations
- **[RxInfer Guide](../rxinfer/gnn_rxinfer.md)** - RxInfer.jl syntax mapping
- **[DisCoPy Guide](../discopy/gnn_discopy.md)** - Category theory syntax

### Development and Support
- **[Type Checker](gnn_tools.md#validation-tools)** - Syntax validation tools
- **[Common Errors](../troubleshooting/common_errors.md)** - Syntax error troubleshooting
- **[FAQ](../troubleshooting/faq.md)** - Frequently asked syntax questions

---

## 📊 Syntax Reference Metadata

> **🏷️ Machine-Readable Syntax Data**

```yaml
syntax_specification:
  version: "1.0"
  operators:
    causal_directed: ">"
    causal_undirected: "-"
    subscript: "_"
    superscript: "^"
    grouping: ["()", "[]", "{}"]
    mathematical: ["+", "-", "*", "/", "|"]
  variable_types:
    states: "s_f{index}"
    observations: "o_m{index}"
    actions: "u_c{index}"
    policies: "π_c{index}"
    matrices: ["A", "B", "C", "D"]
    precision: ["γ", "α", "β"]
  data_types: ["int", "float", "bool", "complex"]
  dimension_format: "[dim1,dim2,...,type=datatype]"
  comment_syntax: "###"
  section_headers: "##"
  validation_tools: ["type_checker", "syntax_validator"]
  framework_compatibility:
    pymdp: "full"
    rxinfer: "full"  
    discopy: "full"
```

---

**Last Updated**: January 2025  
**Syntax Version**: GNN-1.0  
**Status**: Production-Ready Specification  
**Cross-Reference Network**: ✅ Fully Integrated 