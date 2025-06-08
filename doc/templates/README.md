# GNN Template System

## Overview
This directory contains template files for different types of GNN models, providing standardized starting points for model development.

## Available Templates

### Basic GNN Template
**File**: `basic_gnn_template.md`
**Use Case**: Simple static perception models, learning GNN syntax
**Features**: 
- Minimal StateSpaceBlock with 2-3 variables
- Basic Connections section
- Example InitialParameterization
- Clear annotations

### POMDP Template
**File**: `pomdp_template.md` *(to be created)*
**Use Case**: Partially Observable Markov Decision Processes
**Features**:
- Hidden states and observations
- Action/policy variables
- Belief state representations
- Temporal dynamics

### Multi-agent Template  
**File**: `multiagent_template.md` *(to be created)*
**Use Case**: Multiple interacting agents
**Features**:
- Agent-specific state spaces
- Communication variables
- Shared environment representations
- Coordination mechanisms

### Hierarchical Template
**File**: `hierarchical_template.md` *(to be created)*
**Use Case**: Multi-level cognitive architectures
**Features**:
- Multiple temporal scales
- Hierarchical state representations
- Top-down and bottom-up connections
- Cross-level interactions

## Using Templates

### Method 1: Copy and Modify
```bash
cp doc/templates/basic_gnn_template.md my_model.md
# Edit my_model.md with your specific model details
```

### Method 2: Pipeline Generation
```bash
# Generate from template with pipeline
python src/main.py --template basic --output my_new_model.md
```

### Method 3: Interactive Creation
```bash
# Use interactive template wizard
python src/utils/template_wizard.py
```

## Template Structure

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

## Customization Guidelines

### State Space Variables
- Use descriptive names: `perception_f0`, `action_policy_c0`
- Include dimensions: `[3,1,type=int]`
- Add comments for clarity: `### Represents visual attention`

### Connections
- Use consistent directional notation: `A > B` (A influences B)
- Group related connections for readability
- Document connection semantics

### Parameters
- Provide sensible defaults
- Include parameter ranges where applicable
- Reference relevant literature for parameter choices

### Equations
- Use LaTeX for mathematical expressions
- Include probability distributions explicitly
- Document variable meanings

## Validation

All templates are validated using the type checker:

```bash
python src/main.py --only-steps 4 --target-dir doc/templates/
```

Templates should pass basic syntax validation and resource estimation.

## Contributing Templates

### Creating New Templates
1. Follow the standard structure above
2. Include comprehensive annotations
3. Test with the type checker
4. Add example use cases
5. Update this README

### Template Naming Convention
- Use descriptive names: `multimodal_perception_template.md`
- Include complexity level: `advanced_hierarchical_template.md`
- Specify domain if applicable: `robotics_navigation_template.md`

### Review Process
1. Submit template via pull request
2. Include example model based on template
3. Documentation review for clarity
4. Technical review for GNN compliance
5. Integration into template system

## Template Metadata

Each template includes metadata for automatic categorization:

```yaml
# Template Metadata (in comments)
# Complexity: basic|intermediate|advanced
# Domain: general|robotics|nlp|vision|multiagent
# Features: temporal|hierarchical|multimodal|stochastic
# Prerequisites: basic_gnn|pomdp_fundamentals|category_theory
```

## Related Documentation

- [GNN Syntax Reference](../gnn/gnn_syntax.md)
- [Model Examples](../gnn/gnn_examples_doc.md)
- [Implementation Guide](../gnn/gnn_implementation.md)
- [Type Checker Documentation](../gnn/gnn_tools.md#validation-tools) 