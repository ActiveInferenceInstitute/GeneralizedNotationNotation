# Pkl Examples for GNN (Generalized Notation Notation)

This directory contains example [Apple Pkl](https://pkl-lang.org/) configuration files demonstrating how Pkl could enhance the GNN project for Active Inference models.

## 🔧 Prerequisites

Install Pkl from [pkl-lang.org](https://pkl-lang.org/main/current/pkl-cli/index.html):

**macOS (Apple Silicon):**
```bash
curl -L -o pkl https://github.com/apple/pkl/releases/download/0.28.2/pkl-macos-aarch64
chmod +x pkl
./pkl --version
```

**macOS (Intel):**
```bash
curl -L -o pkl https://github.com/apple/pkl/releases/download/0.28.2/pkl-macos-amd64
chmod +x pkl
./pkl --version
```

**Linux:**
```bash
curl -L -o pkl https://github.com/apple/pkl/releases/download/0.28.2/pkl-linux-amd64
chmod +x pkl
./pkl --version
```

## 📄 Example Files

### 1. `BaseActiveInferenceModel.pkl`
**Foundation template for all Active Inference models.**

- Defines core Active Inference components (A, B, C, D matrices)
- GNN naming convention validation (`s_f0`, `o_m0`, `u_c0`)
- Mathematical constraints for stochasticity
- Type-safe model structure

```bash
# Evaluate as YAML
pkl eval -f yaml BaseActiveInferenceModel.pkl

# Evaluate as JSON
pkl eval -f json BaseActiveInferenceModel.pkl
```

### 2. `VisualForagingModel.pkl`
**Concrete Active Inference model demonstrating template inheritance.**

- Extends `BaseActiveInferenceModel.pkl`
- Implements 2x2 spatial grid foraging behavior
- Includes complete A, B, C, D matrices
- Demonstrates model specialization

```bash
# Evaluate the complete model
pkl eval -f yaml VisualForagingModel.pkl

# Export to JSON for PyMDP integration
pkl eval -f json VisualForagingModel.pkl > foraging_model.json
```

### 3. `GNNPipelineConfig.pkl`
**Advanced pipeline configuration demonstrating Pkl's dynamic features.**

- Late binding configuration
- Platform-specific settings
- Resource constraint calculation
- Multi-backend rendering configuration

```bash
# View pipeline configuration
pkl eval -f yaml GNNPipelineConfig.pkl

# Check platform-specific settings
pkl eval -f json GNNPipelineConfig.pkl | jq '.steps.visualization.enabled'
```

### 4. `MultiFormatExportConfig.pkl`
**Multi-format export configuration.**

- Demonstrates Pkl's rendering capabilities
- Format-specific optimizations
- Consistent metadata across formats

```bash
# Evaluate export configuration
pkl eval -f yaml MultiFormatExportConfig.pkl
```

## 🎯 Key Pkl Features Demonstrated

### **Type Safety & Validation**
```pkl
class StateVariable {
  name: String(!isEmpty && matches(Regex("(s_f|o_m|u_c)\\d+")))
  dimensions: List<Int>(length.isBetween(1, 4) && every { it > 0 })
  variableType: "categorical" | "continuous" | "binary"
}
```

### **Mathematical Constraints**
```pkl
values: List<List<Float>>(
  every { row -> 
    row.every { it >= 0.0 && it <= 1.0 } &&
    math.abs(row.sum() - 1.0) < 1e-10 // Stochasticity check
  }
)
```

### **Template Inheritance**
```pkl
/// Visual Foraging Model extends base template
amends "BaseActiveInferenceModel.pkl"

modelName = "VisualForagingAgent"
```

### **Late Binding & Dynamic Configuration**
```pkl
steps: Mapping<String, StepConfig> = new {
  ["render"] = new StepConfig {
    enabled = exportTargets.any { it.format != "json" }  // Late binding
    timeout = (renderingEngines.length * 30).s  // Dynamic timeout
  }
}
```

## 🔬 Active Inference Integration

These examples show how Pkl could enhance GNN's Active Inference model specification:

1. **Type-Safe A, B, C, D Matrices**: Compile-time validation of matrix dimensions and stochasticity
2. **GNN Naming Conventions**: Automatic validation of variable naming (`s_f0`, `o_m0`, `u_c0`)
3. **Multi-Backend Support**: Single configuration generating PyMDP, RxInfer.jl, and JAX code
4. **Mathematical Validation**: Built-in constraints ensuring proper probability distributions

## 🚀 Usage Examples

### Generate PyMDP Code
```bash
# Export model for PyMDP integration
pkl eval -f json VisualForagingModel.pkl > pymdp_model.json
```

### Validate Model Structure
```bash
# Pkl automatically validates structure and constraints
pkl eval BaseActiveInferenceModel.pkl
```

### Multi-Format Export
```bash
# Generate multiple formats from single source
pkl eval -f yaml VisualForagingModel.pkl > model.yaml
pkl eval -f json VisualForagingModel.pkl > model.json
pkl eval -f xml VisualForagingModel.pkl > model.xml
```

## 🌟 Benefits for GNN

1. **Enhanced Type Safety**: Catch Active Inference model errors at configuration time
2. **Template Reuse**: Share common patterns across Active Inference models
3. **Multi-Format Generation**: Single source producing JSON, YAML, XML, GraphML
4. **Mathematical Validation**: Automatic checking of stochasticity and dimensionality
5. **IDE Support**: Rich tooling with auto-complete and error detection
6. **Reproducible Science**: Immutable, deterministic configurations

## 📚 Further Reading

- [Pkl Language Reference](https://pkl-lang.org/main/current/language-reference/)
- [GNN Project Documentation](../../README.md)
- [Active Inference Ontology](../ontology/)
- [Pkl vs Other Config Languages](https://pkl-lang.org/main/current/introduction/comparison.html)

## 🤝 Contributing

To add new Pkl examples:

1. Follow GNN naming conventions (`s_f0`, `o_m0`, `u_c0`)
2. Include mathematical validation constraints
3. Add comprehensive documentation
4. Test with `pkl eval` before committing

See the main [pkl_gnn_demo.py](../pkl_gnn_demo.py) script for automated generation of these examples. 