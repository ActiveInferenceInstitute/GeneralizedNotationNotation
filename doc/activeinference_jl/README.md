# ActiveInference.jl Integration for GNN

> **📋 Document Metadata**  
> **Type**: Framework Integration Guide | **Audience**: Researchers, Developers | **Complexity**: Intermediate-Advanced  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [ActiveInference.jl Guide](activeinference-jl.md) | [Framework Integration](../gnn/framework_integration_guide.md) | [Main Documentation](../README.md)

## Overview

This directory contains documentation, resources, and implementation guides for integrating GNN (Generalized Notation Notation) models with **ActiveInference.jl**, a Julia-based implementation of Active Inference algorithms. ActiveInference.jl provides high-performance inference through Julia's Just-In-Time (JIT) compilation.

**Status**: ✅ Production Ready  
**Version**: 1.0

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Directory overview (this file)
- **[AGENTS.md](AGENTS.md)**: Technical documentation and agent scaffolding
- **[activeinference-jl.md](activeinference-jl.md)**: Complete ActiveInference.jl integration guide
- **[activeinference-jl_source_code.md](activeinference-jl_source_code.md)**: Source code analysis

### Main Documentation
- **[doc/README.md](../README.md)**: Main documentation hub
- **[CROSS_REFERENCE_INDEX.md](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference index
- **[learning_paths.md](../learning_paths.md)**: Learning pathways

### Related Directories
- **[Framework Integration](../gnn/framework_integration_guide.md)**: Framework integration overview
- **[RxInfer Integration](../rxinfer/gnn_rxinfer.md)**: Julia Bayesian inference framework
- **[Execution Guide](../execution/README.md)**: Framework execution strategies
- **[Julia Frameworks](../execution/FRAMEWORK_AVAILABILITY.md)**: Julia framework availability

### Pipeline Integration
- **[Pipeline Documentation](../pipeline/README.md)**: Complete pipeline guide
- **[src/AGENTS.md](../../src/AGENTS.md)**: Implementation details

## Contents

**Files**: 4+ | **Subdirectories**: 3

### Core Files

- **`activeinference-jl.md`**: Complete ActiveInference.jl integration guide
  - ActiveInference.jl framework overview
  - GNN to ActiveInference.jl translation
  - Code generation patterns
  - Example models and usage

- **`activeinference-jl_source_code.md`**: Source code analysis
  - Detailed source code documentation
  - Implementation patterns
  - API reference

- **`AGENTS.md`**: Technical documentation and agent scaffolding
  - Complete documentation structure
  - Integration with pipeline
  - Cross-references and navigation

- **`README.md`**: Directory overview (this file)

### Subdirectories

- **`actinf_jl_src/`**: ActiveInference.jl source code files
- **`test_output/`**: Test output and validation results

## High-Performance Inference with Julia

ActiveInference.jl leverages Julia's capabilities to provide:

### Performance Benefits
- **Zero-Cost Abstractions**: Model agents with high-level GNN syntax that compiles down to efficient machine code
- **Fast Belief Updating**: Typical belief update loops are 10-50x faster than pure Python implementations
- **Type Stability**: GNN's type-checking step ensures generated Julia code is fully type-stable, maximizing LLVM optimization
- **Real-Time Applications**: Ideal for real-time robotic applications and high-frequency simulations

### Key Features
- **Complete Active Inference**: Full Active Inference agent implementation
- **Hierarchical Temporal Models**: Support for multi-level temporal dynamics
- **Comprehensive Belief Updating**: Variational message passing and belief propagation
- **Julia Ecosystem Integration**: Seamless integration with Julia scientific computing stack

## Integration with Pipeline

This documentation is integrated with the 24-step GNN processing pipeline:

1. **Core Processing** (Steps 0-9): GNN parsing, validation, export
   - GNN models parsed and validated
   - ActiveInference.jl code generation (Step 11: Render)

2. **Simulation** (Steps 10-16): Model execution and analysis
   - ActiveInference.jl execution (Step 12: Execute)
   - High-performance simulation results

3. **Integration** (Steps 17-23): System coordination and output
   - ActiveInference.jl results integrated into comprehensive outputs
   - Performance metrics and analysis

See [src/AGENTS.md](../../src/AGENTS.md) for complete pipeline documentation.

## Usage Examples

### Basic ActiveInference.jl Model

GNN models are translated to ActiveInference.jl implementations:

```julia
using ActiveInference

# GNN model translated to ActiveInference.jl
agent = ActiveInferenceAgent(
    A = A_matrix,  # Observation likelihood
    B = B_matrix,  # State transition
    C = C_vector,  # Preferences
    D = D_vector   # Prior beliefs
)

# Run inference
beliefs = infer_states(agent, observations)
actions = sample_actions(agent, beliefs)
```

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/quickstart_tutorial.md)**: Getting started guide
- **[Framework Integration](../gnn/framework_integration_guide.md)**: Framework integration overview

### Framework Integration
- **[RxInfer Integration](../rxinfer/gnn_rxinfer.md)**: Julia Bayesian inference framework
- **[PyMDP Integration](../pymdp/gnn_pymdp.md)**: Python Active Inference framework
- **[Execution Guide](../execution/README.md)**: Framework execution strategies

### Pipeline Architecture
- **[Pipeline Documentation](../pipeline/README.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/README.md)**: Pipeline overview

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with Julia and Active Inference foundations
- **Functionality**: Describes actual ActiveInference.jl integration capabilities
- **Completeness**: Comprehensive coverage of ActiveInference.jl integration
- **Consistency**: Uniform structure and style with GNN documentation ecosystem

## See Also

- **[ActiveInference.jl Cross-Reference](../CROSS_REFERENCE_INDEX.md#activeinference_jl)**: Cross-reference index entry
- **[Framework Integration](../gnn/framework_integration_guide.md)**: Framework integration overview
- **[Julia Frameworks](../execution/FRAMEWORK_AVAILABILITY.md)**: Julia framework availability
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: ✅ Production Ready  
**Compliance**: Professional documentation standards  
**Maintenance**: Regular updates with new ActiveInference.jl features and integration capabilities
