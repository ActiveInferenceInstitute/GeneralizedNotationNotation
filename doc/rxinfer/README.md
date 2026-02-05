# RxInfer.jl Integration for GNN

> **📋 Document Metadata**  
> **Type**: Framework Integration Guide | **Audience**: Researchers, Developers | **Complexity**: Intermediate-Advanced  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [GNN RxInfer Guide](gnn_rxinfer.md) | [Framework Integration](../gnn/framework_integration_guide.md) | [Main Documentation](../README.md)

## Overview

This directory contains documentation, scripts, and resources for integrating GNN (Generalized Notation Notation) models with **RxInfer.jl**, a Julia-based reactive Bayesian inference framework. RxInfer.jl provides efficient message-passing inference for probabilistic models, making it ideal for Active Inference simulations.

**Status**: ✅ Production Ready  
**Version**: 1.0

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Directory overview (this file)
- **[AGENTS.md](AGENTS.md)**: Technical documentation and agent scaffolding
- **[gnn_rxinfer.md](gnn_rxinfer.md)**: Complete RxInfer.jl integration guide
- **[Multiagent_GNN_RxInfer.jl](Multiagent_GNN_RxInfer.jl)**: Validation script

### Main Documentation
- **[doc/README.md](../README.md)**: Main documentation hub
- **[CROSS_REFERENCE_INDEX.md](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference index
- **[learning_paths.md](../learning_paths.md)**: Learning pathways

### Related Directories
- **[Framework Integration](../gnn/framework_integration_guide.md)**: Framework integration overview
- **[PyMDP Integration](../pymdp/gnn_pymdp.md)**: Python Active Inference framework
- **[Execution Guide](../execution/README.md)**: Framework execution strategies
- **[Multi-agent Systems](../gnn/gnn_multiagent.md)**: Multi-agent modeling

### Pipeline Integration
- **[Pipeline Documentation](../gnn/gnn_tools.md)**: Complete pipeline guide
- **[src/AGENTS.md](../../src/AGENTS.md)**: Implementation details

## Contents

**Files**: 12+ | **Subdirectories**: 1

### Core Files

- **`gnn_rxinfer.md`**: Complete RxInfer.jl integration guide
  - RxInfer.jl framework overview
  - GNN to RxInfer.jl translation
  - Code generation patterns
  - Example models and usage

- **`Multiagent_GNN_RxInfer.jl`**: Validation script
  - Validates GNN to RxInfer.jl translation
  - Two-stage validation process
  - Configuration file generation testing

- **`engineering_rxinfer_gnn.md`**: Engineering guide
  - Technical implementation details
  - Best practices and patterns

- **`AGENTS.md`**: Technical documentation and agent scaffolding
  - Complete documentation structure
  - Integration with pipeline
  - Cross-references and navigation

- **`README.md`**: Directory overview (this file)

### Subdirectories

- **`multiagent_trajectory_planning/`**: Multi-agent trajectory planning examples
  - Complete RxInfer.jl implementations
  - Configuration examples
  - Results and analysis

## RxInfer.jl Integration

### Framework Overview

**RxInfer.jl** is a reactive Bayesian inference framework for Julia that provides:

- **Reactive Probabilistic Programming**: Dynamic model construction and inference
- **Efficient Message Passing**: Optimized inference algorithms
- **Factor Graph Models**: Natural representation of Active Inference models
- **Streaming Inference**: Real-time belief updating
- **Multi-agent Support**: Coordinated multi-agent systems

### GNN to RxInfer.jl Translation

The GNN pipeline translates GNN models to RxInfer.jl through:

1. **Model Parsing**: GNN syntax parsed into structured representation
2. **Factor Graph Construction**: Active Inference components mapped to factor graph
3. **Code Generation**: Julia code generation with RxInfer.jl API
4. **Configuration Generation**: TOML configuration files for model parameters
5. **Validation**: Automated validation of generated code

### Validation Process

The `Multiagent_GNN_RxInfer.jl` script validates the translation pipeline:

#### Stage 1: Baseline Simulation
- Locates standard "Multi-agent Trajectory Planning" example
- Runs with original hand-written `config.toml`
- Establishes baseline for successful execution

#### Stage 2: GNN-Configured Simulation
- Creates new validation directory
- Copies Julia script files from original example
- Replaces `config.toml` with GNN-generated configuration
- Executes simulation with GNN-derived configuration
- Compares results with baseline

### Validation Success Criteria

Successful validation demonstrates:
- **Syntactic Correctness**: GNN parser produces valid TOML configuration
- **Parameter Translation**: GNN parameters correctly translated to RxInfer.jl values
- **End-to-End Functionality**: Complete pipeline from GNN model to RxInfer.jl simulation
- **Result Equivalence**: GNN-configured results match baseline expectations

## Integration with Pipeline

This documentation is integrated with the 25-step GNN processing pipeline:

1. **Core Processing** (Steps 0-9): GNN parsing, validation, export
   - GNN models parsed and validated
   - RxInfer.jl code generation (Step 11: Render)

2. **Simulation** (Steps 10-16): Model execution and analysis
   - RxInfer.jl execution (Step 12: Execute)
   - Results processing and analysis

3. **Integration** (Steps 17-24): System coordination and output
   - RxInfer.jl results integrated into comprehensive outputs
   - Multi-agent coordination and analysis

See [src/AGENTS.md](../../src/AGENTS.md) for complete pipeline documentation.

## Usage Examples

### Running Validation

```bash
# Ensure Julia environment with required packages
julia doc/rxinfer/Multiagent_GNN_RxInfer.jl
```

### Basic RxInfer.jl Model

GNN models are translated to RxInfer.jl factor graphs:

```julia
using RxInfer

# GNN model translated to RxInfer.jl
@model function gnn_model(observations, actions)
    # Hidden state beliefs
    s_f0 ~ Categorical(prior)
    
    # Observations
    o_m0 ~ Categorical(A * s_f0)
    
    # State transitions
    s_f0_next ~ Categorical(B[s_f0, actions])
    
    return s_f0, o_m0, s_f0_next
end
```

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/quickstart_tutorial.md)**: Getting started guide
- **[Framework Integration](../gnn/framework_integration_guide.md)**: Framework integration overview

### Framework Integration
- **[PyMDP Integration](../pymdp/gnn_pymdp.md)**: Python Active Inference framework
- **[DisCoPy Integration](../discopy/gnn_discopy.md)**: Category theory framework
- **[Execution Guide](../execution/README.md)**: Framework execution strategies

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/README.md)**: Pipeline overview

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with Julia and RxInfer.jl foundations
- **Functionality**: Describes actual RxInfer.jl integration capabilities
- **Completeness**: Comprehensive coverage of RxInfer.jl integration
- **Consistency**: Uniform structure and style with GNN documentation ecosystem

## See Also

- **[RxInfer Integration](../CROSS_REFERENCE_INDEX.md#rxinferjl)**: Cross-reference index entry
- **[Framework Integration](../gnn/framework_integration_guide.md)**: Framework integration overview
- **[Multi-agent Systems](../gnn/gnn_multiagent.md)**: Multi-agent modeling
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: ✅ Production Ready  
**Compliance**: Professional documentation standards  
**Maintenance**: Regular updates with new RxInfer.jl features and integration capabilities 