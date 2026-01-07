# SPM Documentation Agent

> **📋 Document Metadata**  
> **Type**: Neuroscientific Integration Agent | **Audience**: Neuroscientists, AI Researchers | **Complexity**: Advanced  
> **Cross-References**: [README.md](README.md) | [SPM GNN Guide](spm_gnn.md) | [Cognitive Phenomena](../cognitive_phenomena/README.md) | [Main Documentation](../README.md)

## Overview

This directory contains comprehensive documentation, resources, and implementation guides for integrating **SPM** (Statistical Parametric Mapping) with GNN (Generalized Notation Notation). SPM provides established statistical frameworks for neuroimaging analysis, enabling translation of neuroimaging insights into computational cognitive architectures.

**Status**: ✅ Production Ready  
**Version**: 1.0

## Purpose

SPM integration enables:

- **Brain-Inspired AI**: Translation of neuroimaging insights to computational models
- **Model Calibration**: SPM results inform GNN state space design
- **Connectivity Mapping**: DCM connectivity matrices guide GNN transition models
- **Temporal Dynamics**: SPM temporal dynamics constrain GNN time horizons
- **Population-Level Inference**: Group statistics for model validation

## Contents

**Files**:        3 | **Subdirectories**:        1

## Quick Navigation

- **README.md**: [Directory overview](README.md)
- **GNN Documentation**: [gnn/AGENTS.md](../gnn/AGENTS.md)
- **Main Documentation**: [doc/README.md](../README.md)
- **Pipeline Reference**: [src/AGENTS.md](../../src/AGENTS.md)

## Documentation Structure

This module is organized as follows:

- **Overview**: High-level description and purpose
- **Contents**: Files and subdirectories
- **Integration**: Connection to the broader pipeline
- **Usage**: How to work with this subsystem

## Integration with Pipeline

This documentation is integrated with the 24-step GNN processing pipeline:

### Core Processing (Steps 0-9)
- **Step 3 (GNN)**: SPM-informed GNN model specification
- **Step 5 (Type Checker)**: Neuroimaging data validation
- **Step 6 (Validation)**: SPM-based model validation

### Simulation (Steps 10-16)
- **Step 11 (Render)**: SPM-calibrated code generation
- **Step 12 (Execute)**: Cross-validation with neuroimaging data
- **Step 16 (Analysis)**: SPM statistical analysis integration

### Integration (Steps 17-23)
- **Step 23 (Report)**: Neuroscientific validation and analysis results

See [src/AGENTS.md](../../src/AGENTS.md) for complete pipeline documentation.

## Function Signatures and API

### Neuroimaging Analysis Functions

```python
def calibrate_gnn_from_spm(gnn_model: GNNModel, spm_results: SPMResults) -> GNNModel:
    """
    Calibrate GNN model using SPM neuroimaging results.
    
    Parameters:
        gnn_model: Parsed GNN model structure
        spm_results: SPM statistical analysis results
    
    Returns:
        Calibrated GNNModel with SPM-informed parameters
    """

def extract_connectivity_matrix(spm_results: SPMResults) -> ConnectivityMatrix:
    """
    Extract connectivity matrix from SPM DCM results.
    
    Parameters:
        spm_results: SPM analysis results
    
    Returns:
        ConnectivityMatrix for GNN transition model
    """
```

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing
- **Functionality**: Describes actual capabilities
- **Completeness**: Comprehensive coverage
- **Consistency**: Uniform structure and style

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/quickstart_tutorial.md)**: Getting started guide
- **[Cognitive Phenomena](../cognitive_phenomena/README.md)**: Cognitive modeling applications

### Neuroscientific Resources
- **[Cognitive Phenomena](../cognitive_phenomena/README.md)**: Cognitive modeling applications
- **[Neuroscience](../CROSS_REFERENCE_INDEX.md#neuroscience)**: Neuroscientific methods
- **[Research Tools](../research/README.md)**: Research workflow tools

### Pipeline Architecture
- **[Pipeline Documentation](../pipeline/README.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/README.md)**: Pipeline overview

## See Also

- **[SPM Cross-Reference](../CROSS_REFERENCE_INDEX.md#spm)**: Cross-reference index entry
- **[Cognitive Phenomena](../cognitive_phenomena/README.md)**: Cognitive modeling applications
- **[Neuroscience](../CROSS_REFERENCE_INDEX.md#neuroscience)**: Neuroscientific methods
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: ✅ Production Ready  
**Compliance**: Professional documentation standards  
**Maintenance**: Regular updates with new SPM features and integration capabilities
