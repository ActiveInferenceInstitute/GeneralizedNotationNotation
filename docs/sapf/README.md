# SAPF Integration for GNN

> **📋 Document Metadata**  
> **Type**: Audio Framework Integration Guide | **Audience**: Researchers, Audio Engineers | **Complexity**: Advanced  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [SAPF GNN Guide](sapf_gnn.md) | [SAPF Overview](sapf.md) | [Audio Processing](../audio/README.md) | [Pedalboard Integration](../pedalboard/pedalboard_gnn.md) | [Main Documentation](../README.md)

## Overview

This directory contains documentation, resources, and implementation guides for integrating **SAPF** (Sound As Pure Form) with GNN (Generalized Notation Notation). SAPF provides a concatenative programming paradigm for auditory representation and real-time sonification of Active Inference generative models.

**Role**: SAPF is an **audio sonification capability** (Step 15), not a render or execution framework. It is not an entry in `src/gnn/render/framework_registry.py`, generates no simulation code, and has no Step 12 executor. In the live tree, `src/gnn/sapf/` provides the SAPF module and the Step 15 audio script (`src/gnn/15_audio.py`) accepts `--audio-backend` values including `sapf`; `src/gnn/STEP_INDEX.md` lists Step 15 as audio sonification (SAPF) with soundfile and pedalboard as dependencies.

**Status**: ✅ Production Ready  
**Version**: 1.0

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Directory overview (this file)
- **[AGENTS.md](AGENTS.md)**: Technical documentation and agent scaffolding
- **[sapf_gnn.md](sapf_gnn.md)**: Complete SAPF-GNN integration guide
- **[sapf.md](sapf.md)**: SAPF framework overview

### Main Documentation
- **[docs/README.md](../README.md)**: Main documentation hub
- **[CROSS_REFERENCE_INDEX.md](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference index
- **[learning_paths.md](../learning_paths.md)**: Learning pathways

### Related Directories
- **[Audio Processing](../audio/README.md)**: Audio generation and sonification
- **[Pedalboard Integration](../pedalboard/pedalboard_gnn.md)**: Audio processing library
- **[Sonification](../CROSS_REFERENCE_INDEX.md#audio-and-sonification)**: Audio sonification tools

### Pipeline Integration
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)**: Implementation details

## Contents

**Files**: 5 | **Subdirectories**: 0

### Core Files

- **`sapf_gnn.md`**: Complete SAPF-GNN integration guide
  - Auditory representation of Active Inference models
  - Real-time sonification
  - Concatenative programming paradigm
  - GNN-to-audio mapping specifications

- **`sapf.md`**: SAPF framework overview
  - Sound As Pure Form philosophy
  - Concatenative programming
  - Audio synthesis capabilities

- **`AGENTS.md`**: Technical documentation and agent scaffolding
  - Complete documentation structure
  - Integration with pipeline
  - Cross-references and navigation

- **`README.md`**: Directory overview (this file)

## SAPF Overview

SAPF provides:

### Concatenative Programming
- **Lazy Evaluation**: Infinite sequences and automatic mapping
- **Multi-Dimensional Audio**: Multi-dimensional audio synthesis
- **Real-Time Sonification**: Real-time model sonification capabilities
- **Automatic Mapping**: Automatic mapping of model dynamics to audio

### Key Features
- **Sound As Pure Form**: Philosophy of auditory model representation
- **Infinite Sequences**: Lazy evaluation for continuous audio generation
- **Automatic Mapping**: Automatic mapping of GNN components to audio
- **Real-Time Processing**: Real-time sonification of model dynamics

## Integration with GNN

SAPF integration enables:

- **Auditory Model Representation**: Understanding models through sound
- **Real-Time Sonification**: Real-time sonification of Active Inference dynamics
- **Multi-Dimensional Audio**: Multi-dimensional audio synthesis for complex models
- **Debugging Through Sound**: Audio-based model debugging and analysis

## Integration with Pipeline

This documentation is integrated with the 25-step GNN processing pipeline:

1. **Core Processing** (Steps 0-9): GNN parsing, validation, export
   - No audio processing occurs in these steps; parsed models are simply prepared for downstream sonification

2. **Simulation** (Steps 10-16): Model execution and analysis
   - Real-time sonification (Step 15: Audio)
   - Audio-based model analysis

3. **Integration** (Steps 17-24): System coordination and output
   - SAPF results integrated into comprehensive outputs
   - Audio visualization and analysis

See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/tutorials/quickstart_tutorial.md)**: Getting started guide
- **[Audio Processing](../audio/README.md)**: Audio generation and sonification

### Audio Resources
- **[Pedalboard Integration](../pedalboard/pedalboard_gnn.md)**: Audio processing library
- **[Audio Processing](../audio/README.md)**: Audio generation tools
- **[Sonification](../CROSS_REFERENCE_INDEX.md#audio-and-sonification)**: Audio sonification overview

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/gnn/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/gnn/README.md)**: Pipeline overview

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with audio processing foundations
- **Functionality**: Describes actual SAPF integration capabilities
- **Completeness**: Comprehensive coverage of audio sonification integration
- **Consistency**: Uniform structure and style with GNN documentation ecosystem

## See Also

- **[SAPF Cross-Reference](../CROSS_REFERENCE_INDEX.md#sapf)**: Cross-reference index entry
- **[Audio Processing](../audio/README.md)**: Audio generation and sonification
- **[Pedalboard Integration](../pedalboard/pedalboard_gnn.md)**: Audio processing library
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: ✅ Production Ready  
**Compliance**: Professional documentation standards  
**Maintenance**: Regular updates with new SAPF features and integration capabilities
