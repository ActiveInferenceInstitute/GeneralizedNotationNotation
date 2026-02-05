# GUI Oxdraw Integration for GNN

> **📋 Document Metadata**  
> **Type**: Interface Integration Guide | **Audience**: Users, Developers | **Complexity**: Intermediate  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [Oxdraw GNN Guide](gnn_oxdraw.md) | [Oxdraw Overview](oxdraw.md) | [GUI Documentation](../../src/gui/README.md) | [Main Documentation](../README.md)

## Overview

This directory contains documentation, resources, and implementation guides for integrating **Oxdraw** with GNN (Generalized Notation Notation). Oxdraw serves as a visual interface for the GNN pipeline, enabling drag-and-drop creation of Active Inference graphical models with bidirectional text-visual synchronization.

**Status**: ✅ Production Ready  
**Version**: 1.0

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Directory overview (this file)
- **[AGENTS.md](AGENTS.md)**: Technical documentation and agent scaffolding
- **[gnn_oxdraw.md](gnn_oxdraw.md)**: Complete Oxdraw-GNN integration guide
- **[oxdraw.md](oxdraw.md)**: Oxdraw framework overview
- **[VERIFICATION.md](VERIFICATION.md)**: Verification and testing documentation

### Main Documentation
- **[doc/README.md](../README.md)**: Main documentation hub
- **[CROSS_REFERENCE_INDEX.md](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference index
- **[learning_paths.md](../learning_paths.md)**: Learning pathways

### Related Directories
- **[GUI Documentation](../../src/gui/README.md)**: Interactive GUI interfaces
- **[Visualization](../visualization/README.md)**: Graph and matrix visualization
- **[Advanced Visualization](../advanced_visualization/README.md)**: Advanced visualization tools
- **[D2 Integration](../d2/gnn_d2.md)**: Scriptable diagramming

### Pipeline Integration
- **[Pipeline Documentation](../gnn/gnn_tools.md)**: Complete pipeline guide
- **[src/AGENTS.md](../../src/AGENTS.md)**: Implementation details

## Contents

**Files**: 4 | **Subdirectories**: 0

### Core Files

- **`gnn_oxdraw.md`**: Complete Oxdraw-GNN integration guide
  - Visual diagram-as-code interface
  - Bidirectional text-visual synchronization
  - Mermaid diagram integration
  - Active Inference ontology preservation

- **`oxdraw.md`**: Oxdraw framework overview
  - Hybrid Mermaid-based architecture
  - Visual model construction
  - Drag-and-drop interface

- **`VERIFICATION.md`**: Verification and testing documentation
  - Integration testing
  - Validation procedures

- **`AGENTS.md`**: Technical documentation and agent scaffolding
  - Complete documentation structure
  - Integration with pipeline
  - Cross-references and navigation

- **`README.md`**: Directory overview (this file)

## Oxdraw Overview

Oxdraw provides:

### Visual Model Construction
- **Drag-and-Drop Interface**: Intuitive node-and-edge manipulation
- **Bidirectional Synchronization**: Automatic translation between Mermaid diagrams and GNN plaintext
- **Ontology Preservation**: Active Inference ontology assertions through comment-based metadata
- **Full Pipeline Integration**: Integration with GNN validation, type-checking, and simulation rendering

### Key Features
- **Mermaid Integration**: Hybrid Mermaid-based architecture
- **Visual Editing**: Visual construction of POMDP architectures
- **Text Synchronization**: Automatic GNN file generation from visual edits
- **Validation**: Full integration with GNN validation workflows

## Integration with Pipeline

This documentation is integrated with the 25-step GNN processing pipeline:

1. **Core Processing** (Steps 0-9): GNN parsing, validation, export
   - Oxdraw visual edits translated to GNN files
   - GNN files parsed and displayed visually

2. **Simulation** (Steps 10-16): Model execution and analysis
   - Visual model execution
   - Results visualization

3. **Integration** (Steps 17-24): System coordination and output
   - Oxdraw results integrated into comprehensive outputs
   - Visual documentation generation

See [src/AGENTS.md](../../src/AGENTS.md) for complete pipeline documentation.

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/quickstart_tutorial.md)**: Getting started guide
- **[GUI Documentation](../../src/gui/README.md)**: Interactive GUI interfaces

### Visualization Resources
- **[Visualization Guide](../visualization/README.md)**: Graph and matrix visualization
- **[Advanced Visualization](../advanced_visualization/README.md)**: Advanced visualization tools
- **[D2 Integration](../d2/gnn_d2.md)**: Scriptable diagramming

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/README.md)**: Pipeline overview

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with UI/UX foundations
- **Functionality**: Describes actual Oxdraw integration capabilities
- **Completeness**: Comprehensive coverage of visual model construction
- **Consistency**: Uniform structure and style with GNN documentation ecosystem

## See Also

- **[GUI Oxdraw Cross-Reference](../CROSS_REFERENCE_INDEX.md#gui_oxdraw)**: Cross-reference index entry
- **[GUI Documentation](../../src/gui/README.md)**: Interactive GUI interfaces
- **[Visualization Guide](../visualization/README.md)**: Visualization tools
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: ✅ Production Ready  
**Compliance**: Professional documentation standards  
**Maintenance**: Regular updates with new Oxdraw features and integration capabilities
