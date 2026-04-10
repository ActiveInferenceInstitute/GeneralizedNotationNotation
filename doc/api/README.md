# GNN API Documentation

> **📋 Document Metadata**  
> **Type**: API Reference | **Audience**: Developers, Integrators | **Complexity**: Intermediate-Advanced  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [Comprehensive API Reference](comprehensive_api_reference.md) | [Pipeline Architecture](../gnn/operations/gnn_tools.md) | [Main Documentation](../README.md)

## Overview

This directory contains API-oriented documentation for the GNN (Generalized Notation Notation) codebase. **Authoritative Python exports** for the `gnn` package are in [`src/gnn/__init__.py`](../../src/gnn/__init__.py); a machine-readable index is [`api_index.json`](api_index.json). [`comprehensive_api_reference.md`](comprehensive_api_reference.md) labels legacy illustrative sections—verify names in `src/` before importing.

**Status**: ✅ Production Ready  
**Version**: 1.0

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Directory overview (this file)
- **[AGENTS.md](AGENTS.md)**: Technical documentation and agent scaffolding
- **[comprehensive_api_reference.md](comprehensive_api_reference.md)**: Complete API reference documentation

### Main Documentation
- **[doc/README.md](../README.md)**: Main documentation hub
- **[CROSS_REFERENCE_INDEX.md](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference index
- **[learning_paths.md](../learning_paths.md)**: Learning pathways

### Related Directories
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Pipeline execution and orchestration
- **[Development Guide](../development/README.md)**: Development workflows and contribution guidelines
- **[MCP Integration](../mcp/README.md)**: Model Context Protocol APIs
- **[Framework Integration](../gnn/integration/framework_integration_guide.md)**: Framework-specific APIs

### Pipeline Integration
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[src/AGENTS.md](../../src/AGENTS.md)**: Implementation details

## Contents

**Files**: 4 (`README.md`, `AGENTS.md`, `comprehensive_api_reference.md`, `api_index.json`) | **Subdirectories**: 0

### Core Files

- **`comprehensive_api_reference.md`**: Complete API reference documentation
  - All programmatic interfaces for GNN integration
  - Core Parsing API: GNN file parsing and validation
  - Pipeline API: Pipeline execution and orchestration
  - Framework Integration API: PyMDP, RxInfer, DisCoPy interfaces
  - Visualization API: Programmatic visualization generation
  - LLM Integration API: AI-enhanced model analysis
  - MCP API: Model Context Protocol integration
  - Performance API: Monitoring and optimization interfaces

- **`api_index.json`**: Machine-readable API index
  - Generated map of modules, functions, and classes under `src/`
  - Created by `src/generate_api_index.py`
  - AST-derived with file paths, module names, function signatures, class bases, and docstrings
  - Excludes tests and output directories

- **`AGENTS.md`**: Technical documentation and agent scaffolding
  - Complete documentation structure
  - Integration with pipeline
  - Cross-references and navigation

- **`README.md`**: Directory overview (this file)

## API Categories

### Core Parsing API (package `gnn`)
- **GNNParsingSystem**, **GNNFormat**: Registry-backed multi-format I/O ([`src/gnn/parsers/system.py`](../../src/gnn/parsers/system.py))
- **discover_gnn_files**, **parse_gnn_file**, **process_gnn_directory**, **process_gnn_multi_format**: Discovery and processing ([`src/gnn/__init__.py`](../../src/gnn/__init__.py))
- **validate_gnn**, **ValidationLevel**: Validation entry points
- See [`src/gnn/SPEC.md`](../../src/gnn/SPEC.md) for format counts

### Pipeline API
- **Pipeline**: Pipeline execution and orchestration
- **PipelineConfig**: Configuration management
- **Step Execution**: Individual step processing
- **Result Aggregation**: Result collection and reporting

### Framework Integration API
- **PyMDP Integration**: Python Active Inference framework interfaces
- **RxInfer Integration**: Julia Bayesian inference interfaces
- **DisCoPy Integration**: Category theory and quantum computing interfaces
- **JAX Integration**: High-performance numerical computing interfaces

### Visualization API
- **Visualizer**: Programmatic visualization generation
- **Graph Generation**: Network diagram creation
- **Matrix Visualization**: Heatmap and matrix displays
- **Interactive Diagrams**: Dynamic visualization interfaces

### LLM Integration API
- **LLMProcessor**: AI-enhanced model analysis
- **Provider Interfaces**: Multi-provider LLM support
- **Prompt Generation**: Automated prompt creation
- **Response Processing**: LLM output interpretation

### MCP API
- **MCP Tools**: Model Context Protocol tool registration
- **Tool Discovery**: Automatic tool detection
- **Protocol Compliance**: Standard interface implementation

### Performance API
- **Performance Monitoring**: Operation timing and metrics
- **Resource Tracking**: Memory and CPU usage
- **Optimization Interfaces**: Performance tuning capabilities

## Generating the API Index

The `api_index.json` file is automatically generated from the codebase:

```bash
# Generate API index
python src/generate_api_index.py
```

### Index Generation Details

- **Source**: All Python files under `src/` directory
- **Exclusions**: Tests and output directories are excluded
- **Method**: AST-based parsing for accurate extraction
- **Content**: File paths, module names, function signatures, class bases, and docstrings

## Integration with Pipeline

This documentation is integrated with the 25-step GNN processing pipeline:

1. **Core Processing** (Steps 0-9): GNN parsing, validation, export
   - Core Parsing API used throughout
   - Validation interfaces for type checking

2. **Simulation** (Steps 10-16): Model execution and analysis
   - Framework Integration APIs for code generation
   - Execution interfaces for running simulations

3. **Integration** (Steps 17-24): System coordination and output
   - MCP API for tool integration
   - Performance API for monitoring
   - Visualization API for output generation

See [src/AGENTS.md](../../src/AGENTS.md) for complete pipeline documentation.

## Usage Examples

### Basic API Usage

```python
import logging
from pathlib import Path
from gnn import discover_gnn_files, parse_gnn_file, GNNParsingSystem, GNNFormat, process_gnn_multi_format

paths = discover_gnn_files(Path("input/gnn_files"))
info = parse_gnn_file(paths[0])
system = GNNParsingSystem()
result = system.parse_file(paths[0], format_hint=GNNFormat.MARKDOWN)

logger = logging.getLogger(__name__)
process_gnn_multi_format(Path("input/gnn_files"), Path("output"), logger)
```

Full pipeline runs use `python src/main.py` (see [Pipeline docs](../gnn/operations/gnn_tools.md)).

### Framework Integration

```python
from gnn.render import PyMDPRenderer, RxInferRenderer

# Generate PyMDP code
pymdp_renderer = PyMDPRenderer()
pymdp_code = pymdp_renderer.render(model)

# Generate RxInfer code
rxinfer_renderer = RxInferRenderer()
rxinfer_code = rxinfer_renderer.render(model)
```

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/tutorials/quickstart_tutorial.md)**: Getting started guide
- **[GNN Examples](../gnn/tutorials/gnn_examples_doc.md)**: Example models

### Development Resources
- **[Development Guide](../development/README.md)**: Development workflows
- **[Testing Guide](../testing/README.md)**: Testing strategies
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution guidelines

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/README.md)**: Pipeline overview

### Framework Integration
- **[Framework Integration Guide](../gnn/integration/framework_integration_guide.md)**: Framework-specific documentation
- **[PyMDP Integration](../pymdp/gnn_pymdp.md)**: PyMDP API details
- **[RxInfer Integration](../rxinfer/gnn_rxinfer.md)**: RxInfer API details

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with code examples
- **Functionality**: Describes actual API capabilities
- **Completeness**: Comprehensive coverage of all interfaces
- **Consistency**: Uniform structure and style with GNN documentation ecosystem

## See Also

- **[API Reference](../CROSS_REFERENCE_INDEX.md#api-reference--integration)**: Cross-reference index entry
- **[Development Guide](../development/README.md)**: Development workflows
- **[MCP Integration](../mcp/README.md)**: Model Context Protocol documentation
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: ✅ Production Ready  
**Compliance**: Professional documentation standards  
**Maintenance**: Regular updates with new API features and capabilities
