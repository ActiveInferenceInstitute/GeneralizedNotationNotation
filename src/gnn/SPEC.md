# GNN Module — Specification

## Architecture

The GNN module follows the **Thin Orchestrator** pattern: lightweight coordination scripts delegate to focused processing modules.

## Components

| Component | File | Purpose |
|-----------|------|---------|
| Types | `types.py` | Shared data classes (`ParsedGNN`, `ValidationResult`, `GNNFormat`, etc.) |
| Parser | `parser.py` | GNN file parsing, format detection, validation entry point |
| Schema Validator | `schema_validator.py` | Full `GNNParser` with section-level parsing, `GNNValidator` for multi-level validation |
| Simple Validator | `simple_validator.py` | Lightweight fallback validator without complex dependencies |
| Validation | `validation.py` | `ValidationStrategy` orchestrating multi-level validation |
| Processor | `processor.py` | Lightweight GNN directory processing and file discovery |
| Core Processor | `core_processor.py` | Full pipeline orchestration with phased processing |
| Processors | `processors.py` | Enhanced processing with round-trip and cross-format support |
| Cross-Format | `cross_format_validator.py` | Cross-format consistency validation |
| MCP | `mcp.py` | Model Context Protocol integration for external tool access |
| Multi-Format | `multi_format_processor.py` | Pipeline step for multi-format serialization |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `parsers/` | 48 format-specific parsers and serializers |
| `schemas/` | Schema definition files (JSON, YAML, XSD, Proto, ASN.1, PKL) |
| `grammars/` | BNF and EBNF grammar definitions |
| `testing/` | Round-trip tests, performance benchmarks, integration tests |
| `type_systems/` | Scala and Haskell type system implementations |
| `documentation/` | GNN file format reference (structure, punctuation) |
| `formal_specs/` | Formal mathematical specifications in 8 languages |
| `gnn_examples/` | Reference GNN model files |

## Key Exports

```python
from gnn import (
    discover_gnn_files,      # File discovery
    parse_gnn_file,          # Single-file parsing
    validate_gnn_structure,  # Structure validation
    validate_gnn,            # Full validation (file or content)
    process_gnn_directory,   # Directory processing
    generate_gnn_report,     # Report generation
    GNNParsingSystem,        # Parser registry
    ValidationLevel,         # Validation levels (BASIC, STANDARD, STRICT)
    ParsedGNN,               # Parsed GNN representation
    GNNFormat,               # Format enumeration
)
```

## Testing

```bash
python -m pytest src/tests/test_gnn_overall.py src/tests/test_gnn_validation.py src/tests/test_gnn_parsing.py src/tests/test_gnn_processing.py -v
```
