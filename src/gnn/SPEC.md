# GNN Module — Specification

## Architecture

The GNN module follows the **Thin Orchestrator** pattern: lightweight coordination scripts delegate to focused processing modules.

## Format and serializer counts (canonical)

These definitions are the single source of truth for cross-references in READMEs and status docs:

| Concept | Count | Notes |
|---------|------:|--------|
| **`GNNFormat` enum** | **23** | All supported formats, defined in `parsers/common.py`. |
| **Parsers registered** | **23** | `PARSER_REGISTRY` in `parsers/system.py` — one parser class per format. |
| **Serializers registered** | **22** | `SERIALIZER_REGISTRY` in `parsers/system.py` — **PNML** has a parser but no dedicated serializer (parse-focused / XML-related). |
| **Round-trip test list** | **21** strings in `testing/test_round_trip.py` `FORMAT_TEST_CONFIG['test_formats']` | Includes `markdown` plus **20** other formats. **EBNF** and **PNML** are not in this list (PNML round-trip disabled in config). |

When a document says “100% round-trip,” it refers to the **formats exercised by** `test_round_trip.py`, not necessarily every enum value.

## Components

| Component | File | Purpose |
|-----------|------|---------|
| Types | `types.py` | Shared data classes (`ParsedGNN`, `ValidationResult`, `GNNFormat`, etc.) |
| Parser | `parser.py` | GNN file parsing, format detection, validation entry point |
| Schema Validator | `schema_validator.py` | Full `GNNParser` with section-level parsing, `GNNValidator` for multi-level validation |
| Simple Validator | `simple_validator.py` | Lightweight recovery validator without complex dependencies |
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
| `parsers/` | Format-specific parsers & serializers: `PARSER_REGISTRY` (23) and `SERIALIZER_REGISTRY` (22); per-format `*_parser.py` / `*_serializer.py` modules in `parsers/system.py` |
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
uv run pytest src/tests/test_gnn_overall.py src/tests/test_gnn_validation.py src/tests/test_gnn_parsing.py src/tests/test_gnn_processing.py -v
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
