# GNN Module Specification

## Overview
Core GNN (Generalized Notation Notation) parsing, validation, and processing.

## Components

### Parsing
- `parsers/` - 20+ format serializers (JSON, YAML, XML, etc.)
- `schema_parser.py` - Schema-based parsing
- `schema_validator.py` - Schema validation (1413 lines)

### Testing
- `testing/test_round_trip.py` - Round-trip validation (1792 lines)

### Processing
- `multi_format_processor.py` - Multi-format processing orchestration

## Key Exports
```python
from gnn import GNNParser, GNNValidator, process_gnn_multi_format
```

## Testing
Tests in `tests/test_gnn_processing.py`
