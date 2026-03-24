# Parsers Agent Scaffolding

## Module Overview
**Purpose**: Parsers functionality implementation.
**Status**: Production Ready
**Version**: 1.0.0

## Core Capabilities
- Parsers processing and management
- Integration with GNN pipeline

## API Reference
See the source code for detailed method documentation.

### `GNNParserValidationResult` (`validators.py`)
- Dataclass with `success: bool` plus `issues`, `errors`, `warnings`, `info`.
- **`is_valid`**: read-only property, same as `success`, for callers that expect the cross-module naming used in `parsers/system.py` and schema validation flows.

## Testing
Run tests via `pytest src/tests/`
