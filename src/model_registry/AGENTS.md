# Model Registry Module - Agent Scaffolding

## Module Overview

**Purpose**: Centralized model registry for GNN models with versioning, metadata management, and lifecycle tracking

**Pipeline Step**: Step 4: Model registry (4_model_registry.py)

**Category**: Core Management

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-04-16

---

## Core Functionality

### Primary Responsibilities

1. Register GNN models with unique IDs
2. Track model versions and metadata
3. Manage model lifecycle states
4. Store model relationships and dependencies
5. Enable model discovery and retrieval

### Key Capabilities

- Model registration with automatic ID generation
- Version tracking and history
- Metadata extraction from GNN specifications
- Model search and filtering
- Registry persistence (JSON-based)

---

## API Reference

### Public Functions

#### `process_model_registry(target_dir, output_dir, **kwargs) -> Dict[str, Any]`

**Description**: Main registry processing function. Discovers and registers all GNN files.

**Parameters**:

- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for registry
- `**kwargs`: Additional options

**Returns**: Dictionary with `processed_files`, `successful_registrations`, `registry_path`, and `total_models` counts

**Example**:

```python
from model_registry import process_model_registry

results = process_model_registry(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/4_model_registry_output"),
)
print(
    f"Registered {results['successful_registrations']} of {results['processed_files']} files"
)
```

### Public Classes

#### `ModelRegistry`

**Description**: Main registry class for managing GNN models

**Methods**:

- `register_model(model_path: Path) -> bool` - Register new model
- `get_model(model_id: str) -> Optional[ModelEntry]` - Retrieve model entry
- `search_models(query: str) -> List[ModelEntry]` - Search models by name, description, or tags
- `list_models() -> List[ModelEntry]` - List all registered models
- `delete_model(model_id: str) -> bool` - Delete model from registry
- `save() -> None` - Persist registry to disk
- `load() -> None` - Load registry from disk

**Example**:

```python
registry = ModelRegistry(registry_path)
success = registry.register_model(Path("model.md"))
registry.save()
```

---

## Dependencies

### Required Dependencies

- `json` - Registry persistence
- `pathlib` - File operations
- `datetime` - Timestamp generation

### Internal Dependencies

- `re` - Metadata extraction patterns
- `hashlib` - Model content hashing

---

## Configuration

### Configuration Options

`process_model_registry` accepts two keyword arguments (any others are ignored):

- `registry_path` (str/Path): Custom path for the registry file (default: `output_dir / "model_registry.json"`)
- `query_ontology` (str): If set, keeps only models whose serialized entry contains the given term (case-insensitive substring match)

The step CLI exposes these as `--registry-path` and `--query-ontology`.

---

## Usage Examples

### Basic Usage

```python
from model_registry import process_model_registry

results = process_model_registry(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/4_model_registry_output"),
)
print(f"Registered {results['total_models']} models")
```

### Direct Registry Usage

```python
from model_registry.registry import ModelRegistry

registry = ModelRegistry(Path("registry.json"))
registry.register_model(Path("model.md"))
models = registry.list_models()
registry.save()
```

---
## Output Specification

### Output Products

- `model_registry.json` - Registry database (written by `ModelRegistry.save`): top-level `version`, `updated_at`, and `models` keyed by model ID

### Output Directory Structure

```
output/4_model_registry_output/
└── model_registry.json
```

---

## Registry Schema

### Model Entry Structure

Each entry in `models` is serialized by `ModelEntry.to_dict`:

```json
{
  "model_id": "actinf_pomdp_agent",
  "name": "Active Inference POMDP Agent",
  "description": "",
  "created_at": "2025-09-29T12:00:00",
  "updated_at": "2025-09-29T12:00:00",
  "versions": {
    "1.0.0": {
      "version": "1.0.0",
      "file_path": "input/gnn_files/actinf_pomdp_agent.md",
      "created_at": "2025-09-29T12:00:00",
      "metadata": {},
      "hash": "<sha256 of file content>"
    }
  },
  "tags": [],
  "metadata": {},
  "current_version": "1.0.0"
}
```

`metadata` at both levels holds whatever regex extraction finds in the GNN file (author, date, license).

## Error Handling

- **Unreadable or invalid GNN file**: `register_model` catches the exception, logs it, and returns `False`; `process_model_registry` continues with the remaining files
- **Invalid registry JSON**: `load` logs the error and starts with an empty in-memory registry (existing file is overwritten on the next `save`)
- **Unknown model ID**: `delete_model` returns `False`

There is no file locking, backup, or corruption-recovery mechanism.

---

## Integration Points

### Pipeline Integration

- **Input**: Receives GNN files from Step 3 (gnn processing)
- **Output**: Provides registry data for Step 5 (type checker), Step 6 (validation), and Step 23 (report generation)
- **Dependencies**: Requires GNN parsing results from `3_gnn.py` output

### Module Dependencies

- **gnn/**: Reads parsed GNN model data for registration
- **type_checker/**: Uses registry for model lookup
- **validation/**: Uses registry for model validation
- **report/**: Uses registry for model summaries

### External Integration

- **JSON Storage**: Registry persisted as JSON file
- **File System**: Model metadata stored in directory structure

### Data Flow

```
3_gnn.py (GNN parsing)
  ↓
4_model_registry.py (Model registration)
  ↓
  ├→ 5_type_checker.py (Model lookup)
  ├→ 6_validation.py (Model validation)
  ├→ 23_report.py (Registry summaries)
  └→ output/4_model_registry_output/ (Registry database)
```

---

## Testing

### Test Files

- `src/tests/model_registry/test_model_registry_overall.py`

### Test Coverage

Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/model_registry/ \
    --cov=src/model_registry --cov-report=term-missing
```

---



---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
