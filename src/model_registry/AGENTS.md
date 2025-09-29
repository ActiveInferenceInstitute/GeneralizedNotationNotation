# Model Registry Module - Agent Scaffolding

## Module Overview

**Purpose**: Centralized model registry for GNN models with versioning, metadata management, and lifecycle tracking

**Pipeline Step**: Step 4: Model registry (4_model_registry.py)

**Category**: Core Management

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

#### `process_model_registry_standardized(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main registry processing function

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for registry
- `logger` (Logger): Logger instance
- `registry_path` (Path): Optional custom registry path
- `**kwargs`: Additional options

**Returns**: `True` if registration succeeded

### Public Classes

#### `ModelRegistry`
**Description**: Main registry class for managing GNN models

**Methods**:
- `register_model(gnn_file: Path) -> bool` - Register new model
- `get_model(model_id: str) -> Dict` - Retrieve model metadata
- `list_models() -> List[Dict]` - List all registered models
- `save() -> bool` - Persist registry to disk
- `load() -> bool` - Load registry from disk

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
- `utils.pipeline_template` - Logging utilities
- `pipeline.config` - Configuration management

---

## Usage Examples

### Basic Usage
```python
from model_registry import process_model_registry_standardized

success = process_model_registry_standardized(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/4_model_registry_output"),
    logger=logger
)
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
- `model_registry.json` - Main registry database
- `model_registry_summary.json` - Processing summary
- `{model_id}/` - Per-model metadata directories

### Output Directory Structure
```
output/4_model_registry_output/
├── model_registry.json
├── model_registry_summary.json
└── {model_id}/
    └── {model_id}_metadata.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 56ms
- **Memory**: 28.7 MB
- **Status**: SUCCESS
- **Models Registered**: 1

---

## Registry Schema

### Model Entry Structure
```json
{
  "model_id": "actinf_pomdp_agent",
  "model_name": "Active Inference POMDP Agent",
  "file_path": "input/gnn_files/actinf_pomdp_agent.md",
  "file_size_bytes": 1759,
  "registered_at": "2025-09-29T12:00:00",
  "version": "1.0.0",
  "metadata": {
    "variables": 13,
    "connections": 11,
    "formats_generated": 22
  }
}
```

---

## Testing

### Test Files
- `src/tests/test_model_registry_integration.py`

### Test Coverage
- **Current**: 80%
- **Target**: 85%+

---

**Last Updated**: September 29, 2025  
**Status**: ✅ Production Ready


