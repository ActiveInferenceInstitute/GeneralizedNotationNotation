# Model Registry Module - Agent Scaffolding

## Module Overview

**Purpose**: Centralized model registry for GNN models with versioning, metadata management, and lifecycle tracking

**Pipeline Step**: Step 4: Model registry (4_model_registry.py)

**Category**: Core Management

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2025-12-30

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

#### `process_model_registry(target_dir, output_dir, verbose=False, logger=None, **kwargs) -> bool`
**Description**: Main registry processing function called by orchestrator (4_model_registry.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for registry
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Logger, optional): Logger instance (default: None)
- `registry_path` (str, optional): Path to model registry file
- `**kwargs`: Additional options

**Returns**: `True` if registry processing succeeded

**Example**:
```python
from model_registry import process_model_registry

success = process_model_registry(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/4_model_registry_output"),
    verbose=True,
    registry_path="registry.json"
)
```

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

## Configuration

### Configuration Options

#### Registry Path
- `registry_path` (Path): Custom path for registry file (default: `output_dir / "model_registry.json"`)
- `registry_backup` (bool): Create backup before overwriting (default: `True`)
- `registry_format` (str): Registry format (default: `"json"`)

#### Model Registration Options
- `auto_version` (bool): Automatically increment version on re-registration (default: `True`)
- `extract_metadata` (bool): Extract metadata from GNN files (default: `True`)
- `validate_before_register` (bool): Validate GNN file before registration (default: `True`)

#### Registry Management
- `max_registry_size` (int): Maximum registry size in MB (default: `100`)
- `cleanup_old_versions` (bool): Remove old versions when max size reached (default: `False`)
- `registry_lock` (bool): Use file locking for concurrent access (default: `True`)

---

## Usage Examples

### Basic Usage
```python
from model_registry import process_model_registry

result = process_model_registry(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/4_model_registry_output")
)
success = result.get("success", False)
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

## Error Handling

### Graceful Degradation
- **Registry File Locked**: Wait and retry, log warning
- **Invalid GNN File**: Skip registration, log error, continue with other files
- **Registry Corruption**: Attempt recovery from backup, log warning
- **Disk Full**: Return error, cannot save registry

### Error Categories
1. **File I/O Errors**: Cannot read/write registry file (fallback: use in-memory registry)
2. **Validation Errors**: Invalid GNN file structure (fallback: skip file, continue)
3. **Version Conflicts**: Model already registered with different version (fallback: auto-increment)
4. **Registry Corruption**: Invalid JSON structure (fallback: restore from backup)

### Error Recovery
- **Backup Restoration**: Automatically restore from backup if registry corrupted
- **Partial Registration**: Register what's possible, report failures
- **Resource Cleanup**: Proper cleanup of registry locks on errors

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
- `src/tests/test_model_registry_integration.py`

### Test Coverage
- **Current**: 80%
- **Target**: 85%+

---


