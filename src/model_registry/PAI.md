# Model Registry Module - PAI Context

## Quick Reference

**Purpose:** Central registry for tracking all processed GNN models and their metadata.

**When to use this module:**

- Register new GNN models in the registry
- Query model metadata and version history
- Search models by name, description, or tags
- Track model processing history

## Common Operations

```python
# Batch-register all GNN files in a directory
from model_registry import process_model_registry
results = process_model_registry(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/4_model_registry_output")
)

# Direct registry operations
from model_registry import ModelRegistry
registry = ModelRegistry(registry_path=Path("output/registry.json"))
registry.load()
model = registry.get_model("model_name")
results = registry.search_models("active inference")
all_models = registry.list_models()
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | Parsed GNN model files (.md) |
| **Output** | All modules | Model metadata, version history |

## Key Files

- `registry.py` - Core registry classes (`ModelRegistry`, `ModelEntry`, `ModelVersion`)
- `mcp.py` - MCP tool registrations
- `__init__.py` - Public API exports

## Registry Fields

| Field | Description |
|-------|-------------|
| name | Model identifier (extracted from GNN `ModelName`) |
| version | Current version string |
| description | Model description (extracted from GNN `Description`) |
| author | Author (extracted from GNN `Author`) |
| tags | Searchable tags |
| hash | SHA-256 content hash |
| created_at | Registration timestamp |

## Tips for AI Assistants

1. **Step 4:** Model registry is Pipeline Step 4
2. **Central Index:** Single source of truth for model metadata
3. **Output Location:** `output/4_model_registry_output/`
4. **JSON Format:** Registry persisted as JSON via `registry.save()`
5. **Auto-extraction:** Metadata parsed from GNN section headers

---

**Version:** 1.1.3 | **Step:** 4 (Model Registry)
