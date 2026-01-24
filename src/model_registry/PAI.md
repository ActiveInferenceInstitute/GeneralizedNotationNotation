# Model Registry Module - PAI Context

## Quick Reference

**Purpose:** Central registry for tracking all processed models and their metadata.

**When to use this module:**
- Register new models in the registry
- Query model metadata
- Track model versions and processing history

## Common Operations

```python
# Register model
from model_registry.processor import ModelRegistryProcessor
processor = ModelRegistryProcessor(input_dir, output_dir)
results = processor.process(verbose=True)

# Query registry
from model_registry.registry import ModelRegistry
registry = ModelRegistry(registry_path)
model = registry.get_model("model_name")
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | Parsed models |
| **Output** | All modules | Model metadata |

## Key Files

- `processor.py` - Main processor class
- `registry.py` - Registry operations
- `__init__.py` - Public API exports

## Registry Fields

| Field | Description |
|-------|-------------|
| name | Model identifier |
| version | Model version |
| source | Original GNN file |
| processed | Processing timestamp |
| frameworks | Rendered frameworks |

## Tips for AI Assistants

1. **Step 4:** Model registry is Step 4
2. **Central Index:** Single source of truth for models
3. **Output Location:** `output/4_model_registry_output/`
4. **JSON Format:** Registry stored as JSON

---

**Version:** 1.1.3 | **Step:** 4 (Model Registry)
