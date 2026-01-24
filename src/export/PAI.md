# Export Module - PAI Context

## Quick Reference

**Purpose:** Export parsed models to various formats (JSON, pickle, etc.).

**When to use this module:**
- Export models to JSON format
- Serialize models with pickle
- Create portable model representations

## Common Operations

```python
# Run export
from export.processor import ExportProcessor
processor = ExportProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn, validation | Parsed models |
| **Output** | External tools | JSON, pickle files |

## Key Files

- `processor.py` - Main `ExportProcessor` class
- `__init__.py` - Public API exports

## Export Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| JSON | .json | Portable, human-readable |
| Pickle | .pkl | Python serialization |

## Tips for AI Assistants

1. **Step 7:** Export is Step 7 of the pipeline
2. **Portable:** JSON for cross-platform use
3. **Output Location:** `output/7_export_output/`
4. **Preservation:** Maintains all model metadata

---

**Version:** 1.1.3 | **Step:** 7 (Export)
