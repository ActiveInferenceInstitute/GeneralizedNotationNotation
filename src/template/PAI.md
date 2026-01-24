# Template Module - PAI Context

## Quick Reference

**Purpose:** Initial template processing and GNN file preparation for the pipeline.

**When to use this module:**
- Process raw GNN input files
- Validate input format and structure
- Prepare files for downstream processing

## Common Operations

```python
# Process templates
from template.processor import TemplateProcessor
processor = TemplateProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | input/gnn_files/ | Raw GNN markdown files |
| **Output** | gnn | Validated, processed files |

## Key Files

- `processor.py` - Main `TemplateProcessor` class
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Step 0:** Template is the first processing step
2. **Validation:** Checks GNN file structure and format
3. **Output Location:** `output/0_template_output/`
4. **Entry Point:** All GNN files must pass through template processing

---

**Version:** 1.1.3 | **Step:** 0 (Template Processing)
