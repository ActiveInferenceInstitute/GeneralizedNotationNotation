# GUI Module - PAI Context

## Quick Reference

**Purpose:** Graphical user interface components for model interaction.

**When to use this module:**
- Create model configuration GUIs
- Build visual model editors
- Generate interactive controls

## Common Operations

```python
# Generate GUI components
from gui.processor import GUIProcessor
processor = GUIProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | Model structure |
| **Output** | website | GUI components |

## Key Files

- `processor.py` - Main processor class
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Step 22:** GUI generation is Step 22
2. **Components:** Generates UI elements
3. **Output Location:** `output/22_gui_output/`
4. **Interactive:** Web-based interfaces

---

**Version:** 1.1.3 | **Step:** 22 (GUI)
