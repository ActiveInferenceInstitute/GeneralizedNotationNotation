# Type Checker Module - PAI Context

## Quick Reference

**Purpose:** Type checking for GNN models and generated code.

**When to use this module:**
- Verify type consistency in models
- Check generated code types
- Validate matrix type signatures

## Common Operations

```python
# Run type checking
from type_checker.processor import TypeCheckerProcessor
processor = TypeCheckerProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn, render | Models, code |
| **Output** | validation | Type check results |

## Key Files

- `processor.py` - Main processor class
- `checker.py` - Type checking logic
- `__init__.py` - Public API exports

## Type Checks

| Check | Description |
|-------|-------------|
| Matrix Types | A, B, C, D type consistency |
| Dimensions | Shape compatibility |
| Code Types | Generated code type hints |

## Tips for AI Assistants

1. **Step 5:** Type checking is Step 5
2. **Static Analysis:** No runtime execution
3. **Output Location:** `output/5_type_checker_output/`
4. **mypy Compatible:** Works with Python type hints

---

**Version:** 1.1.3 | **Step:** 5 (Type Checking)
