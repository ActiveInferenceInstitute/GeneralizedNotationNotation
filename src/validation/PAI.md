# Validation Module - PAI Context

## Quick Reference

**Purpose:** Validate parsed models and check structural integrity.

**When to use this module:**
- Validate POMDP matrix dimensions
- Check model consistency
- Verify state-space properties

## Common Operations

```python
# Run validation
from validation.processor import ValidationProcessor
processor = ValidationProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | Parsed models |
| **Output** | render | Validated models |

## Key Files

- `processor.py` - Main `ValidationProcessor` class
- `validator.py` - Validation logic
- `__init__.py` - Public API exports

## Validation Checks

| Check | Description |
|-------|-------------|
| Dimensions | A, B, C, D matrix sizes |
| Probabilities | Sum to 1, non-negative |
| Completeness | All required fields present |
| Consistency | Cross-matrix compatibility |

## Tips for AI Assistants

1. **Step 6:** Validation is Step 6 of the pipeline
2. **Gates:** Validation gates block bad models
3. **Output Location:** `output/6_validation_output/`
4. **Errors:** Returns detailed validation errors

---

**Version:** 1.1.3 | **Step:** 6 (Validation)
