# ML Integration Module - PAI Context

## Quick Reference

**Purpose:** Machine learning integrations for model training and inference.

**When to use this module:**
- Integrate with ML frameworks
- Train models from GNN specifications
- Run ML-based inference

## Common Operations

```python
# Run ML integration
from ml_integration.processor import MLIntegrationProcessor
processor = MLIntegrationProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | execute | Simulation data |
| **Output** | analysis | ML results |

## Key Files

- `processor.py` - Main processor class
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Step 14:** ML integration is Step 14
2. **Frameworks:** PyTorch, JAX, TensorFlow support
3. **Output Location:** `output/14_ml_integration_output/`
4. **Training:** Can train from simulation data

---

**Version:** 1.1.3 | **Step:** 14 (ML Integration)
