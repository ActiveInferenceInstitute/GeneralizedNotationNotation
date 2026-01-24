# Pipeline Module - PAI Context

## Quick Reference

**Purpose:** Core pipeline orchestration and configuration management.

**When to use this module:**
- Run the full 25-step pipeline
- Configure pipeline behavior
- Manage step execution order

## Common Operations

```python
# Run full pipeline
from pipeline.runner import PipelineRunner
runner = PipelineRunner(config)
results = runner.run_all()

# Run specific steps
runner.run_steps([0, 3, 11, 12, 16])

# Get configuration
from pipeline.config import get_pipeline_config
config = get_pipeline_config()
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | All modules | Step processors |
| **Output** | All outputs | Orchestrated results |

## Key Files

- `config.py` - Pipeline configuration
- `runner.py` - Pipeline execution
- `__init__.py` - Public API exports

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 0 | template | Input processing |
| 1 | setup | Environment setup |
| 2 | tests | Test validation |
| 3 | gnn | GNN parsing |
| ... | ... | ... |
| 24 | intelligent_analysis | AI-powered analysis |

## Tips for AI Assistants

1. **Orchestrator:** Pipeline coordinates all 25 steps
2. **Thin Pattern:** main.py is thin, delegates to processors
3. **Config:** `pipeline/config.py` has all settings
4. **Output Structure:** `output/{step}_{name}_output/`
5. **Parallel Safe:** Steps can run in parallel where dependencies allow

---

**Version:** 1.1.3 | **Steps:** 0-24 (Full Pipeline)
