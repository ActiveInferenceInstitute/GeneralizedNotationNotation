# Utils Module - PAI Context

## Quick Reference

**Purpose:** Shared utilities used across all pipeline modules.

**When to use this module:**
- Logging configuration
- Argument parsing
- Common file operations
- Pipeline templates

## Common Operations

```python
# Logging setup
from utils.logging_utils import setup_logging
logger = setup_logging("module_name", verbose=True)

# Argument parsing
from utils.argument_utils import get_common_args
args = get_common_args()

# Pipeline templates
from utils.pipeline_template import PipelineTemplate
template = PipelineTemplate(name="my_step")
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | All modules | Configuration, args |
| **Output** | All modules | Utilities, helpers |

## Key Files

- `logging_utils.py` - Logging configuration
- `argument_utils.py` - CLI argument parsing
- `pipeline_template.py` - Step templates
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Shared:** Utils are used by all pipeline steps
2. **Logging:** Consistent logging format across modules
3. **Arguments:** Common CLI args (--verbose, --input-dir, etc.)
4. **Templates:** Base class for processor implementations
5. **No Dependencies:** Utils should not import other src modules

---

**Version:** 1.1.3 | **Used By:** All Steps
