# Website Module - PAI Context

## Quick Reference

**Purpose:** Generate static website for model documentation and results.

**When to use this module:**
- Generate HTML documentation site
- Create model galleries
- Build interactive web interfaces

## Common Operations

```python
# Generate website
from website.processor import WebsiteProcessor
processor = WebsiteProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | All prior steps | Results, visualizations |
| **Output** | Deployment | Static HTML site |

## Key Files

- `processor.py` - Main processor class
- `templates/` - HTML templates
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Step 20:** Website generation is Step 20
2. **Static:** Generates static HTML
3. **Output Location:** `output/20_website_output/`
4. **Templates:** Jinja2 templating

---

**Version:** 1.1.3 | **Step:** 20 (Website)
