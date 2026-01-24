# Research Module - PAI Context

## Quick Reference

**Purpose:** Research-oriented analysis and literature connections.

**When to use this module:**
- Connect models to research literature
- Generate research summaries
- Link to Active Inference publications

## Common Operations

```python
# Run research processing
from research.processor import ResearchProcessor
processor = ResearchProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | analysis | Model analysis |
| **Output** | report | Research connections |

## Key Files

- `processor.py` - Main processor class
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Step 19:** Research is Step 19
2. **Literature:** Links to publications
3. **Output Location:** `output/19_research_output/`
4. **Active Inference:** Focus on AI research

---

**Version:** 1.1.3 | **Step:** 19 (Research)
