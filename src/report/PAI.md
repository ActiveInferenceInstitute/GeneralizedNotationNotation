# Report Module - PAI Context

## Quick Reference

**Purpose:** Generate comprehensive reports from pipeline execution results.

**When to use this module:**
- Create HTML/Markdown analysis reports
- Generate executive summaries
- Compile cross-framework comparisons

## Common Operations

```python
# Generate reports
from report.processor import ReportProcessor
processor = ReportProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | All prior steps | Results, analyses |
| **Output** | Final deliverables | HTML, Markdown reports |

## Key Files

- `processor.py` - Main `ReportProcessor` class
- `templates/` - Report templates
- `__init__.py` - Public API exports

## Report Types

| Type | Format | Description |
|------|--------|-------------|
| Analysis | HTML | Comprehensive analysis |
| Summary | Markdown | Executive summary |
| Comparison | HTML | Cross-framework comparison |

## Tips for AI Assistants

1. **Step 23:** Report generation is Step 23
2. **Templates:** Uses Jinja2 for report generation
3. **Output Location:** `output/23_report_output/`
4. **Aggregation:** Combines all prior step outputs
5. **Formats:** Generates both HTML and Markdown

---

**Version:** 1.1.3 | **Step:** 23 (Report Generation)
