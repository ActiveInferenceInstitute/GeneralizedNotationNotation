# Intelligent Analysis - PAI Context

## Quick Reference

**Purpose:** AI-powered analysis of entire pipeline execution, generating executive summaries and insights.

**When to use this module:**
- Generate comprehensive analysis after full pipeline run
- Create executive summaries of model processing
- Analyze cross-framework patterns and insights

## Common Operations

```bash
# Run intelligent analysis (Step 24)
python 24_intelligent_analysis.py --input-dir output --output-dir output/24_intelligent_analysis_output --verbose

# Use as module
from intelligent_analysis.processor import IntelligentAnalysisProcessor
processor = IntelligentAnalysisProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | All prior steps | output/0-23_*_output/ |
| **Output** | Final reports | executive_summary.md, insights.json |

## Key Files

- `processor.py` - Main `IntelligentAnalysisProcessor` class
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Final Step:** This is Step 24, runs after all other processing
2. **LLM Integration:** Uses configured LLM for intelligent insights
3. **Cross-Framework:** Analyzes patterns across all framework outputs
4. **Report Generation:** Creates human-readable executive summaries

---

**Version:** 1.1.3 | **Step:** 24 (Intelligent Analysis)
