# LLM Module - PAI Context

## Quick Reference

**Purpose:** LLM-powered processing for model analysis, explanations, and intelligent insights.

**When to use this module:**
- Generate natural language explanations of models
- Extract insights from simulation results
- Create summaries and technical descriptions

## Common Operations

```python
# Run LLM analysis
from llm.processor import LLMProcessor
processor = LLMProcessor(input_dir, output_dir)
results = processor.process(verbose=True)

# Direct LLM operations
from llm.llm_operations import get_llm_processor
llm = get_llm_processor()
response = llm.generate(prompt, model="gpt-4")
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | execute, analysis | Results, metrics |
| **Output** | report | Explanations, summaries |

## Key Files

- `processor.py` - Main `LLMProcessor` class
- `llm_processor.py` - Core LLM interface
- `llm_operations.py` - High-level operations
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Step 13:** LLM processing is Step 13 of the pipeline
2. **Providers:** Supports OpenAI, Anthropic, Ollama
3. **Output Location:** `output/13_llm_output/`
4. **Prompts:** Uses prompt templates for consistent outputs
5. **API Keys:** Configure in environment or config files

---

**Version:** 1.1.3 | **Step:** 13 (LLM Processing)
