---
name: gnn-llm-analysis
description: GNN LLM-enhanced analysis and model interpretation. Use when generating natural language descriptions of GNN models, getting AI-assisted model explanations, or performing LLM-powered analysis of Active Inference specifications.
---

# GNN LLM Analysis (Step 13)

## Purpose

Provides LLM-enhanced analysis of GNN models including natural language interpretation, model summarization, structural analysis, and AI-assisted insights using multiple provider backends.

## Key Commands

```bash
# Run LLM analysis
python src/13_llm.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 13 --verbose
```

## Provider Recovery Chain

The LLM module supports multiple providers with automatic recovery:

1. **Ollama** (local) — Preferred for privacy and speed
2. **OpenAI** — GPT-4/3.5 API
3. **Anthropic** / **OpenRouter** — Additional providers

## API

```python
from llm import (
    process_llm, LLMProcessor, LLMAnalyzer,
    analyze_gnn_file_with_llm, extract_variables, extract_connections,
    perform_semantic_analysis, generate_model_insights,
    generate_documentation, generate_llm_summary,
    get_available_providers, get_module_info
)

# Process LLM step (used by pipeline)
process_llm(target_dir, output_dir, verbose=True)

# Use the LLMProcessor class
processor = LLMProcessor()
result = processor.analyze(gnn_content)
description = processor.generate_description(gnn_content)

# Use the LLMAnalyzer class
analyzer = LLMAnalyzer()
insights = analyzer.analyze_content(gnn_content)

# Analyze a file with LLM
result = await analyze_gnn_file_with_llm(content)

# Extract model components
variables = extract_variables(content)
connections = extract_connections(content)

# Check available providers
providers = get_available_providers()  # e.g., ['ollama', 'openai']
```

## Key Exports

- `LLMProcessor` — class with `analyze()`, `analyze_model()`, `generate_description()`
- `LLMAnalyzer` — class with `analyze_content()`, `extract_insights()`
- `analyze_gnn_file_with_llm` — async full analysis
- `extract_variables`, `extract_connections`, `extract_sections` — component extraction
- `generate_model_insights`, `generate_documentation`, `generate_llm_summary`
- `UnifiedLLMProcessor`, `AnalysisType`, `ProviderType` — advanced provider API

## Dependencies

```bash
# LLM PyPI packages are core dependencies (uv sync)
uv sync

# Optional: uv sync --extra llm  (compatibility; same pins)
# Install Ollama CLI separately for local inference: https://ollama.com
```

## Output

- LLM analysis reports in `output/13_llm_output/`
- Natural language model summaries
- AI-generated insights and recommendations


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `process_llm`
- `analyze_gnn_with_llm`
- `generate_llm_documentation`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification
- [providers/](providers/) — Provider-specific implementations


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
