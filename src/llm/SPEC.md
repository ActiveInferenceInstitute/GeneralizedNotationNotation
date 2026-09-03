# LLM Module Specification

## Overview
LLM (Large Language Model) integration for GNN processing.

## Components

### Core
- `processor.py` - LLM processor

### Providers
- Ollama (local), OpenAI, OpenRouter, Perplexity provider modules (no Anthropic module; its key only appears in the provider matrix)

## Features
- GNN to natural language
- LLM-assisted validation
- Model explanation generation

## Key Exports
```python
from llm import process_llm
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
