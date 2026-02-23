# Step 13: LLM — Large Language Model Processing

## Overview

Orchestrates LLM-powered analysis and processing for GNN models. Uses the provider protocol pattern supporting Ollama, OpenAI, and Anthropic backends.

## Usage

```bash
python src/13_llm.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/13_llm.py` (64 lines) |
| Module | `src/llm/` |
| Processor | `src/llm/processor.py` |
| Module function | `process_llm()` |
| Provider protocol | `src/llm/client.py` |
| Embeddings | `src/llm/embeddings.py` |

## Key Capabilities

- LLM-powered GNN specification analysis
- Multi-provider support via `LLMProvider` protocol and `get_provider()` factory
- Embeddings generation for GNN content
- Configurable model selection and timeout

## Output

- **Directory**: `output/13_llm_output/`
- LLM analysis reports and summaries

## Source

- **Script**: [src/13_llm.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/13_llm.py)
