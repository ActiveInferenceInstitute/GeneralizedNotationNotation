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

## MCP Tools (llm module — 5 real tools)

Registered by `src/llm/mcp.py` via `register_tools()`:

| Tool | Description |
|------|-------------|
| `process_llm` | Run LLM analysis pipeline for all GNN files in a directory |
| `analyze_gnn_with_llm` | Analyse a single GNN file with the configured LLM provider |
| `generate_llm_documentation` | Generate LLM-powered documentation for a GNN model |
| `get_llm_providers` | List available LLM providers and their status (Ollama, OpenAI, Anthropic, Google) |
| `get_llm_module_info` | Return LLM module version and capabilities |

## Source

- **Script**: [src/13_llm.py](#placeholder)
