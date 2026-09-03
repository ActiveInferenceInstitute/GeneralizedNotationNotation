# Research Module

This module (Pipeline Step 19) performs deterministic, rule-based static analysis of GNN models and generates experimental research hypotheses with evidence-backed justification. LLM-powered hypotheses are added opportunistically when an LLM provider is available; the module works fully without one.

## Module Structure

```
src/research/
├── __init__.py                    # Module initialization and exports
├── processor.py                   # Static analysis + hypothesis generation
├── mcp.py                         # MCP tool registrations
└── README.md                      # This documentation
```

## Core Components

### `process_research(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`

Main entry point, called by `19_research.py` (Step 19). Additional kwargs are accepted and ignored (pipeline-template compatibility).

- Per GNN file: detects the model family, extracts state-space dimensions, counts connections
- Runs `generate_rule_based_hypotheses()` to produce hypotheses with discovered evidence
- Opportunistically enriches hypotheses via the shared LLM infrastructure (`llm.llm_processor`) when a provider is configured; failures are logged and non-fatal
- Writes the markdown report and JSON summaries

**Returns:** `bool` — True if processing succeeded.

### `generate_rule_based_hypotheses(content, model_name, output_dir, logger) -> Tuple[List[Dict], str]`

Core rule engine: complexity analysis (high-dimensional matrix detection), structural diagnostics (variable-to-connection ratios), and hypothesis generation with a markdown report justifying every hypothesis.

### `detect_model_family(content: str) -> str`

Model-family detection (POMDP, MDP, continuous, mixed, etc.) from GNN content.

### `extract_state_space_dims(content: str) -> Dict[str, List[int]]` / `count_connections(content: str) -> Dict[str, int]`

Structural feature extraction helpers.

### Exports (`from research import ...`)

- `process_research`
- `FEATURES`, `__version__`

## Usage Examples

### Basic research processing

```python
from research import process_research
from pathlib import Path

success = process_research(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/19_research_output"),
    verbose=True,
)
```

### Direct hypothesis generation

```python
from research.processor import generate_rule_based_hypotheses

hypotheses, report = generate_rule_based_hypotheses(
    content=gnn_content,
    model_name="my_model",
    output_dir=Path("output/19_research_output"),
    logger=logger,
)
```

## Integration with Pipeline

### Pipeline Step 19: Research Processing

`19_research.py` is a thin orchestrator: it parses the standardized `--target-dir`, `--output-dir`, `--recursive`, `--verbose` arguments and delegates to `process_research()`.

### Output Structure

```
output/19_research_output/
├── research_report.md               # Hypotheses with evidence justification
├── research_results.json            # Processing results and hypotheses
├── research_summary.json            # Summary (same payload as results)
└── research_processing_summary.json # Step processing summary
```

## Dependencies

- **Required (stdlib)**: json, logging, re, pathlib — imports are unconditional by design
- **Optional**: LLM provider (Ollama / OpenAI-compatible) for hypothesis enrichment; skipped entirely without one

## Testing

Tests live in `src/tests/research/`: `test_research_overall.py`, `test_research_functional.py`, `test_research_mcp_tools.py`.

```bash
uv run --extra dev python -m pytest src/tests/research/ --cov=src/research
```

## References

- Project overview: ../../README.md
- Pipeline details: ../../doc/pipeline/README.md

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
