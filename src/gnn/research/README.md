# Research Module

This module (Pipeline Step 19) performs deterministic, rule-based static analysis of GNN models and generates experimental research hypotheses with evidence-backed justification. LLM-powered hypotheses are added opportunistically when an LLM provider is available; the module works fully without one.

## Module Structure

```
src/gnn/research/
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

### `generate_rule_based_hypotheses(content, model_family, dims, connections) -> list[dict]`

Core rule engine: pure function from GNN content plus precomputed structural
evidence (model family, state-space dimensions, connection counts) to a list
of hypothesis dicts (`type`, `description`, `rationale`, `priority`).
Rules cover dimensionality, graph density, family-specific gaps, missing
`ActInfOntologyAnnotation`, and missing `InitialParameterization`.

### `analyze_gnn(content: str) -> ModelAnalysis`

One-call pure static analysis returning a frozen `ModelAnalysis`
dataclass bundling `model_family`, `dimensions`, and `connections`.

### `summarize_hypotheses(hypotheses) -> dict`

Pure triage helper: counts hypotheses by `priority` (`high`/`medium`/`low`)
and by `type`, with a `total`. Accepts any iterable of hypothesis mappings.

### `render_research_report(results) -> str` / `write_research_outputs(results_dir, results)`

`render_research_report` is the pure markdown renderer for the report;
`write_research_outputs` writes the three JSON summaries plus exactly that
rendering to `research_report.md` (atomically via temp file + `os.replace`).

### `discover_gnn_files(target_dir, recursive) -> list[Path]`

Sorted `*.md` discovery under `target_dir`; empty list for a missing
directory. `merge_llm_hypotheses(llm, rules)` prefers LLM hypotheses and
dedups rule hypotheses by `type`. `MODEL_FAMILIES` enumerates every value
`detect_model_family` can return.

### `detect_model_family(content: str) -> str`

Model-family detection (POMDP, MDP, continuous, mixed, etc.) from GNN content.

### `extract_state_space_dims(content: str) -> Dict[str, List[int]]` / `count_connections(content: str) -> Dict[str, int]`

Structural feature extraction helpers.

### Exports (`from gnn.research import ...`)

- `process_research`
- `FEATURES`, `__version__`

From `research.processor`: `analyze_gnn`, `ModelAnalysis`, `MODEL_FAMILIES`,
`detect_model_family`, `extract_state_space_dims`, `count_connections`,
`generate_rule_based_hypotheses`, `summarize_hypotheses`,
`render_research_report`, `write_research_outputs`, `discover_gnn_files`,
`merge_llm_hypotheses`

## Usage Examples

### Basic research processing

```python
from gnn.research import process_research
from pathlib import Path

success = process_research(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/19_research_output"),
    verbose=True,
)
```

### Direct analysis and hypothesis generation

```python
from gnn.research.processor import (
    analyze_gnn,
    generate_rule_based_hypotheses,
    summarize_hypotheses,
)

analysis = analyze_gnn(gnn_content)
hypotheses = generate_rule_based_hypotheses(
    content=gnn_content,
    model_family=analysis.model_family,
    dims=analysis.dimensions,
    connections=analysis.connections,
)
summary = summarize_hypotheses(hypotheses)  # {"total", "by_priority", "by_type"}
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

Tests live in `tests/research/`: `test_research_overall.py`, `test_research_functional.py`, `test_research_mcp_tools.py`, `test_research_analysis.py`.

`test_research_analysis.py` pins the pure analysis API: `analyze_gnn` /
`ModelAnalysis`, section-scanning boundaries, `summarize_hypotheses`,
`merge_llm_hypotheses`, report-render purity and parity with the written
report, and sorted `discover_gnn_files`.

```bash
uv run --extra dev python -m pytest tests/research/ --cov=src/gnn/research
```

## References

- Project overview: ../../README.md
- Pipeline details: ../../../doc/pipeline/README.md

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
