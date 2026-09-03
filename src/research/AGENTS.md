# Research Module - Agent Scaffolding

## Module Overview

**Purpose**: Deterministic rule-based static analysis of GNN models with experimental hypothesis generation; LLM-powered hypotheses are added opportunistically when an LLM provider is available.

**Pipeline Step**: Step 19: Research tools (19_research.py)

**Category**: Research / Experimental Analysis

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-09-02

---

## Core Functionality

1. Detect model family (POMDP, MDP, continuous, mixed, etc.) from GNN content
2. Extract state-space dimensions and connection counts
3. Generate evidence-backed hypotheses from rule-based structural diagnostics
4. Opportunistically enrich hypotheses via LLM when a provider is available
5. Write a markdown research report and JSON result summaries

---

## API Reference

### Public Functions

#### `process_research(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
**Description**: Main research processing function called by orchestrator (19_research.py). Runs rule-based analysis per GNN file and writes reports.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for research results
- `verbose` (bool): Enable verbose logging (default: False)
- `**kwargs`: Accepted and ignored; reserved for pipeline-template compatibility

**Returns**: `bool` - True if research processing succeeded, False otherwise

**Example**:
```python
from research import process_research
from pathlib import Path

success = process_research(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/19_research_output"),
    verbose=True,
)
```


#### `generate_rule_based_hypotheses(content: str, model_name: str, output_dir: Path, logger: logging.Logger) -> Tuple[List[Dict], str]`
**Description**: Core rule-based hypothesis generation engine. Analyzes GNN model content, detects complexity patterns, structural diagnostics, and generates evidence-backed hypotheses.

**Parameters**:
- `content` (str): Raw GNN file content
- `model_name` (str): Name of the model being analyzed
- `output_dir` (Path): Output directory for reports
- `logger` (logging.Logger): Logger instance

**Returns**: `Tuple[List[Dict], str]` - (hypotheses list, markdown report)

#### `detect_model_family(content: str) -> str`
**Description**: Detect the model family (e.g., POMDP, MDP, continuous, mixed) from GNN content.

#### `extract_state_space_dims(content: str) -> Dict[str, List[int]]`
**Description**: Extract state space dimensions from variables in GNN content.

#### `count_connections(content: str) -> Dict[str, int]`
**Description**: Count connections by type (directed, undirected) in GNN content.

---

## Dependencies

### Required Dependencies
Standard library only (`json`, `logging`, `re`, `pathlib`) — imports are unconditional by design; no external analysis packages are used.

### Optional Dependencies
- LLM provider (Ollama / OpenAI-compatible) - opportunistic LLM-powered hypothesis enrichment; skipped entirely without one

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities
- `llm.llm_processor` - Optional LLM hypothesis generation

---

## Configuration

### Environment Variables

None dedicated to this module. Hypothesis generation is fixed by
`generate_rule_based_hypotheses()` rules in `research/processor.py`; LLM
enrichment uses the shared Step 13+ LLM configuration (`OLLAMA_MODEL` env /
`llm.defaults`).

---

## Usage Examples

### Basic Research Analysis
```python
from research.processor import process_research

success = process_research(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/19_research_output"),
    verbose=True,
)
```

---

## Output Specification

### Output Products
- `research_report.md` - Research report
- `research_results.json` - Processing results
- `research_summary.json` - Summary (same payload as results)
- `research_processing_summary.json` - Step processing summary

### Output Directory Structure
```
output/19_research_output/
├── research_report.md
├── research_results.json
├── research_summary.json
└── research_processing_summary.json
```

---

## Performance Characteristics

### Latest Execution

- Rule-based analysis completes in milliseconds per model (pure static analysis)
- LLM enrichment adds one provider round-trip per model when enabled

---

## Error Handling

### Research Errors
1. **Data Quality Issues**: GNN files with missing or malformed sections (fewer hypotheses generated)
2. **Analysis Failures**: Per-file analysis failures logged and skipped
3. **LLM Errors**: Enrichment failures logged; rule-based results still written

### Recovery Strategies
- **Analysis Recovery**: Remaining files still analyzed when one fails
- **LLM Recovery**: Rule-based hypotheses and report written regardless of LLM availability

---
## Integration Points

### Orchestrated By
- **Script**: `19_research.py` (Step 19)
- **Function**: `process_research()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `src/tests/research/*` - Research tests

### Data Flow
```
GNN Files → Static Analysis (family, dims, connections) → Rule-Based Hypotheses → (optional) LLM Enrichment → research_report.md + JSON summaries
```

---

## Testing

### Test Files
- `src/tests/research/test_research_overall.py` - Module-level tests
- `src/tests/research/test_research_functional.py` - Functional tests
- `src/tests/research/test_research_mcp_tools.py` - MCP tool tests

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/research/ \
    --cov=src/research --cov-report=term-missing
```

### Key Test Scenarios
1. Model-family detection across GNN variants
2. Hypothesis generation from structural rules
3. Degradation without an LLM provider
4. Error handling with malformed models

---

## MCP Integration

### Tools Registered
- `process_research` - Run research processing on a directory
- `list_research_topics` - List available research analysis topics
- `read_research_results` - Read results from a previous research run
- `get_research_module_info` - Return module metadata

### MCP File Location
- `src/research/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Hypothesis generation produces no results
**Symptom**: Research analysis completes but no hypotheses generated
**Cause**: Model structure doesn't match rule patterns (e.g., missing dimension blocks or Connections section)
**Solution**:
- Verify GNN model has complete StateSpaceBlock/Connections sections
- Check that model has variables and connections
- Use `--verbose` flag for detailed analysis logs

#### Issue 2: LLM enrichment absent
**Symptom**: Only rule-based hypotheses in the report
**Cause**: No LLM provider available or Step 13+ LLM configuration unset
**Solution**:
- Start Ollama or configure a provider; this is optional by design

---

## Version History

### Current Version: 1.6.0 (module `__init__.py`), pipeline release 3.2.0

**Features**:
- Rule-based hypothesis generation
- Model-family detection and structural diagnostics
- Automated evidence-backed reporting

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced hypothesis generation
- **Future**: Machine learning-based hypothesis generation

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Research Module](../research/README.md)

### External Resources
- [Active Inference Research](../../doc/research/README.md)

---

**Last Updated**: 2026-09-02
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: 3.2.0
**Architecture Compliance**: Thin Orchestrator Pattern

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
