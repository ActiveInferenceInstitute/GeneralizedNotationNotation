---
name: gnn-ontology
description: GNN Active Inference ontology processing and validation. Use when working with ActInfOntologyAnnotation sections, mapping GNN variables to ontology terms, validating semantic annotations, or exploring Active Inference concept hierarchies.
---

# GNN Ontology Processing (Step 10)

## Purpose

Processes Active Inference ontology annotations in GNN models. Maps GNN variables to standardized Active Inference concepts and validates semantic consistency.

## Key Commands

```bash
# Run ontology processing
python src/10_ontology.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 10 --verbose
```

## Ontology Term Mapping

| GNN Annotation | Ontology Concept | Description |
| ---------------- | ------------------ | ------------- |
| `LikelihoodMatrix` | A matrix | Observation model |
| `TransitionMatrix` | B matrix | State dynamics |
| `HiddenState` | s vector | Latent state beliefs |
| `Observation` | o vector | Sensory observations |
| `Policy` | π | Action sequences |
| `Preference` | C matrix | Preferred observations |

## API

```python
from ontology import (
    process_ontology, parse_gnn_ontology_section, process_gnn_ontology,
    load_defined_ontology_terms, validate_annotations,
    generate_ontology_report_for_file, parse_annotation,
    OntologyProcessor, OntologyValidator
)

# Process ontology section from GNN content
parsed = parse_gnn_ontology_section(gnn_content)

# Load defined ontology terms
terms = load_defined_ontology_terms()

# Validate annotations against defined terms
result = validate_annotations(annotations_list, terms)

# Use OntologyProcessor class
processor = OntologyProcessor()
result = processor.process_ontology(data)

# Use OntologyValidator class
validator = OntologyValidator()
is_valid = validator.validate_ontology(content)
```

## Key Exports

- `process_ontology` — main pipeline processing function
- `parse_gnn_ontology_section` — extract ontology from GNN content
- `validate_annotations` — validate against known terms
- `OntologyProcessor` — class with `process_ontology()`, `validate_terms()`
- `OntologyValidator` — class with `validate_ontology()`, `check_consistency()`

## Output

- Ontology mappings in `output/10_ontology_output/`
- Validation reports for annotation consistency


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `extract_ontology_annotations`
- `list_standard_ontology_terms`
- `process_ontology`
- `validate_ontology_terms`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
