# Step 10: Ontology — Term Mapping and Validation

## Overview

Processes ontology terms, validates GNN models against ontological definitions, and maps relationships between terms.

## Usage

```bash
python src/10_ontology.py --target-dir input/gnn_files --output-dir output --verbose
python src/10_ontology.py --ontology-terms-file path/to/terms.json --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/10_ontology.py` (61 lines) |
| Module | `src/ontology/` |
| Processor | `src/ontology/processor.py` |
| Module function | `process_ontology()` |

## CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--ontology-terms-file` | `Path` | Path to ontology terms JSON file |

## Output

- **Directory**: `output/10_ontology_output/`
- Ontology validation reports, compliance scores, term mappings, and relationship analysis

## Source

- **Script**: [src/10_ontology.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/10_ontology.py)
