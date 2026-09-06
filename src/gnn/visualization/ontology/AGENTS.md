# Visualization Ontology Sub-module

## Overview

Ontology-specific visualization for Active Inference terminology. Generates visual mappings between GNN model elements and Active Inference ontology concepts.

## Architecture

```
ontology/
├── __init__.py       # Package exports
└── visualizer.py     # OntologyVisualizer (annotation table visualization)
```

## Key Functions

- **`OntologyVisualizer.visualize_directory(input_dir, output_dir) -> List[str]`** — Visualizes ontology annotations from every GNN file in a directory; returns generated artifact paths.
- **`OntologyVisualizer.visualize_ontology(parsed_data, output_dir) -> Optional[str]`** — Renders an annotated ontology-table PNG (`ontology_annotations.png`) from the model's `ActInfOntologyAnnotation` section; returns `None` when the section is absent.
- **Coverage reporting** — The table shows which Active Inference terms are present and their annotations.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 3.2.0
