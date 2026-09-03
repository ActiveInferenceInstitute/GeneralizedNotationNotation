# Visualization Ontology

Ontology-aware visualization rendering. Maps Active Inference terms to visual elements.

## Exports

- `OntologyVisualizer` — Class that renders an annotated matplotlib table of a
  model's `ActInfOntologyAnnotation` section, plus batch visualization for whole directories

## Dependencies

- `matplotlib` for rendering
- `src/ontology/` module for term definitions

## Usage

```python
from visualization.ontology import OntologyVisualizer

viz = OntologyVisualizer()
viz.visualize_ontology(parsed_model, output_dir)   # single model; None if no annotations
viz.visualize_directory(Path("input/gnn_files"), output_dir)  # batch
```
