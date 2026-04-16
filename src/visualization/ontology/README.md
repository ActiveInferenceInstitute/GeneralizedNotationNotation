# Visualization Ontology

Ontology-aware visualization rendering. Maps Active Inference terms to visual elements.

## Exports

- `OntologyVisualizer` — Class that renders ontology term graphs, annotation overlays,
  and semantic relationship diagrams between Active Inference constructs

## Dependencies

- `matplotlib` for rendering
- `ontology` module for term definitions and validation

## Usage

```python
from visualization.ontology import OntologyVisualizer
viz = OntologyVisualizer()
viz.render(parsed_model, output_dir)
```
