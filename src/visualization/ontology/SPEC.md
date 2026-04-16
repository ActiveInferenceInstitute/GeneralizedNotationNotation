# Ontology Visualization — Technical Specification

**Version**: 1.6.0

## Ontology Categories

- **Beliefs** — Hidden state variables (blue)
- **Observations** — Sensory inputs (green)
- **Actions** — Control outputs (red)
- **Preferences** — Prior preferences over observations (purple)

## Mapping Algorithm

1. Extract variable names from GNN model
2. Match against Active Inference ontology dictionary
3. Generate color-coded diagram with coverage metrics

## Coverage Reporting

- Matched terms: count and percentage
- Unmapped terms: listed with suggestions
