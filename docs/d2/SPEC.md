# Specification: D2

## Design Requirements
This module (`d2`) maps structural logic to the overall execution graph.
It ensures that `D2` tasks resolve without runtime dependency loops.
Role: `d2` is a visualization capability (D2 diagrams from the advanced visualization module, `src/gnn/advanced_visualization/d2_visualizer.py`, Step 9) — not a render/execution framework; no entry in `src/gnn/render/framework_registry.py`, no Step 12 executor.

## Components
Expected available types: No specific classes exported.
