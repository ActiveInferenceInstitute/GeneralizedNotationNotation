# Specification: Type-Inference-Zoo

## Design Requirements
This module (`type-inference-zoo`) maps structural logic to the overall execution graph.
It ensures that `Type-Inference-Zoo` tasks resolve without runtime dependency loops.
Role: `type-inference-zoo` is type-system reference material (related sources under `src/gnn/type_systems/`; Step 5 type checking is the separate `src/gnn/type_checker/`) — not a render/execution framework; no entry in `src/gnn/render/framework_registry.py`, no Step 12 executor.

## Components
Expected available types: No specific classes exported.
