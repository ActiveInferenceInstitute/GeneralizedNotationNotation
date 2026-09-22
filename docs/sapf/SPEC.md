# Specification: Sapf

## Design Requirements
This module (`sapf`) maps structural logic to the overall execution graph.
It ensures that `Sapf` tasks resolve without runtime dependency loops.
Role: `sapf` is an audio sonification capability at Step 15 (`src/gnn/audio/sapf/`; Step 15 audio backend) — not a render/execution framework; no entry in `src/gnn/render/framework_registry.py`, no Step 12 executor.

## Components
Expected available types: No specific classes exported.
