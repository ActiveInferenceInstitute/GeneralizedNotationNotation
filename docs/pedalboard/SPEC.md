# Specification: Pedalboard

## Design Requirements
This module (`pedalboard`) maps structural logic to the overall execution graph.
It ensures that `Pedalboard` tasks resolve without runtime dependency loops.
Role: `pedalboard` is an audio post-processing dependency in the Step 15 audio-sonification path (`src/gnn/audio/`, optional backend alongside soundfile) — not a render/execution framework; no entry in `src/gnn/render/framework_registry.py`, no Step 12 executor.

## Components
Expected available types: No specific classes exported.
