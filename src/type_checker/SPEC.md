# Specification: Type Checker Subsystem

## Overview
The `src/type_checker/` module represents a strict architectural barrier guaranteeing that invalidly mapped Generative Notation models are accurately constrained before mathematical serialization.

## Interface Mapping
- `5_type_checker.py`: The single executable orchestrator bounding `processor.GNNTypeChecker`.
- `estimation_strategies.py`: Computational Algebra mappings measuring matrix load requirements.
- `visualizer.py`: The isolated generation of all model Dashboard reporting structures.

## Standards 
- Rejects legacy mathematical assignments (e.g. `error = sum`); evaluating strict multidimensional references intrinsically bounded (e.g. `s[3,1,type=float]`).
- Evaluates against Active Inference topologies directly (`POMDP`, `Categorical`, `Continuous`).
- Output formats MUST natively synthesize both `.json` data files tracking parameter scores AND high-level `matplotlib` abstractions (Dashboards/Mosaics/Model Cards) straight into `type_check_summary.md`.
