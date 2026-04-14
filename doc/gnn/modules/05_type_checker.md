# Step 5: Type Checker

## Architectural Mapping

**Orchestrator**: `src/5_type_checker.py` (65 lines)
**Implementation Layer**: `src/type_checker/`

## Module Description

This module provides the rigorous, integrated validation layer powering the GNN zero-mock processing pipeline. It evaluates syntax structures, maps multidimensional parameters across generative mathematical bounds, performs full structural cross-validation, and renders executive dashboard trading cards evaluating physical resource estimations natively.

The Type Checker subsystem was structurally unified in Version 1.1.4, deprecating isolated redundancy and consolidating logic natively into the production pipeline.

```mermaid
graph TD
    Pipeline[5_type_checker.py] --> Orchestrator[processor.py: GNNTypeChecker]
    Orchestrator --> MathExt[estimation_strategies.py]
    Orchestrator --> MatrixOutput[visualizer.py]
    MatrixOutput --> ExecSumm[type_checker_summary.md]
    MatrixOutput --> Cards[visualizations/cards/]
```

The central `GNNTypeChecker` orchestrator class evaluated directly by the main pipeline flow. It scans all nested GNN files via recursive mapping, extracts semantic type arrays like `Categorical` or `POMDP` nodes, checks the syntactical bounds of connections, and passes the entire payload into the downstream resource and visual evaluators seamlessly.

## Agent Identity & Capabilities

# Type Checker Agent Documentation

## Agent Identity
**Name**: GNNTypeChecker
**Role**: Structural Analyst & Resource Forecaster
**Domain**: GNN Pipeline Step 5 (`src/type_checker/`)

## Capabilities
The Type Checker agent provides a rigorously unified structural mapping layer ensuring that no invalid matrix parameters propagate into mathematical execution layers. It functions flawlessly under the Active Inference ontology.

- **Unified Structural Evaluation**: Validates core matrix shapes iteratively mapping elements utilizing active inference domain-specific terminology (`Categorical`, `Dirichlet`, `POMDP`).
- **Deep Analytics Proxy**: Evaluates computational limits intrinsically. Leverages advanced floating point operations mapping, RAM profiling, and dense dimensionality scaling metrics straight from `estimation_strategies.py`.
- **Trading Card Embeds**: Deploys bespoke graphical generation. Extracts isolated mathematical contexts mapping validation validity, warnings, dimensions, and complexities into highly detailed trading-card style visual reports.
- **Visual Dashboards**: Tracks aggregate data via holistic Pie charts and colored Mosaics identifying total stability of a complete pipeline run instantly.

## Component Flow

*   **`processor.py`**: The main execution node. Iteratively tests directories full of models simultaneously locking dimensions down completely. Connects automatically to `estimation_strategies.py` for advanced math.
*   **`estimation_strategies.py`**: A specialized computational array evaluating edge densities and variable bounds simulating raw hardware execution strains.
*   **`visualizer.py`**: Natively generates `[model_name]_card.png` files rendering trading cards, dropping previews directly into `type_check_summary.md` via inline images to ensure highly legible output execution states.

## Zero-Mock Status
The unit testing infrastructure (`src/tests/test_type_checker_overall.py`) strictly isolates down to `processor.GNNTypeChecker` directly simulating pipeline orchestration flows without isolated duplicate boundaries, retaining robust stability scores globally.


---

**Source Reference**: [src/type_checker](../../../src/type_checker)
