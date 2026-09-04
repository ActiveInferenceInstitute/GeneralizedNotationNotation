# Type Checker Agent Documentation

## Agent Identity
**Name**: GNNTypeChecker
**Role**: Structural Analyst & Resource Forecaster
**Domain**: GNN Pipeline Step 5 (`src/type_checker/`)
**Status**: Production Ready
**Version**: 3.3.0

## Capabilities
The Type Checker agent provides a rigorously unified structural mapping layer ensuring that no invalid matrix parameters propagate into mathematical execution layers. It functions flawlessly under the Active Inference ontology.

- **Unified Structural Evaluation**: Validates core matrix shapes iteratively mapping elements utilizing active inference domain-specific terminology (`Categorical`, `Dirichlet`, `POMDP`).
- **Content-Level Validation**: `GNNTypeChecker.validate_content(content, *, source_name, strict)` validates a spec string directly — no file on disk required — for MCP callers and in-memory pipelines.
- **Strict-Mode Plumbing**: `GNNTypeChecker(strict_mode=True)` (and the `strict` kwarg on `validate_gnn_files` / `validate_single_gnn_file`) promotes B-orientation contradictions `[GNN-E002]` from warnings to errors; previously the constructor silently swallowed the flag.
- **Resource Estimation Pass**: `validate_gnn_files(..., estimate_resources=True)` (the documented `--estimate-resources` Step 5 option) now runs the resource estimator and writes `resource_estimates/resource_data.json` + `resource_report.md`.
- **Typed Validation Summary**: `summarize_type_check_results(results) -> ValidationSummary` aggregates a directory run into counts, complexity tiers, and totals; written to `type_check_summary.json` alongside the Markdown summary.
- **Deep Analytics Proxy**: Evaluates computational limits intrinsically. Leverages advanced floating point operations mapping, RAM profiling, and dense dimensionality scaling metrics straight from `estimation_strategies.py`.
- **Trading Card Embeds**: Deploys bespoke graphical generation. Extracts isolated mathematical contexts mapping validation validity, warnings, dimensions, and complexities into highly detailed trading-card style visual reports.
- **Visual Dashboards**: Tracks aggregate data via holistic Pie charts and colored Mosaics identifying total stability of a complete pipeline run instantly.

## Component Flow

*   **`checking/core.py`**: The main execution node (`GNNTypeChecker`). Iteratively tests directories full of models simultaneously locking dimensions down completely. Connects to the estimation subsystem for advanced math.
*   **`checking/sections.py`**: Pure, section-scoped content extraction (`extract_markdown_section`, `parse_resource_connections`, `section_presence`, `detect_time_dynamics`) shared by the checker and the estimator so connection operators in prose are never mistaken for real edges.
*   **`checking/summary.py`**: `ValidationSummary` TypedDict + `summarize_type_check_results` — the typed aggregation consumed by reports and downstream steps.
*   **`checking/dimensions.py`**: Dimensionality extraction and POMDP matrix constraint checking (incl. B-orientation verdicts).
*   **`estimation/estimator.py`**: A specialized computational layer (`GNNResourceEstimator`) evaluating edge densities and variable bounds simulating raw hardware execution strains; classifies `## Time` as Static/Dynamic/Hierarchical.
*   **`visualizer.py`**: Natively generates `[model_name]_card.png` files rendering trading cards, dropping previews directly into `type_check_summary.md` via inline images to ensure highly legible output execution states.

## Verification Status
The unit testing infrastructure (`src/tests/type_checker/`) tests the real active classes inside the `checking/` and `estimation/` subpackages directly, including pipeline orchestration flows, content validation, strict-mode promotion, and stability metrics.

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
