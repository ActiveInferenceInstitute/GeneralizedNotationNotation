# CatColab Framework Implementation

> **GNN Integration Layer**: Categorical / Structural (bidirectional)
> **Framework Base**: [CatColab](https://catcolab.org) (Topos Institute) + [AlgebraicJulia](https://algebraicjulia.org)
> **Integration Type**: Bidirectional — GNN → CatColab export and CatColab → GNN import
> **Documentation Version**: 1.0

## Overview

CatColab is a collaborative environment for formal scientific modeling using double-categorical theory, developed by the Topos Institute. Unlike the numerical simulation frameworks (PyMDP, JAX, RxInfer), CatColab provides **structural** and **categorical** output — string diagrams, factor graphs, and formally-verified compositional models rather than simulation traces.

GNN and CatColab are **complementary**: GNN specifies the numerical Active Inference parameters; CatColab provides formal model-theoretic verification and composition.

## Architecture

CatColab integration operates across three pipeline stages:

| Stage | GNN Module | CatColab Role |
|-------|-----------|---------------|
| Export (Step 7) | `src/export/` | GNN JSON → CatColab JSON (Schema/Stock-and-Flow/Olog) |
| Render (Step 11) | `src/render/discopy/` | DisCoPy string diagrams (shared categorical foundation) |
| Execute (Step 12) | `src/execute/discopy/` | Categorical evaluation of circuit structure |

## Conceptual Bridges

### GNN Matrices → CatColab Logics

| GNN Matrix | CatColab Logic | Categorical Role |
|------------|----------------|-----------------|
| **A** (Likelihood) | Schema | Profunctor from states to observations |
| **B** (Transition) | Stock-and-Flow | Controlled dynamical system |
| **C** (Preference) | Regulatory Network | Signed evaluation functor |
| **D** (Prior) | Olog | Probabilistic initialization |
| **E** (Habit) | Petri Net | Stochastic policy selection |

### GNN Sections → CatColab Constructs

| GNN Section | CatColab Equivalent |
|-------------|---------------------|
| `StateSpaceBlock` | Objects and types |
| `Connections` | Morphisms and relations |
| `Equations` | 2-cells (coherence conditions) |
| `Parameters` | Instance data |
| `Annotations` | Olog propositions |

## Installation

CatColab is a web application — no local installation required:

1. Open [catcolab.org](https://catcolab.org)
2. Create a new document (choose logic: Schema, Petri Net, Olog, etc.)
3. Import GNN-generated JSON (see export below)

For the Julia backend (AlgebraicJulia):

```julia
# Julia >= 1.9 required
using Pkg
Pkg.add(["Catlab", "AlgebraicDynamics", "AlgebraicPetri"])
```

## Export GNN → CatColab

```bash
# Export to CatColab-compatible JSON via Step 7
python src/7_export.py --target-dir input/gnn_files/ --format catcolab --verbose
# Output: output/7_export_output/<model>/model_catcolab.json

# Or run the full pipeline with export
python src/main.py --only-steps "3,7" --verbose
```

**Logic selection guide:**

| GNN Model Type | Choose CatColab Logic |
|----------------|-----------------------|
| Perception-only POMDP | Schema |
| Full POMDP with dynamics | Stock-and-Flow Diagram |
| Multi-agent system | Regulatory Network |
| Discrete action selection | Petri Net |
| Ontology-annotated | Olog |

## Import CatColab → GNN

1. In CatColab, export your model: **File → Download → JSON**
2. Run the GNN importer:

```bash
python src/gnn/catcolab_importer.py \
    --input path/to/catcolab_export.json \
    --output input/gnn_files/imported_model.md \
    --logic stock-and-flow
```

1. Validate and run the pipeline normally.

## DisCoPy as Bridge Technology

`src/render/discopy/` provides the categorical bridge between GNN and CatColab:

- GNN connection graphs → DisCoPy string diagrams
- DisCoPy morphisms → CatColab Regulatory Network morphisms
- Both share the **monoidal category** mathematical foundation

```bash
# Render GNN model to DisCoPy string diagram (for CatColab import)
python src/11_render.py --target-dir input/gnn_files/ --framework discopy
```

## Telemetry Output

DisCoPy/CatColab execution produces structural (not numerical) telemetry:

```json
{
  "circuit_info": {
    "num_morphisms": 12,
    "num_objects": 5,
    "diagram_type": "monoidal",
    "composition_valid": true
  },
  "categorical_summary": {
    "objects": ["HiddenState", "Observation", "Policy"],
    "morphisms": ["A_likelihood", "B_transition", "G_efe"]
  }
}
```

## Correlation Results

CatColab/DisCoPy provides structural output only and is **excluded** from numerical cross-framework correlation analysis.

## Source Code Connections

| Stage | Module | Key Function |
|-------|--------|-------------|
| Export | [discopy_renderer.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/render/discopy/discopy_renderer.py) | `render_gnn_to_discopy()` |
| Execute | [discopy_executor.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/execute/discopy/discopy_executor.py) | `execute_discopy_script()` |
| Analysis | [analysis/discopy/analyzer.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/analysis/discopy/analyzer.py) | `generate_analysis_from_logs()` |

## Deep Dive

Full categorical bridge documentation: **[doc/catcolab/catcolab_gnn.md](../../../doc/catcolab/catcolab_gnn.md)**

## Improvement Opportunities

| ID | Area | Description | Impact |
|----|------|-------------|--------|
| C-1 | Export | CatColab JSON schema not yet formally validated | Medium |
| C-2 | Import | CatColab→GNN importer stubs partial logic types | Medium |
| C-3 | Analysis | DisCoPy analyzer only outputs 2 structural diagrams | Low |
