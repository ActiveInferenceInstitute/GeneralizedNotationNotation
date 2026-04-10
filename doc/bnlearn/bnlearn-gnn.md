# bnlearn-GNN Integration Architecture

This document defines the architectural integration pathways between the `bnlearn` Python package and the 25-step **GNN (Generalized Notation Notation)** pipeline utilized for processing Active Inference generative models.

## Concept Overview

Active Inference relies natively on Partially Observable Markov Decision Processes (POMDPs) modeled using Generative Graphical structures. `bnlearn` provides a robust Bayesian abstraction that integrates naturally into the GNN pipeline, acting as a mathematical bridge for structure learning and inference on simulated traces. By converting generalized notation payloads into `bnlearn` DAGs and DataFrames, the GNN ecosystem can conduct deep causal validations, parameter tracking, and interactive visualizations.

---

## GNN Pipeline Touchpoints

### Step 3: GNN File Discovery & Parsing
Within the `src/3_gnn.py` thin orchestrator layer, `bnlearn` can consume structurally serialized model dictionaries. 
*   **Structural Import**: GNN representations of generative models ($A$, $B$, $C$, $D$ matrices) can be directly transformed into an adjacency matrix using `bn.vec2adjmat()` or `bn.dag2adjmat()`.
*   **Validation Check**: Passing the extracted matrices through `bn.check_model()` immediately surfaces disconnected nodes or cyclical violations, ensuring rigorous structural integrity early in the pipeline.

### Step 5 & Step 6: Validation and Consistency Verification
Once the types are correctly deduced in Step 5, the model parameters (CPTs corresponding to standard POMDP matrices) are verified:
*   **Edge Strength**: `bn.independence_test()` is deployed against structural traces to validate expected epistemic separation properties natively assumed by the GNN ontological model.
*   **Topological Scoring**: Metrics exposed by `bn.structure_scores()` evaluate network robustness. 

### Step 8: Visualization
The network visualization framework in `bnlearn` complements GNN's Graph and Matrix Visualization scripts.
*   **Interaction graphs**: Calling `bn.plot(model, interactive=True)` emits a dynamic `D3.js` HTML artifact that GNN packages into the `output/` directory for visual evaluation.
*   **Comparison tracking**: For models moving through registry tracking (Step 4), `bn.compare_networks()` provides visual diffs of active inference networks evolving across versions.

### Step 12: Simulation & Execute
Active Inference execution engines (`pymdp`, `rxinfer`) inherently generate trajectory traces as they sample hidden state inferences over time.
*   The raw simulation traces produced in `Step 12: Execute` act as the raw historical `DataFrame` fed into `bnlearn`. 
*   This creates a powerful recursive loop where Bayesian structures are continuously updated utilizing real-time simulated outputs:
    ```python
    # After simulation yields a trajectory dataframe `df_sim`
    learned_structure = bn.structure_learning.fit(df_sim, methodtype='hc', scoretype='bic')
    ```

### Step 16: Analysis
In this step, advanced statistical analysis utilizes `bnlearn` specifically for **Causal Inference**:
*   Given evidence from a particular time step slice of the simulation, `bn.inference.fit()` runs exact probabilistic counterfactuals to evaluate agent policy consequences over different time horizons. 

---

## Architectural Advantages for GNN

1.  **Unified CPT Generation**: `bn.generate_cpt()` reduces boilerplate in mapping Active Inference matrix initialization properties to valid Probability Distributions within the validation layer.
2.  **Streamlined Matrix Conversions**: Robust internal mappings between `Vector <-> DAG <-> Adjacency Matrix` remove the fragile `NumPy` transpositions often encountered when cross-loading PyMDP objects.
3.  **Functional Edge Tracking**: Exposes specific edge strengths, preventing "dead" states or isolated actions in an Active Inference generative model from stalling the execution loop undetected.

Through this integration, `bnlearn` effectively acts as both a validation sandbox and an analytical powerhouse, securing the epistemic integrity of the Bayesian systems orchestrated across the 25 GNN framework steps.
