# EXTRACT_PARAMETERS

Based on the provided information, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the action-observation mapping (A) and the policy-action matrix (B).
   - B represents the reward distribution over actions, while A is used to represent the rewards across states.
   - C represents the prior probability vector of actions, while D represents the prior probabilities for each state.

2. **Precision Parameters**:
   - γ: precision parameters and their roles in GNN estimation (e.g., learning rate).
   - α: learning rates and adaptation parameters.
   - Other precision/confidence parameters are not provided as parameter values but can be inferred from the context information.
3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales are not provided as parameter values but can be inferred from the context information.
5. **Initial Conditions**:
   - Initial state parameters
   - Initialization strategies for each factor

6. **Configuration Summary**:
   - Parameter file format recommendations:
    - `output/10_ontology_input/simple_mdp_ontology_report.json` and
    `output/10_ontology_input/multi_armed_bandit_ontology_report.json`.
7. **Tunable Parameters**:
   - Sensitivity analysis priorities for each factor:
      - `gamma`: sensitivity analysis prioritizes the choice of gamma parameter value based on the context information.
      - `α`: sensitivity analysis prioritizes the choice of α parameter value based on the context information.
    - `other_precision/confidence` parameters are not provided as parameter values but can be inferred from the context information.
8. **Configuration Summary**:
   - Parameter file format recommendations:
    - `output/10_ontology_input/simple_mdp_ontology_report.json`.