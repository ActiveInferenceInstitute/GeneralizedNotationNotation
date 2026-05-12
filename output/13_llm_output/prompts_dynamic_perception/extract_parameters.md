# EXTRACT_PARAMETERS

Based on the provided information, here are the key parameters for active inference:

1. **Model Matrices**:
   - A matrices representing the model structure and prior beliefs over initial states
   - B matrices representing the learned belief representations of the hidden state
   - D matrices representing the learned belief representations of the observed states
   - C matrices representing the learned belief representations of the action-observation pairs
2. **Precision Parameters**:
   - γ (gamma): precision parameters and their roles
   - α (alpha): learning rates and adaptation parameters
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor
4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales

5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies

6. **Configuration Summary**:
   - Parameter file format recommendations

Here are the key parameters for each type of parameter:
- **Model Matrices**:
   - A matrices representing the model structure and prior beliefs over initial states
   - B matrices representing the learned belief representations of the hidden state
   - D matrices representing the learned belief representations of the observed states
   - C matrices representing the learned belief representations of the action-observation pairs

1. **Parameter File Format Recommendations**:
- **Initial Conditions**:
    - Initial parameters for each type of parameter:
      - γ (gamma): precision parameters and their roles
      - α (alpha): learning rates and adaptation parameters
      - Other precision/confidence parameters

    Example: `initial_paramfile` should contain the following file structure:

      ```json
  "input"
   "model.py":
    "input/10_ontology_output/simple_mdp_ontology_report.json",
    "input/10_ontology_output/multi_armed_bandit_ontology_report.json",
    "input/10_ontology_output/actinf_pomdp_agent_ontology_report.json",
    "input/10_ontology_output/hmm_baseline_ontology_report.json"
  "outputs":
    "output/10_ontology_output/simple_mdp_ontology_report.json",
    "output/10_ontology_output/actinf_pomdp_agent