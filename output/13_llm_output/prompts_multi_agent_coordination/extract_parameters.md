# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the model structure and representation of each agent's belief space (A)
   - B matrices representing the action-belief matrix representations of each agent (B)
   - C matrices representing the policy vector representations of each agent (C)
   - D matrices representing the action-policy matrix representations of each agent (D)

2. **Precision Parameters**:
   - γ: precision parameters and their roles
   - α: learning rates and adaptation parameters
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
   - Tunable vs. fixed parameters
   - Sensitivity analysis priorities