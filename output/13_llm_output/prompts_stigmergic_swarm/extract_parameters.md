# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN model:

1. **Model Matrices**:
   - A matrices representing the state space dimensions and dimensionality of each agent's actions (represented as vectors)
   - B matrices representing the action-belief matrix representation of each agent (represented as vectors)
   - C matrices representing the action-belief matrix representation of each agent (represented as vectors)

2. **Precision Parameters**:
   - γ: precision parameters and their roles
   - α: learning rates and adaptation parameters
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

4. **Temporal Parameters**:
   - Time horizons (t)
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