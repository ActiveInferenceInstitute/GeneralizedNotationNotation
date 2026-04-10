# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **ModelMatrices**:
   - A matrices representing the model structure and its properties (e.g., number of hidden states, number of actions)
   - B matrices representing the model's components (e.g., type of action, type of state)
   - C matrices representing the model's components (e.g., type of observation, type of action)

2. **Precision Parameters**:
   - γ: precision parameters and their roles
   - α: learning rates and adaptation parameters
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each modality
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