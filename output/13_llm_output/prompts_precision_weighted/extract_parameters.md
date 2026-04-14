# EXTRACT_PARAMETERS

Based on the provided information, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the model matrix and its structure (e.g., Lambda, Policy, EFE).
   - B matrices representing the transition matrix and action vector representations of actions.
   - C matrices representing the habit distribution over observations.
   - D matrices representing the habit distribution over actions.

2. **Precision Parameters**:
   - γ: learning rate parameter for each modality (e.g., α, β)
   - α: learning rate parameters for each modality (e.g., δ)
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions of the model matrices and their interpretation (dimensions, structure).
   - Observation space dimensions of the state matrix and its interpretation (dimensionality).
   - Action space dimensions of the action vector representation of actions (dimensionality).
   - Action space dimensionality for each modality

4. **Temporal Parameters**:
   - Time horizons (t)
   - Temporal dependencies and windows
   - Update frequencies and timescales

5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies

6. **Configuration Summary**:
   - Parameter file format recommendations for each model matrix, state space matrices, action vectors, and observation spaces (dimensions).