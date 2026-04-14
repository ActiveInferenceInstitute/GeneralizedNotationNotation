# EXTRACT_PARAMETERS

Based on the document, here are the key parameters:

1. **Model Matrices**:
   - A matrices representing the model structure and its components (actions, states, etc.)
   - B matrices representing the state space dimensions for each action/action pair
   - C matrices representing the observation spaces for each modality
   - D matrices representing the action-observation relationships
2. **Precision Parameters**:
   - α: learning rate parameters
   - γ: precision parameter
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor
   - Initialization strategies (e.g., using a specific number of initial states, actions)
4. **Temporal Parameters**:
   - Time horizons (t): time horizon parameters
   - Temporal dependencies and windows: window parameters
   - Update frequencies and timescales

5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies (e.g., using a specific number of initial states, actions)
6. **Configuration Summary**: A structured document that summarizes the model structure, parameters, and configuration details for each parameter.