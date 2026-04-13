# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN model:

1. **Model Matrices**:
   - A matrices representing the hidden states and actions of the agent
   - B matrices representing the transition matrix and prior over initial states
   - C matrices representing the habit vector and prior over action selection
   - D matrices representing the habit vector and prior over action selection
   - E matrices representing the action selection parameters

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
   - Tunable vs. fixed parameters
   - Sensitivity analysis priorities