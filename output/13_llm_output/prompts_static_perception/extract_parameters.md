# EXTRACT_PARAMETERS

Based on the GNN specification, here are the key parameters:

1. **Model Matrices**:
   - A matrices representing the state space dimensions for each factor
   - B matrices representing the prior belief over hidden states
   - C matrices representing the initial beliefs and their interpretation
   - D matrices representing the initial parameter values
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
   - Initial parameter values
   - Initialization strategies

6. **Configuration Summary**:
   - Parameter file format recommendations
   - Tunable vs. fixed parameters
   - Sensitivity analysis priorities