# EXTRACT_PARAMETERS

Based on the GNN specification, here are the key parameters:

1. **Model Matrices**:
   - A matrices representing the hidden states and their corresponding action distributions
   - B matrices representing the transition matrices and their corresponding action distributions
   - D matrices representing the forward and backward updates of the model parameters
   - α (alpha) is a parameter that controls the rate at which the model learns from the data, with lower values leading to slower learning rates

2. **Precision Parameters**:
   - γ: precision parameters for each modality
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