# EXTRACT_PARAMETERS

Based on the information provided, here are the key components of the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the model structure and its relationships with other models (e.g., ActInfContinuous)
   - B matrices representing the bias-variance matrix for each action parameter
   - C matrices representing the covariance matrix for each sensory prediction error
2. **Precision Parameters**:
   - γ: precision parameters, roles in the model hierarchy
   
   - α: learning rate and adaptation parameters
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