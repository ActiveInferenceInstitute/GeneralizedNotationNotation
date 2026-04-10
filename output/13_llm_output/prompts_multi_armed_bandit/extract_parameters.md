# EXTRACT_PARAMETERS

Based on the document, here's a systematic approach to parameterize GNNs:

1. **Model Matrices**:
   - A matrices representing the model structure and inference processes (e.g., Likelihood Matrix, Transition Matrix)
   - B matrices representing the action-observation mapping (e.g., Policy Vector, Action Vector)
   - C matrices representing the hidden state information (i.e., prior over reward context)

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