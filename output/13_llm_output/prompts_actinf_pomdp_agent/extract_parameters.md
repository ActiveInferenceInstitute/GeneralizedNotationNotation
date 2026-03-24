# EXTRACT_PARAMETERS

Based on the information provided, here are the key parameters for the Active Inference POMDP agent:

1. **Model Matrices**:
   - A matrices representing the model's structure and interpretation of the input data (state space)
   - B matrices representing the policy prior distribution over actions
   - C matrices representing the habit prior distribution over actions
   - D matrices representing the action selection from policy posterior
2. **Precision Parameters**:
   - γ: precision parameters for each factor
   - α: learning rates and adaptation parameters

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