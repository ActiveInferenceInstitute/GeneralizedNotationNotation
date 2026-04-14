# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the model structure and interpretation of the input data (policy space)
   - B matrices representing the action sequences and their probabilities
   - C matrices representing the policy distributions and their probability density functions
   - D matrices representing the decision boundaries, actions, and control variables
2. **Precision Parameters**:
   - γ: precision parameters for each factor
   - α: learning rate parameters
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each modality (action sequence)
   - Observation space dimensions for each action variable
   - Action space dimensions for each control variable
4. **Temporal Parameters**:
   - Time horizons for each factor
   - Temporal dependencies and windows for each modality
   - Update frequencies and timescales for each parameter

5. **Initial Conditions**:
   - Prior beliefs over initial states (policy sequence)
   - Initial parameters values (actions, actions sequences, etc.)
6. **Configuration Summary**:
   - Parameter file format recommendations
   - Tunable vs. fixed parameters
   - Sensitivity analysis priorities