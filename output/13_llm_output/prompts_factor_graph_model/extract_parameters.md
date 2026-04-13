# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN model:

1. **Model Matrices**:
   - A matrices representing the matrix representation of the model
   - B matrices representing the learned probabilities and confidence vectors
   - C matrices representing the learned information flow (IF)
   - D matrices representing the learned decision-making rules (DMR)
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