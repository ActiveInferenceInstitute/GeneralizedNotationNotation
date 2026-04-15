# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the ActInfPOMDP model structure and its components (A)
   - B matrices representing the ActInfPOMDP model structure and its components (B)
   - C matrices representing the ActInfPOMDP model structure and its components (C)
   - D matrices representing the ActInfPOMDP model structure and its components (D)

2. **Precision Parameters**:
   - γ: precision parameters for each factor
   - α: learning rates and adaptation parameters
   - Other precision/confidence parameters 

3. **Dimensional Parameters**:
   - State space dimensions for each modality
   - Observation space dimensions for each control factor
   - Action space dimensions for each control factor

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales