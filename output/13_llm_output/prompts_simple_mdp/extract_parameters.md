# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the MDP agent's state space and action spaces (A)
   - B matrices representing the MDP agent's policy and action distributions over states and actions (B)
   - C matrices representing the MDP agent's prior distribution over initial states and actions (C)

2. **Precision Parameters**:
   - γ: precision parameters for each modality
   - α: learning rates and adaptation parameters
   - Other precision/confidence parameters 

3. **Dimensional Parameters**:
   - State space dimensions for each modality
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales