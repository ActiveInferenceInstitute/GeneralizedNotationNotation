# EXTRACT_PARAMETERS

Based on the information provided, here are the key parameters for the GNN model:

1. **Model Matrices**:
   - A matrices representing the state space and action spaces of the agent
   - B matrices representing the belief matrix and prediction matrix
   - C matrices representing the action-based predictions and confidence matrices

2. **Precision Parameters**:
   - γ (gamma): precision parameters for each modality
   - α (alpha): learning rate and adaptation parameters
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions: 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
   - Observation space dimensions: 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
   - Action space dimensions: 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2

4. **Temporal Parameters**:
   - Time horizons (t): time horizon for each modality
   - Temporal dependencies and windows: window parameters for each modality
   - Update frequencies and timescales: update frequency and timestamps for each modality
5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies

6. **Configuration Summary**:
   - Parameter file format recommendations
   - Tunable vs. fixed parameters
   - Sensitivity analysis priorities