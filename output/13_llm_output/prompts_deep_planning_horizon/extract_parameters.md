# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the model structure and its relationships with other models (e.g., ActInfPOMDP)
   - B matrices representing the action sequences and their dependencies over actions (e.g., GNN-based Actions)
   - C matrices representing the policy distributions and their dependence on actions (e.g., GNN-based Policies)

2. **Precision Parameters**:
   - γ: precision parameters for each modality
   - α: learning rates and adaptation parameters
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each modality
   - Observation space dimensions for each control factor
   - Action space dimensions for each action variable (e.g., Actions)

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