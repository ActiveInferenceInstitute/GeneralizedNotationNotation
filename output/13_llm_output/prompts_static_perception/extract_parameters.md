# EXTRACT_PARAMETERS

Based on the information provided, here is a systematic parameter breakdown of the Active Inference framework:

1. **Model Matrices**:
   - A matrices representing the model structure and inference logic (e.g., GNNs)
   - B matrices representing the model parameters (represented as `A`)
   - C matrices representing the model parameters (`D`, `S`, etc.)
2. **Precision Parameters**:
   - γ: precision parameter for each factor
   - α: learning rate and adaptation parameters
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
   - Initial parameters (`A`)
   - Initialization strategies (e.g., using a fixed parameter, tuning the model to fit into a specific configuration)
6. **Configuration Summary**:
   - Parameter file format recommendations
   - Tunable vs. fixed parameters
   - Sensitivity analysis priorities