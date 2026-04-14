# EXTRACT_PARAMETERS

Here is a systematic parameter breakdown of the GNN model:

1. **Model Matrices**:
   - A matrices: dimensions, structure, interpretation
   - B matrices: dimensions, structure, interpretation
   - C matrices: dimensions, structure, interpretation

   The matrix M(x) represents the mapping from states to actions and transitions between states. The matrix M is used for inference on the model parameters.

2. **Precision Parameters**:
   - γ (gamma): precision parameters and their roles
   - α (alpha): learning rates and adaptation parameters
   - Other precision/confidence parameters

   These are used as input arguments to the model, which can be tuned using a tuning strategy. The parameter file format recommendations provide guidance on how to tune these parameters based on specific training data or evaluation metrics.

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality

   These are used as input arguments to the model, which can be tuned using a tuning strategy. The parameter file format recommendations provide guidance on how to tune these parameters based on specific training data or evaluation metrics.

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales

   These are used as input arguments to the model, which can be tuned using a tuning strategy. The parameter file format recommendations provide guidance on how to tune these parameters based on specific training data or evaluation metrics.

5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies

   These are used as input arguments to the model, which can be tuned using a tuning strategy. The parameter file format recommendations provide guidance on how to tune these parameters based on specific training data or evaluation metrics.

6. **Configuration Summary**:
   - Parameter file format recommendations
   - Tunable vs. fixed parameters
   - Sensitivity analysis priorities