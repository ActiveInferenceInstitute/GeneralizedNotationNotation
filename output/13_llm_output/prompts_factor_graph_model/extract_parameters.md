# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN model:

1. **Model Matrices**:
   - A matrices with dimensions of 3x2 (Visual modality) and 4x2 (Proprioceptive modality). The matrix represents the joint probability distribution over the visual modalities and proprioceptive modalities.
   - B matrices representing the joint probabilities for each modality, which are used to compute the likelihood factor for each modality.

2. **Precision Parameters**:
   - γ: precision parameters and their roles (e.g., learning rate, adaptation parameter).
   - α: learning rates and adaptation parameters (see Section 3.4)

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor
   - Temporal dependencies and windows
   - Update frequencies and timescales

4. **Initial Conditions**:
   - Prior beliefs over initial states (e.g., visual modality)
   - Initial parameter values
   - Initialization strategies (see Section 3.5)

5. **Configuration Summary**:
   - Parameter file format recommendations for each parameter:
    - `model_parameters` contains the list of parameters associated with each model type, 
    including initial conditions and initialization strategies.
    - `initial_conditions` is a dictionary containing all initializations that can be used to initialize the model.
    - `parameter_file` contains a list of parameters for each parameter type (e.g., `visual`, `proprioceptive`) associated with each parameter type.

6. **Tunable Parameters**:
   - Sensitivity analysis priorities:
    - For Visual modality, sensitivity is set to 0.1 and 25% of the total variance in the model parameters for each modality (see Section 3.4).
    - For Proprioceptive modality, sensitivity is set to 0.1 and 25% of the total variance in the model parameters for each modality (see Section 3.6).

7. **Temporal Parameters**:
   - Time horizons (t) are specified as a list of timesteps (e.g., `time`)
    - Temporal dependencies and windows are set to a list of window sizes (`window_size`), which can be used for updating the model parameters based on temporal dependencies, such as when the modality is in view or outview.