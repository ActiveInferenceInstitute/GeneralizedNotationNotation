# EXTRACT_PARAMETERS

Based on the specifications, here is a systematic parameter breakdown of the GNN model:

1. **Model Matrices**:
   - A matrices: dimensions, structure, interpretation
   - B matrices: dimensions, structure, interpretation
   - C matrices: dimensions, structure, interpretation

   These matrices represent the active inference agent's state and observation space. The matrix for each modality (action) is initialized with a random value based on its properties.

2. **Precision Parameters**:
   - α (alpha): learning rates and adaptation parameters
   - Other precision/confidence parameters:
    - γ (gamma): learning rate, adaptation parameter, and sensitivity analysis priorities

   These parameters represent the action-specific biases and uncertainties in the model. The choice of α depends on the specific actions being learned from the data.

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

The parameter file format recommendations are:

  1. **Initialization strategies**
    - Random initialization of parameters based on initializations (e.g., random choice of α, other precision/confidence parameters)

    This is a good approach when the initial values are not well-defined or have different interpretations.

2. **Tunable vs. fixed parameters**:
   - Tuning parameter choices:
    - Random initialization of parameters based on initializations (e.g., random choice of α, other precision/confidence parameters)

    This is a good approach when the initial values are not well-defined or have different interpretations.

3. **Sensitivity analysis priorities**
    - Sensitivity analysis prioritization for each parameter:
      - Random initialization of parameters based on initializations (e.g., random choice of α, other precision/confidence parameters)

    This is a good approach when the initial values are not well-defined or have different interpretations.

4. **Tunable vs. fixed parameters**:
    - Tuning parameter choices:
      - Random initialization of parameters based on initializations (e.g., random choice of α, other precision/confidence parameters)

    This is a good approach when the initial values are not well-defined or have different interpretations.

Overall, these recommendations provide a structured and systematic way to analyze GNN models with active inference agents.