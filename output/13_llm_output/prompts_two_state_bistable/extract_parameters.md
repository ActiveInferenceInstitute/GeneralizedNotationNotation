# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the model structure and interpretation of the input data.
   - B matrices representing the action-dependent transitions between states.
   - C matrices representing the policy updates over actions.
   - D matrices representing the habit biases in the hidden state space.
   - E matrices representing the expected free energy per policy across different actions.

2. **Precision Parameters**:
   - γ (gamma): precision parameters and their roles.
   - α (alpha): learning rates and adaptation parameters.
   - Other precision/confidence parameters, such as δ, ε, and β are not specified but can be inferred based on the document.

3. **Dimensional Parameters**:
   - State space dimensions for each factor:
      - A matrices representing the model structure and interpretation of the input data.
      - B matrices representing the action-dependent transitions between states.
      - C matrices representing the policy updates over actions.
      - D matrices representing the habit biases in the hidden state space.

4. **Temporal Parameters**:
   - Time horizons (t): The time horizon for each parameter, which can be inferred from the document.
   - Temporal dependencies and windows: The temporal dependencies between different parameters are not specified but can be inferred based on the document.
   - Update frequencies and timescales: The update frequency is set to 10% of the total number of states (T=2), while the update time is set to 5 seconds for each parameter.

5. **Initial Conditions**:
   - Initial parameters:
      - γ = 0.9
      
    - α = 0.8
      
    - Other initial conditions are not specified but can be inferred based on the document.

The document provides a systematic parameter breakdown, with the following key information for each parameter:

1. **Model Matrices**:
   - A matrices representing the model structure and interpretation of the input data.
   - B matrices representing the action-dependent transitions between states.
   - C matrices representing the policy updates over actions.
   - D matrices representing the habit biases in the hidden state space.
   - E matrices representing the expected free energy per policy across different actions.

2. **Precision Parameters**:
   - γ = 0.9
      
    - α = 0.8
      
    - Other precision/confidence parameters, such as δ, ε, and β are not specified but can be