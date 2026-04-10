# EXTRACT_PARAMETERS

Based on the information provided, here are the key parameters for the GNN model:

1. **Model Matrices**:
   - A matrices representing the state space dimensions and their interpretation (e.g., "A" represents a set of states with dimension 2)
   - B matrices representing the transition matrix structure and its interpretation (e.g., "B" represents a set of hidden states, which can be thought of as a sequence of transitions from one state to another).

2. **Precision Parameters**:
   - γ: precision parameters for each factor
   - α: learning rate parameters
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each modality
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor
4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales

5. **Initial Conditions**:
   - Initial parameters:
    - Prior beliefs over initial states
    - Initial parameter values
    - Initialization strategies
    1. **Uniformity**: All parameters are initialized with a uniform prior distribution, which is then updated based on the observed data.
    2. **Fixed Parameters**: All parameters remain fixed at their initial value, and subsequent updates are made using the available information.

6. **Configuration Summary**:
   - Parameter file format recommendations
    - Tunable vs. fixed parameters
    - Sensitivity analysis priorities