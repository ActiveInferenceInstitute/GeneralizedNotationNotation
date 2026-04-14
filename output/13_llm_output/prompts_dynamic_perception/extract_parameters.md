# EXTRACT_PARAMETERS

Based on the information provided, here are the key parameters for the GNN model:

1. **Model Matrices**:
   - A matrices representing the state space dimensions and the corresponding probabilities of each observation (state inference) over time.
   - B matrices representing the transition matrix and the prior belief distribution over states.
   - D matrices representing the prior distributions over hidden states, observed states, and action parameters.

2. **Precision Parameters**:
   - γ: precision parameter for each modality
   - α: learning rate for each modality (default is 0.1)
   - Other precision/confidence parameters
3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

4. **Temporal Parameters**:
   - Time horizons (t): time horizon over which the model is trained
   - Temporal dependencies and windows: window parameters to allow for updating of parameter values based on temporal dependencies
5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies

6. **Configuration Summary**: A tabular representation of the model parameters, including their roles, priorities, and sensitivity analysis.