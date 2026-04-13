# EXTRACT_PARAMETERS

Based on the information provided, here are the key parameters for the Active Inference POMDP agent:

1. **Model Matrices**:
   - A matrices representing the model structure and interpretation of the input data (observation space dimensions) and action selection policies (policy distributions).
   - B matrices representing the policy prior distribution over actions and habit preferences, respectively.
   - C matrices representing the initial policy prior distribution over actions and habit preferences, respectively.

2. **Precision Parameters**:
   - γ: precision parameters for each factor
   - α: learning rates and adaptation parameters
   - Other precision/confidence parameters (optional)
3. **Dimensional Parameters**:
   - State space dimensions for each modality
   - Observation space dimensions for each action dimension
   - Action space dimensions for each control variable

4. **Temporal Parameters**:
   - Time horizons (t): number of simulation timesteps for all frameworks and their roles in the parameter file format recommendations
5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies (optional)