# EXTRACT_PARAMETERS

Based on the information provided, here are the key parameters of the GNN model:

1. **Model Matrices**:
   - A matrices representing the input data (state space) and output predictions for each action parameter.
   - B matrices representing the biases/weights in the neural network architecture.
   - C matrices representing the conditional probabilities of actions based on their respective states.
   - D matrices representing the conditional probabilities of actions based on their corresponding states.

2. **Precision Parameters**:
   - α (alpha): learning rate and adaptation parameters for each action parameter.
   - Other precision/confidence parameters:
   - Sensitivity analysis priorities, such as sensitivity to initial conditions or prior beliefs over initial state space dimensions.

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales

5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies, such as sensitivity to initial conditions or prior beliefs of actions/actions-based biases.