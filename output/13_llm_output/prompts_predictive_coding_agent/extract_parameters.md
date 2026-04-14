# EXTRACT_PARAMETERS

Based on the provided information, here are some key parameters that can be used to analyze GNN models:

1. **Model Matrices**:
   - A matrices representing the model's input and output data (e.g., `model_matrices`).
   - B matrices representing the predicted values of each input variable (`state_dim`) and corresponding predictions (`action`, etc.).
   - C matrices representing the prediction errors for each modality (`bias`, etc.)

2. **Precision Parameters**:
   - γ parameter: defines how to update the model's accuracy based on the current state space dimensions, while controlling the learning rate of the predicted values (e.g., `learning_rate`).
   - α parameter: controls the learning rate for each modality (`bias`) and its interpretation in terms of a specific loss function or other parameters.

3. **Dimensional Parameters**:
   - State space dimensions for each input variable (`state_dim`, etc.).
   - Observation space dimensions for each modality (`obs_dim`).
   - Action space dimensions for each control factor (`action`, etc.)

4. **Temporal Parameters**:
   - Time horizons (t)
   - Temporal dependencies and windows
   - Update frequencies and timescales

5. **Initial Conditions**:
   - Prior beliefs over initial states (`prior`)
   - Initial parameter values (`bias`), `bias_x`, etc. for each input variable (`action`, etc.)

These parameters can be used to:
- Control the learning rate of predictions based on current state space dimensions and biases (e.g., `learning_rate`)
- Control the learning rate of prediction errors based on current state space dimensions and bias values (e.g., `learning_rate`)
- Control the learning rate for each modality (`bias`, etc.) to improve accuracy in predictions
- Control the learning rate for actions based on current state space dimensions and biases
- Control the learning rate for predictions based on current state space dimensions and biases