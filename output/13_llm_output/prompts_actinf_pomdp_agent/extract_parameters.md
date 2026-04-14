# EXTRACT_PARAMETERS

Based on the information provided, here are the key parameters for the Active Inference POMDP Agent:

1. **Model Matrices**:
   - A matrices representing the model structure and interpretation of the agent's actions and beliefs.
   - B matrices representing the action probabilities over observed states and hidden states.
   - C matrices representing the policy prior distribution, preferences, and habit distributions for each observation modality.
   - D matrices representing the initial belief distribution and action selection from policy posterior.

2. **Precision Parameters**:
   - γ (gamma): precision parameters and their roles.
   - α (alpha): learning rates and adaptation parameters.
   - Other precision/confidence parameters:
   - δ (delta) : sensitivity analysis priorities, including sensitivity to the choice of parameter values for each action.