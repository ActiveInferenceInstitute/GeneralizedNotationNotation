# EXTRACT_PARAMETERS

Based on the provided information, here are the key parameters for Active Inference (AI) on a GNN model:

1. **Model Matrices**:
   - A matrices representing the input data and corresponding predictions/beliefs of the model
   - B matrices representing the prior beliefs over hidden states
   - C matrices representing the action probabilities and their joint distribution across actions
   - D matrices representing the uncertainty parameters for each modality (e.g., prediction accuracy)

2. **Precision Parameters**:
   - γ: precision parameter, which controls the rate at which predictions are made based on the input data
   - α: learning rate parameter, controlling the rate of updates in the model's beliefs and predictions
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

Here are some summary statistics on these parameters:
- **Initial Conditions**:
  - Initial belief states (B)
  - Initial prediction accuracy (P)
  - Initialization strategy (Sensitivity analysis):
    - Random initialization of initial beliefs and predictions.
    - Random initialization of action probabilities and their joint distribution across actions.
    - Random initialization of uncertainty parameters for each modality based on the input data.
- **Configuration Summary**:
  - Parameter file format recommendations:
    - "input_data" (Input Data) specifies the dataset to be used as input to the model
    - "action_probability" and "prediction_accuracy" specify the action probabilities and accuracy, respectively
    - "observation_space" is a dictionary containing all possible actions in the observation space
  - "initialization_strategy" parameter:
    - "random initialization of initial beliefs and predictions",
    - "regularization_factor": 0.1 for regularization (default),
    - "learning_rate": 0.2,
    - "action_probability_distribution": dict with action probabilities as keys and their joint distribution across actions as values
  - "sensitivity_analysis" parameter:
    - "random initialization of initial beliefs",
    - "regularization_factor": 1 for regularization (default),
    - "learning_rate": 0.2,
    - "action_probability_distribution": dict with action probabilities as keys and their joint distribution across actions as values
  - "sensitivity_analysis" parameter:
    - "random initialization of initial beliefs",
    - "regularization_factor": 1 for regularization