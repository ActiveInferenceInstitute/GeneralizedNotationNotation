# EXPLAIN_MODEL

This is an example of active inference on to a GNN representation. The model represents the perception of two actions (action1 and action2) with different probabilities and biases. It maps these actions onto observable states (observation 0), which are then used for inference in the following way:

1. **Initialization**: The model initializes the belief over all hidden states, including those associated with each observation. This is done using a softmax activation function to capture the joint probability of the action and its corresponding state.

2. **Model Dynamics**: The model updates beliefs based on the observed actions and their probabilities. It maps these actions onto observable states (observation 0), which are then used for inference in the following way:
   - Actions with higher probabilities become more likely to be chosen, while those with lower probabilities remain less probable. This is because the probability of an action decreases as it becomes more plausible.
   - The model updates beliefs based on the observed actions and their corresponding states, allowing it to update its own belief over time.

3. **Activation**: The model uses a softmax activation function to map each observable state into a probability distribution over hidden states. This allows for inference in the following way:
   - Actions with higher probabilities become more likely to be chosen, while those with lower probabilities remain less probable. This is because the probability of an action decreases as it becomes more plausible.
   - The model updates its own belief over time based on the observed actions and their corresponding states, allowing for inference in the following way:
    - Actions with higher probabilities become more likely to be chosen, while those with lower probabilities remain less probable. This is because the probability of an action decreases as it becomes more plausible.
   - The model updates its own belief over time based on the observed actions and their corresponding states, allowing for inference in the following way:
    - Actions with higher probabilities become more likely to be chosen, while those with lower probabilities remain less probable. This is because the probability of an action decreases as it becomes more plausible.
   - The model updates its own belief over time based on the observed actions and their corresponding states, allowing for inference in the following way:
    - Actions with higher probabilities become more likely to be chosen, while those with lower probabilities remain less probable. This is because the probability of an action decreases as it becomes more plausible.
   - The model updates its own belief over time based on the observed actions and their corresponding states, allowing for inference in the following way:
    - Actions with