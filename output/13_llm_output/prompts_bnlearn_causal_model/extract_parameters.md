# EXTRACT_PARAMETERS

Based on the provided specification, here are the key parameters and their corresponding descriptions:

**Model Matrices:**

1. **A matrix representing the hidden state**: A 2D tensor with shape (num_hidden_states,) containing the initial states of the hidden state. The dimensions are `(num_hidden_states)` for each hidden state, which is a list of lists representing the number of hidden states and their corresponding actions.

2. **B matrix representing the action space**: A 1D tensor with shape (num_actions,) containing the initial actions of the action map. The dimensions are `(num_actions)`, where `action` represents the action to be taken in each state, which is a list of lists representing the number of states and their corresponding actions.

3. **C matrix representing the transition probabilities**: A 2D tensor with shape (num_hidden_states,) containing the initial hidden states for each modality. The dimensions are `(num_hidden_states)`, where `action` represents the action to be taken in each state, which is a list of lists representing the number of actions and their corresponding probabilities.

4. **D matrix representing the decision variables**: A 1D tensor with shape (num_actions,) containing the initial states for each modality. The dimensions are `(num_actions)`, where `action` represents the action to be taken in each state, which is a list of lists representing the number of actions and their corresponding probabilities.

**Precision Parameters:**

1. **γ**: A parameter that controls the precision of predictions made by the Bayesian network model. It can range from 0 (no prediction) to 1 (maximum prediction). The default value for γ is `-2`.

2. **α**: A parameter that controls the learning rate of the Bayesian network model. It can range from 0 (constant) to 1 (maximize accuracy). The default value for α is `-0.5` and `0.9`, which are optimal values for learning rates in active inference models.

3. **Other precision/confidence parameters**: These parameters control the sensitivity of predictions made by the Bayesian network model to changes in the input data, such as changes in action probabilities or actions themselves. They can range from 1 (maximum confidence) to infinity (no prediction). The default values for these parameters are `-0.5` and `0`, which are optimal values for learning rates in active