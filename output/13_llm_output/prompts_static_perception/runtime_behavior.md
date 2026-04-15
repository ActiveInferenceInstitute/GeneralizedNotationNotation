# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

```python
import numpy as np
from typing import List, Dict

class GNNModel:
    """GNN Representation Model."""

    def __init__(
        self,
        **kwargs: dict,
        model_annotations=None,
        num_hidden_states=2,
        num_obs=2,
        activation="relu",
        softmax_activation=False,
        prior_probability=0.95,
        prior_weight=np.array([1]),
    ):

        """
        Initialize the GNN Representation Model with given parameters and annotations.

        Args:
            **kwargs: Key-value arguments for the model annotation dictionary.

            - `model_annotations`: A list of dictionaries representing the input data.
                Each dictionary represents a single observation, where each key is an
                observation name (e.g., "observation" or "hidden state") and each value
                is a dictionary containing the corresponding hidden states.
                The values are lists of 2-dimensional arrays representing the probability
                of observing the given observation in that hidden state.

            - `num_hidden_states`: A list of integers representing the number of
                hidden states for this model.
            - `num_obs`: A list of integers representing the number of observations
                for which we are interested in learning a belief from the data.
            - `activation`: A function that takes an observation and returns its probability,
                or None if no action is taken (default).
            - `softmax_activation`: A function that takes an observation and returns its softmax
                value as output.
            - `prior_probability` (optional): A boolean indicating whether to use prior probabilities for
                the activation functions. Defaults to False.
            - `prior_weight`: A list of numbers representing the weight assigned to each
                activation in the model.
            - `activation_type`: A string specifying which activation type is used: "relu",
                "sigmoid" or "exponential".
            - `activation_name` (optional): A name for the activation function, e.g., "ReLU".
            - `activation_weight` (optional): A list of numbers representing the weight assigned to
                each activation in the model.
            - `activation_type`: A string specifying which activation type is used: "sigmoid",
                "exponential" or "relu".
            - `activation_name` (optional): A name for the activation function, e.g., "ReLU".
        """

        self._model = model