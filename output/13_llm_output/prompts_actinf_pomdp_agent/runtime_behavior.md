# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

You can validate this model and understand the behavior by validating its structure:

1. Validate that it has been trained on the whole domain space. For example, a simple validation script with `torchvision` will work (see doc) to verify that all supported values are represented in the training data set.

2. Verify the activation functions used for the states and actions. They must be able to map input tensor sequences to output tensors representing the same state and action. Check if they can handle inputs from different channels or have some type of constraint imposed upon them, such as a "weight" assigned by the training data set to each dimension in the input sequence prior to activation.

3. Validate that all the activations are correctly applied and that the parameters (Likelihood Matrix, Action Vector) do not suffer from loss when overfitting or collapsing into sub-populations of similar values if an "unbounded horizon" is observed at one time step in a sequence being trained with multiple data sets.

4. Validate that no activation has been over-specified and/or collapsed to zero, even though the model outputs all actions correctly as inputs (because they will be reweighted) - this indicates loss of "learning". You may want to perform more advanced validation checks such as checking for continuity across input sequences or evaluating if there is an observable bias that affects certain activation values.

5. Validate that the reward/penalty distribution has been trained on all data sets and not biased towards specific actions (although this will depend on which data set you are using) - it cannot be a uniform distribution between actions in each iteration step because it would lead to an overfitting of the training data to those actions, instead fitting values directly within the policy distribution.

6. Validate that the agent does not have loss while learning for every possible action when all inputs are pasted into outputs from prior sequences - unlike `torchvision`, you cannot use `torchvision` as a validation script because it is unable to learn and predict actions/actions based on predictions made by other data sets, even though all input sequences contain the same inputs.

7. Validate that there are no bias introduced due to overfitting (although this will depend on which data set you are using) - unlike `torchvision`, you cannot use `torchvision` as a validation script because it is unable to learn and predict actions/actions based on predictions made by other data sets, even though all input sequences contain the same inputs.
```python
  # TODO: Validate this line of code

    x = torch.randn(num_hidden_states)
    x[x < 0] -= delta * num_actions[:, i][:, 0].data
    
    ys = x + [action for action in action_permutations if action == 1 and action!=2][:i+len('outputs')]

    x, zt = torch.from_iterable(ys)
    ys = x[yts] - (zx**3/(zw**4)) * num_actions[:, i].data


```