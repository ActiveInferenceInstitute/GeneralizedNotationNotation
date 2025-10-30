# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

You're on the right track! Here's a step-by-step guide to implement the GNN model:

1. Define your GNN model using `SmolLM`:
```python
model = SmolLM(
    model_name='GNN',
    num_hidden_states=3,
    num_actions=3,
    model_parameters=[
        {
            'lhs': {'state': 10},
            'action'=(0.95) * (10 - 0.02),
            'next state': [
                [
                    # Define the initial observation and policy
                   ('', [], True)
              ],
             ...
             ]
       )
    ],
    model_fn=lambda x: ([
        {
            'observation': ({'x': []}, [{'action': {'state': 10}], {'hits': [{'obstinations': {}}}]})
    ],
    
    # Define the action probabilities for each observation.
    actions=[
        {
          'observations': ['', [], True'],
          'actions'=(0.9) * (3 - 0.8),
          'probability_state': [
            {'observation': [{'x': []}, {'obstinations': {}}}],
            # Define the action probabilities for each observation, as a list of tuples
        ],
    ]
)

2. Create your GNN representations using `SmolLM`:
```python
graph = SmolLM(model_name='GNN', num_hidden_states=3, model_parameters=[
    {
        'lhs': {'state': 10},
        'action'=(0.95) * (10 - 0.8),
        'next state': [
            [
                # Define the initial observation and policy
                   ('', [], True)
              ],
             ...
             ]
       )
    ],
]

3. Initialize a GNN agent with all actions:
```python
agents = [(A, {}) for A in model_parameters['actions'] if A else [])
# Use a list of actions (available from action=dict()) as the default policy and initialize the state transition matrix with zeros if it has no policies.
for action in A[1]:
    for state in B:
        agents[[action]].append(graph)

4. Initialize the belief distribution using the action probability:
```python
belief = {**B}  # Use a list of actions to initialize a belief (available from action=dict()) as the default policy and initializing the next observation with zero if it has no policies or there are no actions for that observation, use an empty collection as the new policy.
# Initialize the beliefs distribution across all observations and then initialize a belief distribution on each observation in each of those observations.
for action in A[1]:
    for state, b in B:
        # Use the default view from belief distributions (available from actions=dict()) to update it.
        belief = {**B}  # Use a list of actions to initialize a belief (available from action=dict()) as the default policy and initializing the next observation with zero if it has no policies or there are no actions for that observation, use an empty collection as the new policy.
```

5. Initialize all observed beliefs using an observable distribution:
```python
observations = {**B}  # Use a list of observations to initialize a belief (available from action=dict()) and then initialize it with zero if there are no policies or there is nothing else for that observation, use an empty collection as the new policy.
# Initialize the beliefs distribution across all observations in each of those observations by using the actions distribution as the default policy.
```

6. Iterate over all observable states, and iterate over all observed observables:
```python
observation_observations = {**B}  # Use a list to initialize the next observation for that observation (available from action=dict()) in each of those observations (can be empty since there is nothing else for that observation)
for i, observable in enumerate(observation_observations):
    observation = {*observations.copy() if observable == 'observation' else observation}  # Initialize the next observation with zero based on actions distribution and state distribution (available from action=dict())

    beliefs = {**B}  # Use a list to initialize the next observed belief for that observation in each of those observations (can be empty since there is nothing else for that observation)
    if observable == 'observation':
        beliefs[observations['x'][i]].append(belief)
    else:
        actions.clear()
```

7. Initialize all other observed beliefs and the last hidden states using an observable distribution from the previous step:
```python
observer_observations = {**B}  # Use a list to initialize the next observation for that observation (available from action=dict()) in each of those observations, and then update belief distributions across actions with zero if there are no policies or there is nothing else for that observation, use an empty collection as the new policy.
# Initialize the beliefs distribution across all observations in each of those observables to initialize a belief distribution on every observation within that observed observable by using the actions distribution from step 2 (available from action=dict()) and then updating the last hidden state distribution accordingly after initialization.
```