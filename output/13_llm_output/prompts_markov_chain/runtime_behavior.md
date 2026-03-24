# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

You can use the following code to simulate a simple Markov chain with no action and no observation model:


```python
import random
from collections import deque

def gnn_model(states, actions):
    states = [state for state in states]

    def gNN():
        # Initialize the initial state distribution
        s = StatesRandomState()

        # Initialize the transition matrix
        B = StatesTransitionMatrix([s])

        # Initialize the observation vector
        o = StatesObservationVector([o[0]])

        # Initialize the next state distribution
        s_prime = StatesInitialDistribution(states)

        # Initialize the next state distribution
        s_prime.append((1, 1))

    def gNN_update():
        for _ in range(num_actions):
            # Update the transition matrix
            B[0][s] += (random.uniform(-2*np.pi/6, np.pi/6) * random.uniform(-2*np.pi/6, np.pi/6))

            # Update the observation vector
            o[1][s_prime[0]] = s_prime[1][s] + (random.uniform(0.5, 1)*np.random.normal([0.7, 0.3]))

    def gNN_update_next():
        for _ in range(num_actions):
            # Update the next state distribution
            o[2][s_prime[1]] = s_prime[2][s] + (random.uniform(-2*np.pi/6, np.pi/6) * random.normal([0.7, 0.3]))

    def gNN_update(state):
        # Update the next state distribution
        o[2][state] = s_prime[1][s] + (random.uniform(-2*np.pi/6, np.pi/6) * random.normal([0.7, 0.3]))

    def gNN_update(state):
        # Update the observation vector
        o[2][state] = s_prime[1][s] + (random.uniform(-2*np.pi/6, np.pi/6) * random.normal([0.7, 0.3]))

    def gNN_update(state):
        # Update the next state distribution
        o[2][state] = s_prime[1][s] + (