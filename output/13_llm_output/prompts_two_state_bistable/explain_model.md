# EXPLAIN_MODEL

This GNN implementation represents a minimal 2x2x2 POMDP with no bias towards one side of the action space and no habit biases. It performs Bayesian inference on the observed data to update beliefs about the actions taken by the agent. The model is composed of two main components:

1. **hidden states**: "left" and "right". These represent the current state-of-the-world (SOTW) for each observation.

2. **actions**: Push_left/Push_right, which are actions that move from one side to the other.

The model updates its beliefs based on a Bayesian inference of the observed data and action sequences. The policy is chosen based on the current state-of-the-world (SOTW) for each observation. The agent prefers observation 1 ("left") over observation 0 ("right").

1. **Initialization**: The model initializes the hidden states with a random value, which represents the current state of the POMDP.

2. **Action selection**: The policy is chosen based on the current SOTW for each observation. This action sequence is then applied to all observations in order to update their beliefs about the actions taken by the agent.

The model performs Bayesian inference and updates its beliefs using a probabilistic graphical model (PGM). The POMDP consists of two main components:

1. **hidden states**: "left" and "right". These represent the current state-of-the-world (SOTW) for each observation.

2. **actions**: Push_left/Push_right, which are actions that move from one side to the other.

The model updates its beliefs based on a Bayesian inference of the observed data and action sequences. The policy is chosen based on the current SOTW for each observation in order to update their beliefs about the actions taken by the agent.

Key relationships:

1. **Initialization**: The hidden states are initialized with random values, representing the current state-of-the-world (SOTW).

2. **Action selection**: The policy is chosen based on the current SOTW for each observation in order to update their beliefs about the actions taken by the agent.

Practical implications:

1. **Bayesian inference**: The model performs Bayesian inference and updates its beliefs using a probabilistic graphical model (PGM). This allows the model to learn from data, making predictions based on past behavior.

