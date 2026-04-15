# EXPLAIN_MODEL

This is an example of active inference on to a GNN (Generalized Notation Notation) system with continuous state space and Gaussian belief updating. The model represents the action-based decision making process for a robot agent navigating a 2D environment. It uses Laplace approximation, generalized coordinate matrices, and probabilistic graphical models to update beliefs based on new data.

The core components of this model are:

1. **Input**: A continuous state space (s_f0, s_f1) representing the current state-of-the-world configuration. The input is updated using a linear neural network with Laplace approximation and Gaussian belief updating.

2. **Output**: A continuous action map (A_μ), which represents the predicted next position based on the previous predictions. This prediction is updated using a probabilistic graphical model to update beliefs for actions.

3. **State Transition Matrix**: A matrix representing the joint probability of each state-action pair, with the first state being the input and the second state being the output. The action map represents the predicted next position based on the previous predictions.

4. **Action**: A vector representing a sequence of actions (actions_1, actions_2). These are updated using probabilistic graphical models to update beliefs for actions.

5. **Prediction Matrix**: A matrix representing the joint probability of each state-action pair, with the first state being the input and the second state being the output. The prediction map represents the predicted next position based on the previous predictions.

The model performs active inference by updating its belief using a probabilistic graphical model to update beliefs for actions. This process is repeated until convergence, where all possible actions are updated in the network. The goal of this process is to learn and predict future outcomes based on past decisions made by the agent.

Key insights:

1. **Action**: A vector representing a sequence of actions (actions_1, actions_2). These are updated using probabilistic graphical models to update beliefs for actions.

2. **Prediction Matrix**: A matrix representing the joint probability of each state-action pair, with the first state being the input and the second state being the output. The prediction map represents the predicted next position based on the previous predictions.

The model's goal is to learn and predict future outcomes using active inference principles. This process involves updating beliefs for actions based on new data, learning from past decisions made by the agent, and making predictions about uncertain future states.