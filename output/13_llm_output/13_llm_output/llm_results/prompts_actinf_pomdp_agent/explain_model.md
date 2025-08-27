# EXPLAIN_MODEL

======================================================
Summary:

Here is a complete overview of the ModelPurpose section of the [GNN](https://github.com/Hu-KaiJung/model_purpose) document:

1. **Model Purpose**: This description covers what type of action inference agent does, how it represents real world phenomena or problems, and its core components (hidden state & actions). It explains that this action inference model is based on Variational Free Energy (VFE), Expected Free Energy (FFE) estimation, and Belief updating.

2. **Core Components**: This section summarizes the key parts of the `ActiveInferencePomdp` graph algebra module:
   - The `v1_paths` tensor represents states from which actions are obtained by taking steps forward through state space.
   - The `h(t)` tensor captures the history of observed data points $(x_{t-1}, x_{t+1})$ and its corresponding beliefs $(p_{i, t}), where $p_{i}$ is defined as $b_0 + \frac{a_{s}_{i}(f)}{B[n]^2} < 0$, the action sequence (action(x_{t-1}, x_{t+1})), and its history.
   - The `h(x)` tensor represents the expected actions from which observed data points $(x_t, a_{s}_{i}(f))$ are derived by taking steps forward through state space.
   - `π` is an element of the hidden state (state observation) matrix represented as the output of `inference()` action operator for each observation at time $t$. It represents the probability distribution over observations $(x_0, a_{s}_{i}(f))$, where $a_{s}$ represent the prior distributions over actions.

3. **Model Dynamics**: This section provides an overview of how the model evolves across timestep and action iteration:
   - Actions are updated based on the learned beliefs at each state observation $(x_t, a_{s}_{i}(f))$. The learned belief distribution ([π]) is used as a prior for actions calculated by `inference()`.
   - The learning process continues until there exist no new beliefs to update. This enables estimation of all the observed observations and their corresponding actions at each time step.
   - Actions are updated based on the probabilities computed from previous states $(x_t, b0 + \frac{a_{s}_{i}(f)}{B[n]^2} < 0$ for $(x_1, a_{c}^{-1}, f)). This process is repeated until there exist no new beliefs to update.

4. **Active Inference Context**: This section provides an overview of how the model implements active inference principles:
   - The `v1_paths` tensor represents states from which actions are obtained by taking steps forward through state space. It captures the history of observed data points $(x_{t-1}, a_{s}_{i}(f))$, where $a_{s}$ represent the prior distributions over actions (see above).
   - The `h(x)` tensor represents the expected actions from which observed data points $(x_0, a_{c}^{-1}, f)), where $a_{s}$ represent the prior distribution over actions. It captures the history of actions tracked in the `v1_path` graph algebra module and its corresponding beliefs $(p_{i=o}^*). This allows applying policies to sequences of observations $(x, a)$.
   - The `h(t)` tensor represents the expected actions from which observed data points $(y_0, b^{-1}(f))$, where $b^{-1}$ represent the prior distributions over actions. It captures the history of actions tracked in the `h` graph algebra module and its corresponding beliefs $(p_{i=o}^*). This allows applying policies to sequences of observations $(x)$.