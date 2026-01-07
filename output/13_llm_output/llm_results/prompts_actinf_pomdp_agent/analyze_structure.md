# ANALYZE_STRUCTURE

Here is a detailed analysis of the model implementation for the Active Inference POMDP agent:

**Graph Structure:**
The graph structures used to represent the input and output data are:

1. **Variable Analysis**:
    - Each observation variable has 20 variables, each labeled as either "observation", "action", or "policy". These are stored in a `variable_set` structure.
    - The number of variables is determined by the number of hidden states (6) and actions (3).

2. **Matrix Analysis**:
   - Each time step is represented via an 8-dimensional matrix containing two columns for each variable, corresponding to the action selected during the iteration process ($t=1$):
    - The action vector $\mathbf{A}=(\mathbf{\theta},\mathbf{\vbm})$, where $\mathbf{\theta}$ are random steps chosen from the policy prior (Habit), and $(\vbm)$ is the probability of moving to a new state. Each row represents one observation, with each column representing one action ($action$.

3. **Connection Patterns**:
    - There are two types of connections between variables:
     - **Forward Connection**: The network connection from an observed action (action) to its predicted output variable $u(x_i)$ is a directed edge. This type connects the input $\mathbf{A}=(\mathbf{\theta}, \vbm)$, and gives the probability distribution over the values of $x_1$ for action actions $(a, b)$. The forward connection propagates $(\mathbf{z}_u^* = (\mathbf{\theta}^\top_{action}(d(a), a))^{-t}$ across two new states.
    - **Backward Connection**: The network connection from an observed observation ($x_i$) to its predicted output variable $y_o$. This type connects the input $\vbm$ (probability distribution over actions) and its corresponding value at the next state, giving the conditional probability of moving to a new state $(a, b)$. The backward connection propagates $(\mathbf{z}_x^* = (\mathbf{\theta}^\top_{observation}(d(a), a))^{-t}$ across one observation from action actions ($action$.

**Mathematical Structure:**

1. **Matrix Analysis**:
   - Each time step is represented via an 8-dimensional matrix containing two columns for each variable, corresponding to the action selected during the iteration process ($i=1$). For example:
    $(\mathbf{\theta}^*_2 = (\mathbf{a}, \mathbf{z})$.

2. **Variable Analysis**:
   - Each observation is represented via an 8-dimensional vector containing two columns for each variable, corresponding to the action selected during the iteration process ($i=1$).
    $(\vbm)^t$ is a directed edge with its first parameter $(x_1)$ and second parameter $(d(a))$. The forward connection propagates $\mathbf{z}_u^* = (\mathbf{\theta}^\top_{action}(d(a), d(a)))^{-i}$ across two new states.
   - Each observation is represented via an 8-dimensional vector containing the predicted output variable $(y_o)^t$.
    $(\vbm)^T$ and $\mathbf{z}_x^* = (\mathbf{\theta}^\top_{observation}(d(a), d(a)))^{-i}$ across one observation.

3. **Connection Patterns**:
   - Each time step is represented via an 8-dimensional matrix containing two columns for each variable, corresponding to the action selected during the iteration process ($i=1$).
    $(\vbm)^t$, etc., represent forward and backward connection connections between a sequence of input observations $(x_i)$.

4. **Symmetries/Special Properties**:
   - The graph structure is composed of 28 nodes representing variables (both fixed and variable), where each node has two neighboring nodes for its output value ($u(a)$). Each node can be connected to the previous or next node by a directed edge, and connected backwards. This corresponds to the idea that one input observation $x_1$ gives rise to multiple outputs $\mathbf{z}_x^*$.
   - There are two types of connections between variables:
     - **Forward Connection**: The network connection from an observed action ($a) $to its predicted output variable $(y_o)^t)$ is a directed edge. This type connects the input vector (input observation $x_1$), followed by all nodes connected to the same input, and then returns to the original node for each subsequent action ($action$, represented as the value of the current state-observation sequence). The forward connection propagates $(\mathbf{z}_u^* = (\vbm)^t_{obs}$ across multiple observations $(x_1$), followed by all nodes connected to the same input, and then returns to itself.
     - **Backward Connection**: The network connection from an observed observation ($a) $to its predicted output variable $(y_o)^T)$ is a directed edge. This type connects the input vector (input observation $x_1$), followed by all nodes connected to the same observable, and then returns back to itself after it has propagated through all observations before returning forward again. The backward connection propagates $(\mathbf{z}_y^* = (\vbm)^T_{obs}$ across one observation $(a$, represented as the value of the current state-observation sequence).
   - There is a duality between these two types of connections, i.e., each input variable connects to all corresponding observations. This corresponds to the idea that it is possible for an action $u(x_1)$ to move from one observable observation ($x_i$) in one step (forward connected), and also to return back to itself after moving through multiple observations $(a, \vbm)^T$.