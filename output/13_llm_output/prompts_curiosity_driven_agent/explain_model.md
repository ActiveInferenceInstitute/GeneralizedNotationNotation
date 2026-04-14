# EXPLAIN_MODEL

Here's a concise overview of the key points:

**Model Purpose:** This model represents an active inference agent that uses hidden states (s_f0, s_f1) and observations to update beliefs based on actions/control inputs. The goal is to explore exploration-exploitation scenarios using this agent.

**Core Components:**

1. **Hidden States**: These are represented by the state space blocks `d` (`D`) in the model. Each block represents a single action, and each block has two states: one for the current observation (s_f0) and another for the next observation (s_f1). The hidden states represent epistemic values that can be updated based on actions/control inputs.

2. **Observations**: These are represented by the state space blocks `o` (`O`) in the model. Each block represents a single action, and each block has two observations: one for the current observation (s_f0) and another for the next observation (s_f1). The observables represent beliefs that can be updated based on actions/control inputs.

3. **Actions**: These are represented by the state space blocks `p` (`P`) in the model. Each block represents a single action, and each block has two actions: one for the current observation (s_f0) and another for the next observation (s_f1). The actions represent beliefs that can be updated based on actions/control inputs.

**Model Dynamics:**

1. **Initialization**: The model initializes with a set of states (`d`) and observations (`o`), which are represented by the state space blocks `d`. Each block represents an action, and each block has two actions: one for the current observation (s_f0) and another for the next observation (s_f1).

2. **Initialization**: The model initializes with a set of states (`d`) and observations (`o`), which are represented by the state space blocks `d`. Each block represents an action, and each block has two actions: one for the current observation (s_f0) and another for the next observation (s_f1).

3. **Initialization**: The model initializes with a set of states (`d`) and observations (`o`), which are represented by the state space blocks `d`. Each block represents an action, and each block has two actions: one for the current observation (s