# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

Below is a **comprehensive technical breakdown** of the **Active Inference POMDP Agent** as specified in the GNN model, structured into key components, mathematical foundations, and implementation details.

---

## **1. Overview & Architectural Design**
### **1.1 Core Concepts**
- **Active Inference (AI)** is a Bayesian framework for **exploration vs. exploitation** in uncertain environments, where the agent optimizes **free energy minimization** (a proxy for prediction error).
- **POMDP (Partially Observable Markov Decision Process)** formalizes sequential decision-making under uncertainty, where the agent must infer hidden states from noisy observations.
- The model is **discrete-time**, with **one hidden state factor** (`location`) and **one observation modality** (`state_observation`).

### **1.2 Key Features**
- **Deterministic transitions** (no stochasticity in `B`).
- **Log-preference encoding** (`C`) for observation-based rewards.
- **Uniform prior** (`D`) and **uniform habit** (`E`) for initial policies.
- **Single-step planning** (no deep planning).
- **Variational Free Energy (F)** for belief updating.

---

## **2. Mathematical Formalism**
### **2.1 State Representation**
- **Hidden State (`s`)**:
  - A **probability distribution** over 3 discrete states:
    \[
    s \sim \mathcal{D}_s \quad \text{(e.g., } s = [p(s_1), p(s_2), p(s_3)]\text{)}
    \]
  - **Initial prior (`D`)** is uniform:
    \[
    D = \left[ \frac{1}{3}, \frac{1}{3}, \frac{1}{3} \right]
    \]

- **Next Hidden State (`s_prime`)**:
  - Updated via **transition dynamics** (`B`):
    \[
    s' \sim \text{Transition}(s, u)
    \]
  - **Action (`u`)** is sampled from the **policy (`π`)**.

### **2.2 Observation Likelihood (`A`)**
- **Deterministic mapping** from hidden states to observations:
  \[
  A = \begin{bmatrix}
  0.9 & 0.05 & 0.05 \\
  0.05 & 0.9 & 0.05 \\
  0.05 & 0.05 & 0.9
  \end{bmatrix}
  \]
  - **Row**: Observation index (0, 1, 2).
  - **Column**: Hidden state index (0, 1, 2).
  - **Interpretation**: If `s = s_2`, observation `o = 1` has likelihood `0.9`.

### **2.3 Transition Dynamics (`B`)**
- **Deterministic transitions** for each action:
  \[
  B = \begin{bmatrix}
  \text{Action 0: } \begin{bmatrix} 1.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 \\ 0.0 & 0.0 & 1.0 \end{bmatrix} \\
  \text{Action 1: } \begin{bmatrix} 0.0 & 1.0 & 0.0 \\ 1.0 & 0.0 & 0.0 \\ 0.0 & 0.0 & 1.0 \end{bmatrix} \\
  \text{Action 2: } \begin{bmatrix} 0.0 & 0.0 & 1.0 \\ 0.0 & 1.0 & 0.0 \\ 1.0 & 0.0 & 0.0 \end{bmatrix}
  \end{bmatrix}
  \]
  - **Interpretation**: Action `u=0` moves from `s_i` to `s_i` (no change), `u=1` cycles states, `u=2` reverses the cycle.

### **2.4 Preferences (`C`)**
- **Log-preference vector** over observations:
  \[
  C = [\log(0.1), \log(0.1), \log(1.0)] = [-2.302585, -2.302585, 0]
  \]
  - **Interpretation**: Observation `o=2` is preferred (highest log-probability), while `o=0` and `o=1` are equally penalized.

### **2.5 Policy (`π`)**
- **Initial habit (`E`)** is uniform:
  \[
  E = \left[ \frac{1}{3}, \frac{1}{3}, \frac{1}{3} \right]
  \]
- **Policy posterior (`π`)** is updated via **Expected Free Energy (G)**:
  \[
  G = \mathbb{E}_s[\log \pi(u | s) + \log p(o | s) - \log p(s)]
  \]
  - **Action selection**: Sample `u ~ π(u | s)` from the policy posterior.

### **2.6 Free Energy (`F` and `G`)**
- **Variational Free Energy (`F`)** for belief updating:
  \[
  F = \mathbb{E}_s[\log p(s) + \log p(o | s) - \log \pi(s)]
  \]
  - Minimized to infer `s` from `o`.
- **Expected Free Energy (`G`)** for policy optimization:
  \[
  G = \mathbb{E}_s[\log \pi(u | s) + \log p(o | s) - \log p(s)]
  \]
  - Minimized to optimize `π`.

---

## **3. Dynamic Process**
### **3.1 Time Evolution**
1. **Initialization**:
   - Set `s = D` (uniform prior).
   - Sample `u ~ E` (uniform habit).
   - Compute `s' = B(s, u)`.

2. **Observation**:
   - Sample `o` from `A(s)` (likelihood).
   - Update `s` via **variational inference** (`F`).

3. **Policy Update**:
   - Compute `G` (Expected Free Energy).
   - Optimize `π` to minimize `G`.

4. **Action Selection**:
   - Sample `u ~ π(u | s)`.
   - Transition to `s' = B(s, u)`.

5. **Repeat** for unbounded time steps.

---

## **4. Implementation Notes**
### **4.1 Key Assumptions**
- **Deterministic transitions** (`B`).
- **No stochasticity** in `A` or `B`.
- **Single-step planning** (no lookahead).
- **Uniform prior and habit** (no learned biases).

### **4.2 Limitations**
- **No deep planning**: Only one-step lookahead.
- **No precision modulation**: Fixed precision in `F`/`G`.
- **No hierarchical nesting**: Flat policy space.

### **4.3 Extensions (Hypothetical)**
- **Stochastic transitions**: Replace `B` with a stochastic matrix.
- **Deep planning**: Add a planning horizon (`k` steps).
- **Hierarchical policies**: Nest policies in a tree structure.

---

## **5. Summary Table**
| Component          | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| **Hidden State (`s`)** | 3 discrete states, updated via `B`.                                         |
| **Observation (`o`)** | 3 discrete outcomes, sampled from `A(s)`.                                    |
| **Likelihood (`A`)**  | Deterministic mapping `s → o`.                                               |
| **Transition (`B`)**  | Deterministic cycles for each action.                                       |
| **Preferences (`C`)** | Log-preferences over observations.                                           |
| **Policy (`π`)**     | Optimized via Expected Free Energy (`G`).                                     |
| **Free Energy (`F`)** | Variational inference for belief updating.                                  |
| **Action (`u`)**     | Sampled from `π(u | s)`.                                                                      |

---
This model provides a **fully specified, deterministic POMDP agent** for active inference, suitable for simulation or inference backends. Would you like any clarifications or additional details on specific components?