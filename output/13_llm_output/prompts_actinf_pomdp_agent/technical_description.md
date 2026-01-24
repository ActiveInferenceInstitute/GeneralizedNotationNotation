# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

Below is a **comprehensive technical breakdown** of the **Active Inference POMDP Agent** as specified in the GNN model, structured into key components: **mathematical foundations, variational inference, policy learning, and computational architecture**.

---

## **1. Core Model Overview**
The model is a **discrete-time Active Inference agent** operating in a **Partially Observable Markov Decision Process (POMDP)** with:
- **1 hidden state factor** (`location`) with **3 discrete states** (`s ∈ {0,1,2}`).
- **1 observation modality** (`state_observation`) with **3 outcomes** (`o ∈ {0,1,2}`).
- **3 discrete actions** (`u ∈ {0,1,2}`), each deterministically transitioning the state.
- **Preferences encoded as log-probabilities** over observations (`C`).
- **Initial policy prior (habit)** over actions (`E`).

---

## **2. Mathematical Foundations**
### **(A) Likelihood Matrix (`A`)**
Defines the **conditional probability distribution** of observations given hidden states:
\[
A_{o|s} = \begin{bmatrix}
0.9 & 0.05 & 0.05 \\
0.05 & 0.9 & 0.05 \\
0.05 & 0.05 & 0.9
\end{bmatrix}
\]
- **Interpretation**: Each hidden state (`s`) deterministically produces a unique observation (`o`) with high probability (identity mapping).

### **(B) Transition Matrix (`B`)**
Defines **state transitions** given previous state (`s`) and action (`u`):
\[
B_{s'|s,u} = \begin{bmatrix}
\text{Action 0:} & (1.0,0.0,0.0) \\
\text{Action 1:} & (0.0,1.0,0.0) \\
\text{Action 2:} & (0.0,0.0,1.0)
\end{bmatrix}
\]
- **Interpretation**: Each action deterministically moves the state to a unique next state (`s'`).

### **(C) Preference Vector (`C`)**
Encodes **log-preferences** over observations:
\[
C = (0.1, 0.1, 1.0)
\]
- **Interpretation**: Observation `2` is most preferred (highest log-probability), while observations `0` and `1` are equally less preferred.

### **(D) Prior Over Hidden States (`D`)**
Uniform prior over initial hidden states:
\[
D = (0.333, 0.333, 0.333)
\]

### **(E) Initial Policy Prior (`E`)**
Uniform habit over actions:
\[
E = (0.333, 0.333, 0.333)
\]

---

## **3. Variational Inference Framework**
The agent uses **Active Inference** to:
1. **Infer hidden states** (`s`) from observations (`o`) using **Variational Free Energy** (`F`).
2. **Infer policies** (`π`) over actions using **Expected Free Energy** (`G`).
3. **Select actions** (`u`) from the policy posterior.

### **(A) State Inference (`F`)**
The **Variational Free Energy** approximates the posterior belief over hidden states:
\[
F = \mathbb{E}_s[\log p(o|s) + \log p(s)] - \mathbb{E}_s[\log p(s|o)]
\]
- **Optimization**: Minimized via variational inference to update `s`.

### **(B) Policy Inference (`G`)**
The **Expected Free Energy** defines the policy posterior:
\[
G = \mathbb{E}_s[\log p(u|s) + \log p(s)] - \mathbb{E}_s[\log p(s|u)]
\]
- **Optimization**: Maximized to infer `π` over actions.

---

## **4. Computational Architecture**
### **(A) State Transitions (`s → s'`)**
Given current state `s` and action `u`, the next state `s'` is determined by `B`:
\[
s' = \text{Action}(s, u)
\]

### **(B) Observation (`o`)**
The observation `o` is sampled from `A` given the current state `s`:
\[
o \sim \text{Categorical}(A_{o|s})
\]

### **(C) Policy (`π`)**
The policy `π` is a distribution over actions, initialized by the habit `E`:
\[
\pi \sim \text{Categorical}(E)
\]

### **(D) Action Selection (`u`)**
The action `u` is sampled from the policy posterior:
\[
u \sim \text{Categorical}(\pi)
\]

---

## **5. Key Properties**
- **Deterministic Transitions**: Each action deterministically moves the state (`B`).
- **Identity Likelihood**: Each hidden state produces a unique observation (`A`).
- **Unbounded Time Horizon**: The agent operates indefinitely.
- **No Deep Planning**: Only 1-step policy inference (`G`).

---

## **6. Limitations**
- **No Hierarchical Nesting**: The model does not support multi-level policies.
- **No Precision Modulation**: No adaptive sampling or exploration strategies.
- **Fixed Preference Vector**: Preferences are hardcoded (`C`).

---
### **Conclusion**
This **Active Inference POMDP Agent** is a **discrete-time, fully observable (in terms of state transitions) yet partially observable** agent with a **deterministic transition model** and **preference-based policy learning**. It leverages **variational inference** to balance exploration and exploitation in a **3-state, 3-action, 3-observation** environment. The model is **simplified** (no deep planning, no hierarchical nesting) but provides a **foundational framework** for active inference in POMDPs.