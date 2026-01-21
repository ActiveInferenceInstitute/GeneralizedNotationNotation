# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

### **How This GNN-Based Active Inference POMDP Agent Runs**
This model implements a **discrete-time, one-step lookahead** agent using **Active Inference** principles within a **Partially Observable Markov Decision Process (POMDP)** framework. Below is a step-by-step breakdown of its operation and behavior in different settings.

---

## **1. Core Mechanics of the Agent**
### **(A) Hidden State & Observations**
- The agent operates in a **3-state hidden world** (`location`), where each state (`s`) can transition deterministically based on actions (`u`).
- Observations (`o`) are **discrete (3 outcomes)**, but they are **not directly tied to the hidden state** (unlike a deterministic POMDP). Instead, they follow a **likelihood matrix (`A`)** that maps hidden states to observation probabilities.
- The agent’s **belief** is represented as a distribution over hidden states (`s`), updated via **Variational Free Energy (F)**.

### **(B) Likelihood Matrix (`A`)**
The matrix defines how observations are generated from hidden states:
```python
A = [
    [0.9, 0.05, 0.05],  # Observation 0 (highest prob for state 0)
    [0.05, 0.9, 0.05],  # Observation 1 (highest prob for state 1)
    [0.05, 0.05, 0.9]   # Observation 2 (highest prob for state 2)
]
```
- If the agent is in **state 0**, it is most likely to observe **0** (90% chance).
- If it is in **state 1**, it is most likely to observe **1** (90% chance).
- If it is in **state 2**, it is most likely to observe **2** (90% chance).

### **(C) Transition Matrix (`B`)**
The matrix defines how the hidden state evolves based on actions:
```python
B = [
    [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],  # Action 0 → state 0
    [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],  # Action 1 → state 1
    [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]   # Action 2 → state 2
]
```
- **Action 0** → Always moves to **state 0**.
- **Action 1** → Always moves to **state 1**.
- **Action 2** → Always moves to **state 2**.

### **(D) Preferences (`C`) & Prior (`D`)**
- **Preferences (`C`)** encode the agent’s **utility function** over observations:
  ```python
  C = [0.1, 0.1, 1.0]  # Observation 2 is most preferred (log-prob = 1.0)
  ```
  - The agent **prefers observation 2** (highest utility).
- **Prior (`D`)** is uniform over hidden states:
  ```python
  D = [0.333, 0.333, 0.333]  # Equal chance of being in any state initially.
  ```

### **(E) Habit (`E`) & Policy (`π`)**
- **Habit (`E`)** is the agent’s **initial policy prior** (uniform over actions):
  ```python
  E = [0.333, 0.333, 0.333]  # Equal chance of choosing any action initially.
  ```
- **Policy (`π`)** is updated via **Expected Free Energy (G)** and sampled to choose an action.

---

## **2. How the Agent Runs in a Step**
### **(A) Initialization**
1. The agent starts with a **uniform belief over hidden states** (`s = [0.333, 0.333, 0.333]`).
2. It selects an action based on its **habit** (`π = [0.333, 0.333, 0.333]`).

### **(B) Observation & Belief Update**
1. The agent takes an action (`u`), transitions to a new hidden state (`s_prime`), and observes an outcome (`o`).
2. It updates its **belief** using **Variational Free Energy (F)**:
   - Computes the **likelihood of the observation given the hidden state** (`A`).
   - Updates `s` to reflect the most probable hidden state given the observation.
3. The agent now has a **new belief distribution** over hidden states.

### **(C) Policy Update & Action Selection**
1. The agent computes the **Expected Free Energy (G)** for each possible action (`π`).
2. It samples an action (`u`) from the **policy posterior** (weighted by `G`).
3. The cycle repeats.

---

## **3. Behavior in Different Settings**
### **(A) Deterministic vs. Stochastic Environments**
- **Deterministic (`B` is identity matrix)**:
  - The agent’s actions **directly control the hidden state** (e.g., `Action 0 → State 0`).
  - It can **predict the next state** and optimize actions accordingly.
- **Stochastic (`B` has non-zero probabilities)**:
  - The agent must **learn the transition probabilities** and update its belief dynamically.
  - It may need **more observations** to refine its policy.

### **(B) Different Observation Modalities**
- If observations were **continuous** (e.g., sensor readings), the likelihood matrix (`A`) would be a **probability density function (PDF)** instead of a discrete table.
- If observations were **more complex** (e.g., multi-modal), the agent would need to **generalize across different observation patterns**.

### **(C) Different Preferences (`C`)**
- If the agent **prefers observation 0** (`C = [1.0, 0.1, 0.1]`), it would **prioritize actions that lead to state 0** (since `A[0,0] = 0.9`).
- If the agent **prefers no observation** (e.g., in a silent environment), it might **avoid actions that lead to high-probability observations**.

### **(D) Different Transition Matrices (`B`)**
- If `B` was **non-deterministic** (e.g., `Action 0 → [0.7, 0.3, 0.0]`), the agent would **randomly sample from the transition probabilities** rather than deterministically choosing the next state.
- This would require **more frequent updates** to its belief.

### **(E) Hierarchical or Deep Planning**
- The current model **only plans one step ahead** (`ModelTimeHorizon=Unbounded` but no deep planning).
- If extended to **multi-step planning**, the agent would:
  - **Forecast future states** based on current actions.
  - **Optimize policies** to maximize long-term utility.
  - Require **more complex belief updates** (e.g., using **recurrent GNNs** or **hierarchical POMDPs**).

---

## **4. Limitations & Extensions**
### **(A) Current Limitations**
1. **No Deep Planning**: Only one-step lookahead.
2. **No Precision Modulation**: Belief updates are fixed (no adaptive sampling).
3. **No Hierarchical Nesting**: No sub-goals or task decomposition.
4. **Fixed Transition Matrix**: Deterministic (`B` is identity).

### **(B) Possible Extensions**
1. **Stochastic Transitions**: Allow `B` to have non-zero probabilities.
2. **Continuous Observations**: Replace `A` with a PDF.
3. **Hierarchical POMDPs**: Break tasks into sub-goals.
4. **Recurrent GNNs**: Enable memory for long-term dependencies.
5. **Bayesian Optimization**: Optimize preferences (`C`) dynamically.

---

## **5. Summary of Key Takeaways**
| **Component**       | **Role**                                                                 | **Behavior in Different Settings**                          |
|----------------------|--------------------------------------------------------------------------|-------------------------------------------------------------|
| **Hidden State (`s`)** | Determines observations via `A`.                                         | If `A` is deterministic, observations are fixed per state.   |
| **Transition (`B`)**  | Defines how actions move the state.                                       | Deterministic (`B` identity) → exact control; stochastic → random jumps. |
| **Observations (`o`)** | Likelihood of seeing `o` given `s`.                                       | If `C` favors `o=2`, agent prioritizes states that map to `o=2`. |
| **Preferences (`C`)** | Agent’s utility function over observations.                             | If `C` is high for `o=0`, agent seeks states that produce `o=0`. |
| **Policy (`π`)**      | Action selection based on belief and preferences.                       | If `B` is stochastic, agent must sample actions from transition probabilities. |

### **Final Thoughts**
This GNN-based Active Inference POMDP agent is a **flexible framework** for solving **discrete POMDPs** with **one-step lookahead**. Its behavior depends heavily on:
- The **likelihood matrix (`A`)** (observation structure).
- The **transition matrix (`B`)** (how actions affect the state).
- The **preferences (`C`)** (what the agent values).

By modifying these parameters, the agent can adapt to **different domains** (e.g., robotics, game AI, reinforcement learning). For **longer horizons or more complex tasks**, extensions like **hierarchical planning** or **recurrent GNNs** would be needed.