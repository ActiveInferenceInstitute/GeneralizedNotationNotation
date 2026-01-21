# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

### **How This GNN-Based Active Inference POMDP Agent Runs**
This model implements an **Active Inference** framework within a **Partially Observable Markov Decision Process (POMDP)**, where the agent optimizes its observations to minimize its **variational free energy** (a measure of surprise and uncertainty). Below is a step-by-step breakdown of its operation and behavior in different settings.

---

## **1. Core Mechanics of the Agent**
### **(A) State Representation & Hidden Dynamics**
- The agent operates in a **discrete hidden state space** (`s[3]`) with 3 possible locations (e.g., rooms in a building, positions in a grid).
- The **transition dynamics** (`B`) are deterministic:
  - Each action (`u ∈ {0,1,2}`) moves the agent to a new state deterministically (e.g., action 0 → state 0, action 1 → state 1, etc.).
- The agent’s **initial belief** (`D`) is uniform over the 3 states.

### **(B) Observations & Likelihoods**
- The agent observes **3 possible outcomes** (`o ∈ {0,1,2}`) with a **deterministic likelihood matrix (`A`)**:
  - Each hidden state (`s`) maps to a unique observation (e.g., state 0 → observation 0, state 1 → observation 1, etc.).
- The agent’s **preference vector (`C`)** encodes its utility over observations:
  - Observation 2 is the most preferred (highest log-probability), while observations 0 and 1 are equally less preferred.

### **(C) Policy & Action Selection**
- The agent’s **initial policy (`E`)** is uniform over actions (no bias).
- At each step, it computes:
  - **Variational Free Energy (`F`)** to infer the posterior belief over hidden states.
  - **Expected Free Energy (`G`)** to evaluate policies.
  - It then **selects an action** (`π`) that minimizes `G` (greedy policy).

---

## **2. How the Agent Behaves in Different Settings**
### **(A) Exploration vs. Exploitation**
- **Initial Exploration**: Since `E` is uniform, the agent randomly samples actions. It may take multiple steps to converge to a preferred state (e.g., observation 2).
- **Exploitation**: Once it observes a high-preference outcome (e.g., `o=2`), it will **stay in that state** (since `B` is deterministic, it cannot leave unless it takes a different action).

### **(B) Dynamic Environments**
- If the **transition matrix (`B`)** changes (e.g., actions no longer deterministically move the agent), the agent must **recompute its policy** to adapt.
- If the **likelihood matrix (`A`)** changes (e.g., observations become less deterministic), the agent must **update its belief** to reflect new uncertainty.

### **(C) Different Domains**
#### **(1) Grid World Example**
- **States**: 3 rooms (e.g., `s=0` = kitchen, `s=1` = living room, `s=2` = bedroom).
- **Actions**: Move left, right, or stay.
- **Observations**: A sensor detects whether the agent is in a "safe" (high-preference) or "dangerous" (low-preference) area.
- **Behavior**: The agent will **explore to find the safe area** and then **stay there** until a new action is taken.

#### **(2) Robot Navigation**
- **States**: 3 positions in a maze.
- **Actions**: Move forward, turn left, turn right.
- **Observations**: A camera detects whether the agent is in a "high-value" (e.g., food source) or "low-value" (e.g., obstacle) area.
- **Behavior**: The agent will **actively query observations** to minimize free energy, avoiding obstacles and maximizing rewards.

#### **(3) Financial Trading (POMDP-like)**
- **States**: 3 market conditions (bullish, neutral, bearish).
- **Actions**: Buy, sell, hold.
- **Observations**: A sensor detects whether the market is trending up/down.
- **Behavior**: The agent will **adapt its strategy** based on observed trends, avoiding overtrading in uncertain conditions.

---

## **3. Key Limitations & Extensions**
### **(A) Current Limitations**
- **No Deep Planning**: The agent only considers **one-step actions** (no lookahead).
- **No Hierarchical Control**: It does not decompose decisions into subgoals.
- **Fixed Likelihoods & Preferences**: If the environment changes, the agent must relearn.

### **(B) Possible Extensions**
- **Recurrent Active Inference**: Use **RNNs/GNNs** to model longer-term dependencies.
- **Hierarchical POMDPs**: Break decisions into subroutines (e.g., explore → exploit).
- **Adaptive Preferences**: Let the agent learn **dynamic utility functions** over time.
- **Non-Deterministic Transitions**: Allow probabilistic actions (`B` with softmax).

---

## **4. Summary of Behavior**
| **Setting**               | **Agent’s Strategy**                                                                 |
|---------------------------|------------------------------------------------------------------------------------|
| **Exploration**           | Randomly samples actions until it finds a high-preference state.                   |
| **Exploitation**          | Once in a preferred state, it **stays there** (since `B` is deterministic).       |
| **Dynamic Environments**  | Must **recompute policy** if `A` or `B` changes.                                   |
| **Grid World**            | Finds the safest room and **stays there**.                                          |
| **Robot Navigation**      | Actively queries observations to **minimize free energy** (avoid obstacles).       |
| **Financial Trading**    | Adapts to **market trends** based on observed data.                                |

This model is a **minimal but powerful** example of how **Active Inference + GNNs** can be used to build **adaptive, exploration-exploitation agents** in structured POMDPs. Would you like a deeper dive into any specific aspect?