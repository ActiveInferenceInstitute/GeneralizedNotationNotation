# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

### **How This GNN-Based Active Inference POMDP Agent Runs**
This model implements an **Active Inference** framework within a **Partially Observable Markov Decision Process (POMDP)** using a **Graph Neural Network (GNN)-inspired specification**. Below is a step-by-step breakdown of its operation, behavior, and adaptability across different domains.

---

## **1. Core Mechanics of the Agent**
The agent operates in a discrete, fully-controllable environment with:
- **3 hidden states** (`location`) representing possible states (e.g., rooms, positions).
- **3 observations** (`state_observation`) with deterministic likelihoods (e.g., sensors, cameras).
- **3 actions** (e.g., move left, right, stay) that transition the state deterministically.
- **Preferences encoded as log-probabilities** over observations (e.g., higher reward for certain outcomes).

### **Key Components**
| Component          | Role                                                                 |
|--------------------|----------------------------------------------------------------------|
| **Likelihood Matrix (A)** | Maps hidden states → observations (e.g., if in State 2, Observation 1 is likely). |
| **Transition Matrix (B)** | Determines next state given current state + action (e.g., Action 1 → State 1). |
| **Preference Vector (C)** | Encodes reward/log-probability for observations (e.g., C[2] = 1.0 means highest reward for Observation 2). |
| **Prior (D)**       | Initial belief over hidden states (uniform here).                     |
| **Habit (E)**       | Initial policy prior (uniform here).                                  |
| **Variational Free Energy (F)** | Computes belief updates after observations.                          |
| **Policy (π)**      | Distribution over actions (inferred via Expected Free Energy).        |
| **Action (u)**      | Chosen action (sampled from policy).                                  |

---

## **2. How the Agent Runs in a Single Timestep**
1. **Initialization**
   - Start with uniform prior `D` and habit `E` over actions.
   - Current belief `s` is uniform over hidden states.

2. **Observation**
   - Agent receives an observation `o` (e.g., sensor reading).
   - Uses **Variational Free Energy (F)** to update belief `s`:
     \[
     F = -\log P(o|s) + \text{regularization}
     \]
     (Here, `A` maps `s` → `o` deterministically, so `P(o|s)` is a delta function.)

3. **Policy Inference**
   - Uses **Expected Free Energy (G)** to infer a policy `π` over actions:
     \[
     G = \mathbb{E}_s[\log P(o|s) + \log P(s|D)] - \mathbb{E}_s[\log P(s)]
     \]
     (This balances likelihood and prior.)

4. **Action Selection**
   - Samples `u` from `π` (e.g., Action 2 with probability 0.5).
   - Transitions to next state `s_prime` via `B`:
     \[
     s' = \text{Action}(s, u)
     \]

5. **Repeat**
   - New belief `s` is updated with `s_prime` and next observation.

---

## **3. Behavior in Different Domains**
### **(A) Static vs. Dynamic Environments**
- **Static (Deterministic):**
  - If `B` and `A` are fixed (e.g., grid world with no noise), the agent follows a deterministic path.
  - Example: Always move right if in State 1 → State 2 → State 3.
- **Dynamic (Stochastic):**
  - If `B` or `A` has noise (e.g., probabilistic transitions), the agent must adapt its policy.
  - Example: If `B` allows randomness, it may explore more actions.

### **(B) Different Observation Modalities**
- **Binary Observations:**
  - If `o` has 2 outcomes, `C` would encode preferences (e.g., `C = (0.1, 0.9)`).
  - Agent would prefer the higher-probability observation.
- **Continuous Observations:**
  - If observations were real-valued (e.g., sensor readings), `A` would be a continuous likelihood function.
  - The agent would still use variational inference but with smoother updates.

### **(C) Hierarchical or Multi-Task Learning**
- **Hierarchical POMDPs:**
  - If the agent had sub-goals (e.g., "find treasure" → "collect item"), it could use a **hierarchical GNN** to stack policies.
  - Example: A high-level policy decides "search" vs. "stay," while a low-level policy handles actions.
- **Multi-Task Learning:**
  - If the same agent had to solve multiple POMDPs (e.g., navigation + object collection), it could share latent states via a **shared GNN encoder**.

### **(D) Real-World Adaptations**
- **Robotics:**
  - Replace `A` with camera/sensor data → `o`.
  - Replace `B` with physics-based transitions (e.g., `u` = "push" → `s'` = new position).
- **Finance:**
  - `s` = stock market state, `o` = price tick, `C` = risk/reward preferences.
  - Agent could predict future prices or trade strategies.
- **Healthcare:**
  - `s` = patient symptoms, `o` = lab results, `C` = treatment preferences.
  - Agent could recommend diagnoses or treatments.

---

## **4. Limitations & Extensions**
### **Limitations**
- **No Deep Planning:**
  - The agent only plans for 1 step (no lookahead). For longer horizons, a **recurrent GNN** or **reinforcement learning** extension would help.
- **No Precision Modulation:**
  - Observations are discrete. For continuous data, a **neural variational inference** approach would be needed.
- **No Hierarchy:**
  - No sub-goals or task decomposition. A **hierarchical GNN** could add layers.

### **Possible Extensions**
| Extension               | How to Implement                                                                 |
|-------------------------|----------------------------------------------------------------------------------|
| **Deep Planning**       | Replace `π` with a recurrent GNN (e.g., LSTM) to store past states.              |
| **Hierarchical POMDP**  | Add a high-level policy that selects sub-tasks (e.g., "search" vs. "explore").   |
| **Continuous Observations** | Use a neural network to approximate `A` as a continuous likelihood.           |
| **Multi-Task Learning** | Share latent states across tasks via a shared GNN encoder.                     |
| **Bayesian Neural Networks** | Replace variational inference with Bayesian GNNs for uncertainty estimation.     |

---

## **5. Summary of Key Takeaways**
1. **Active Inference POMDP Agent** uses variational free energy to update beliefs and policies in a discrete environment.
2. **Behavior depends on**:
   - Deterministic vs. stochastic transitions (`B`).
   - Observation modalities (`A`).
   - Preferences (`C`).
3. **Adaptable to**:
   - Robotics, finance, healthcare, etc.
   - Static or dynamic domains.
   - Hierarchical or multi-task setups.
4. **Limitations**:
   - No deep planning, discrete observations.
   - Can be extended with GNNs for richer architectures.

Would you like a deeper dive into any specific aspect (e.g., how to modify the model for a new domain)?