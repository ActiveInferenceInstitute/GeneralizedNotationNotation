# IDENTIFY_COMPONENTS

Here is a **systematic, scientifically rigorous breakdown** of the **Active Inference POMDP Agent** GNN specification, organized by Active Inference, Bayesian inference, and GNN semantics:

---

### **1. State Variables (Hidden States)**
#### **Variable Names & Dimensions**
- **`s[3,1,type=float]`**: Current hidden state distribution over **3 discrete states** (e.g., locations: *State 0, State 1, State 2*).
  - Dimension: `(3,1)` → 3 possible states, 1-dimensional vector (probability distribution).
- **`s_prime[3,1,type=float]`**: Next hidden state distribution (posterior belief after action).
  - Same structure as `s`.

#### **Conceptual Meaning**
- The hidden state represents the **true, unobserved state** of the environment (e.g., a robot’s location in a grid world).
- Discrete and finite (3 states), but the model allows for **probabilistic belief updates** (e.g., Bayesian inference over states).

#### **State Space Structure**
- **Discrete**: Only 3 possible states.
- **Finite**: No continuous states; transitions are deterministic or probabilistic.
- **Markovian**: The next state depends only on the current state and action (no memory of past states).

---

### **2. Observation Variables**
#### **Observation Modalities & Meanings**
- **`o[3,1,type=int]`**: Current observation (integer index, 0–2).
  - Represents **3 possible observation outcomes** (e.g., sensor readings: *Observation 0, Observation 1, Observation 2*).
- **Observation modality**: Single modality (no multi-modal observations).

#### **Sensor/Measurement Interpretations**
- The **likelihood matrix `A`** defines the probability of observing each outcome given a hidden state:
  - `A[o|s]`: `P(o|s)` (e.g., `A[0,0] = 0.9` means if the true state is *State 0*, the observation *Observation 0* is most likely).
- **Noise model**: Likelihoods are deterministic (no noise variance specified; assumes perfect observations).

#### **Uncertainty Characterization**
- No explicit noise variance is given, but the model assumes **discrete, deterministic observations** (e.g., binary or categorical).
- If observations were noisy, a **variance parameter** (e.g., `γ` in variational inference) would be needed.

---

### **3. Action/Control Variables**
#### **Available Actions & Their Effects**
- **`u[1,type=int]`**: Chosen action (integer index, 0–2).
  - Represents **3 discrete actions** (e.g., *Action 0, Action 1, Action 2*).
- **Action space properties**:
  - Fully controllable (no hidden actions).
  - Deterministic transitions (no probabilistic action effects).

#### **Control Policies & Decision Variables**
- **`π[3,type=float]`**: Policy (distribution over actions).
  - Represents the agent’s **initial policy prior** (habit) over actions.
  - `E[3,type=float]` encodes this prior (uniformly distributed in this case).
- **Action selection**: No planning horizon (only 1-step lookahead).
  - The agent samples an action from `π` and takes it.

#### **Dynamic Control**
- The agent’s behavior is **deterministic** (no stochasticity in actions).
- The **transition matrix `B`** defines how actions affect the next state:
  - `B[s'|s,u]`: `P(s'|s,u)` (e.g., `B[1,0,0] = 1.0` means if the current state is *State 0* and action is *Action 0*, the next state is *State 1*).

---

### **4. Model Matrices**
#### **A Matrices: Observation Models (`P(o|s)`)**
- **Structure**: `A[3,3,type=float]` (3 observations × 3 hidden states).
- **Content**:
  - Rows: Observations (0, 1, 2).
  - Columns: Hidden states (0, 1, 2).
  - Example: `A[0,0] = 0.9` → If state is *0*, observation *0* is most likely.
- **Interpretation**: Likelihood of observing each outcome given a hidden state.

#### **B Matrices: Transition Dynamics (`P(s'|s,u)`)**
- **Structure**: `B[3,3,3,type=float]` (3 next states × 3 previous states × 3 actions).
- **Content**:
  - Each slice corresponds to an action (0, 1, 2).
  - Example: `B[1,0,0]` is the transition matrix for *Action 0*:
    - `B[1,0,0] = (1.0, 0.0, 0.0)` → If current state is *0* and action is *0*, next state is *1* deterministically.
- **Interpretation**: How actions change the hidden state.

#### **C Matrices: Preferences/Goals (`P(o)`)**
- **Structure**: `C[3,type=float]` (3 observations).
- **Content**: `C = (0.1, 0.1, 1.0)` → Log-preferences over observations.
  - Higher values = more preferred outcomes.
- **Interpretation**: The agent’s **utility function** for observations.

#### **D Matrices: Prior Beliefs (`P(s)`)**
- **Structure**: `D[3,type=float]` (3 hidden states).
- **Content**: `D = (0.333, 0.333, 0.333)` → Uniform prior over states.
- **Interpretation**: Initial belief over hidden states.

#### **E Matrices: Habit (Initial Policy Prior)**
- **Structure**: `E[3,type=float]` (3 actions).
- **Content**: `E = (0.333, 0.333, 0.333)` → Uniform initial policy.
- **Interpretation**: The agent’s **default action selection** before learning.

---

### **5. Parameters and Hyperparameters**
| Parameter          | Role                                                                 | Value/Type               |
|--------------------|----------------------------------------------------------------------|--------------------------|
| **Precision (γ)** | Not explicitly defined; assumes deterministic observations.         | N/A                      |
| **Learning rate**  | Not specified; model is static (no online learning).                 | N/A                      |
| **Fixed parameters** | `A`, `B`, `C`, `D`, `E` are hardcoded.                                | Hardcoded (e.g., `A` as identity matrix) |
| **Adaptation**     | No learning; model is fixed for simulation.                          | N/A                      |

---

### **6. Temporal Structure**
#### **Time Horizons & Temporal Dependencies**
- **Discrete time**: `t[1,type=int]` represents a single time step.
- **Dynamic components**:
  - The agent updates its belief (`s`) and policy (`π`) at each step.
  - No deep planning (only 1-step lookahead).
- **Model horizon**: Unbounded (`ModelTimeHorizon=Unbounded`).
- **Markov property**: The next state depends only on the current state and action (no memory of past states).

#### **Dynamic vs. Static Components**
- **Dynamic**:
  - Belief updates (`s` → `s_prime`) via variational inference.
  - Policy updates (`π`) via expected free energy.
- **Static**:
  - Matrices `A`, `B`, `C`, `D`, `E` are fixed.
  - No hierarchical nesting or precision modulation.

---

### **Key Active Inference Concepts in the Model**
1. **Variational Free Energy (`F`)**:
   - Used for belief updating (`s` → `s_prime`).
   - Minimizes the KL divergence between the true posterior and a variational distribution.

2. **Expected Free Energy (`G`)**:
   - Used for policy inference (`π`).
   - Balances exploration/exploitation via the policy prior (`E`).

3. **Belief Propagation**:
   - The agent updates its belief over hidden states (`s`) using observations (`o`) and transitions (`B`).

4. **Policy Gradient**:
   - The agent’s action selection is guided by the policy posterior (`π`), which is updated via `G`.

5. **Noisy Channel Model**:
   - Observations are deterministic (no noise variance), but the model could be extended to include stochasticity.

---

### **Practical Implications**
- **Simulation Use Case**:
  - This model is suitable for discrete POMDPs (e.g., robotics, game AI).
  - The deterministic transitions (`B`) and observations (`A`) make it easy to simulate.
- **Learning Extension**:
  - If the model were to learn parameters (e.g., `A`, `B`), a reinforcement learning approach (e.g., PPO) could be applied.
- **Scalability**:
  - The 3-state/3-action space is small, but the model could be extended to larger state/action spaces with variational inference.

---
### **Summary Table**
| Component          | Role                                                                 |
|--------------------|----------------------------------------------------------------------|
| **Hidden States**  | `s[3,1]`: Current belief over 3 states.                               |
| **Observations**   | `o[3]`: 3 possible outcomes (deterministic).                         |
| **Actions**        | `u[3]`: 3 discrete actions (deterministic transitions).              |
| **A (Likelihood)** | `P(o|s)`: Deterministic observation model.                            |
| **B (Transitions)**| `P(s'|s,u)`: Deterministic state transitions.                           |
| **C (Preferences)**| `P(o)`: Agent’s utility for observations.                           |
| **D (Prior)**      | `P(s)`: Uniform prior over states.                                   |
| **E (Policy)**     | `P(u)`: Uniform initial policy.                                     |
| **Temporal**       | Discrete, unbounded horizon; 1-step lookahead.                     |
| **Inference**      | Variational free energy for belief updates; expected free energy for policy. |

This model is a **foundational POMDP agent** with deterministic dynamics, suitable for simulation and extension to more complex scenarios.