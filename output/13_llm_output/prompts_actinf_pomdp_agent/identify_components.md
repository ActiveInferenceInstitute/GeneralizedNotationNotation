# IDENTIFY_COMPONENTS

Here is a **systematic breakdown** of the **Active Inference POMDP Agent** GNN specification, focusing on its **Active Inference, Bayesian inference, and POMDP structure** with detailed explanations of each component:

---

### **1. State Variables (Hidden States)**
#### **Variable Names & Dimensions**
- **`s[3,1,type=float]`**: Current hidden state distribution over 3 discrete states (e.g., locations).
  - **Shape**: `(3,1)` → A vector of length 3 representing the posterior belief over hidden states.
- **`s_prime[3,1,type=float]`**: Next hidden state distribution (predicted belief).
  - **Shape**: Same as `s`, representing the updated belief after an action.

#### **Conceptual Meaning**
- The hidden state represents a **discrete, fully observable (but unknown to the agent) environment variable** (e.g., a location in a grid world).
- The agent maintains a **belief distribution** over possible states (e.g., `s = [p(s₁), p(s₂), p(s₃)]`).
- The state space is **finite and discrete** (3 states), with no continuous components.

#### **State Space Structure**
- **Discrete**: Only 3 possible states.
- **Finite**: No infinite state space.
- **Fully controllable**: The agent can directly influence the state via actions (no hidden dynamics).

---

### **2. Observation Variables**
#### **Observation Modalities & Meanings**
- **`o[3,1,type=int]`**: Current observation (integer index).
  - **Shape**: `(3,1)` → A vector of length 3 representing the observed outcome (e.g., a sensor reading).
  - **Possible values**: `{0, 1, 2}` (3 discrete outcomes).

#### **Sensor/Measurement Interpretations**
- The **likelihood matrix `A`** defines how observations are generated from hidden states:
  - `A[o|s]`: Probability of observing `o` given state `s`.
  - Example: `A = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]` (identity-like mapping).
- **Noise model**: Observations are noisy but deterministic (each state maps to a unique observation with high probability).

#### **Uncertainty Characterization**
- The agent infers the **true hidden state** from noisy observations using **Variational Free Energy (F)**.
- The **likelihood `A`** encodes the **observation model**, while the **belief `s`** updates to maximize expected evidence.

---

### **3. Action/Control Variables**
#### **Available Actions & Their Effects**
- **`u[1,type=int]`**: Chosen action (integer index).
  - **Shape**: `(1,)` → A single action from `{0, 1, 2}`.
  - **Possible actions**: 3 discrete actions (e.g., move left, right, or stay).

#### **Control Policies & Decision Variables**
- **`π[3,type=float]`**: Policy (distribution over actions).
  - **Shape**: `(3,)` → A vector of log-probabilities over actions (e.g., `π = [p(a₀), p(a₁), p(a₂)]`).
  - **Initial policy (`E`)**: Uniform prior (`E = [0.333, 0.333, 0.333]`).
- **`G[π,type=float]`**: Expected Free Energy (per policy).
  - Computed as `G = -F + C`, where `F` is the variational free energy and `C` is the preference vector.

#### **Action Space Properties**
- **Discrete**: Only 3 actions.
- **No planning horizon**: The agent acts greedily (no lookahead).
- **Fully controllable**: Actions directly influence the state transition.

---

### **4. Model Matrices**
#### **A Matrices: Observation Models (`P(o|s)`)**
- **Shape**: `(3,3)` → Likelihood of observing `o` given state `s`.
- **Content**:
  - `A = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]`
  - **Interpretation**: Each row is an observation, each column is a hidden state.
  - **Example**: `A[0|1] = 0.9` → If state `s=1`, observe `o=0` with probability 0.9.

#### **B Matrices: Transition Dynamics (`P(s'|s,u)`)**
- **Shape**: `(3,3,3)` → Transition probabilities given previous state `s` and action `u`.
- **Content**:
  - `B = [ [ (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0) ],
           [ (0.0,1.0,0.0), (1.0,0.0,0.0), (0.0,0.0,1.0) ],
           [ (0.0,0.0,1.0), (0.0,1.0,0.0), (1.0,0.0,0.0) ] ]`
  - **Interpretation**: Each slice corresponds to an action. For example, `B[0|0,0]` = 1.0 → If `s=0` and `u=0`, stay in state `s=0`.
  - **Example**: `B[1|1,1]` = 1.0 → If `s=1` and `u=1`, transition to `s=0`.

#### **C Matrices: Preferences/Goals (`C`)**
- **Shape**: `(3,)` → Log-preferences over observations.
- **Content**: `C = [0.1, 0.1, 1.0]`
  - **Interpretation**: Higher values indicate stronger preferences.
  - **Example**: `C[2] = 1.0` → Observation `o=2` is most preferred.

#### **D Matrices: Prior Beliefs (`P(s)`)**
- **Shape**: `(3,)` → Prior over initial hidden states.
- **Content**: `D = [0.333, 0.333, 0.333]`
  - **Interpretation**: Uniform prior (no bias toward any state).

#### **E Matrices: Habit (Initial Policy)**
- **Shape**: `(3,)` → Initial policy prior over actions.
- **Content**: `E = [0.333, 0.333, 0.333]`
  - **Interpretation**: Uniform initial policy (no preference for any action).

---

### **5. Parameters and Hyperparameters**
| Parameter | Role | Value | Learnable? |
|-----------|------|-------|------------|
| **A**     | Likelihood matrix | Fixed (identity-like) | No |
| **B**     | Transition matrix | Fixed (deterministic) | No |
| **C**     | Preference vector | `[0.1, 0.1, 1.0]` | No |
| **D**     | Prior over states | Uniform `[0.333, 0.333, 0.333]` | No |
| **E**     | Habit (initial policy) | Uniform `[0.333, 0.333, 0.333]` | No |
| **F**     | Variational Free Energy | Computed dynamically | No (fixed by model) |
| **G**     | Expected Free Energy | Computed as `-F + C` | No |
| **Precision parameters** | None | - | - |

- **No learnable parameters**: All matrices (`A`, `B`, `C`, `D`, `E`) are fixed.
- **Dynamic components**: Only `s`, `s_prime`, `o`, `π`, and `u` are updated during inference.

---

### **6. Temporal Structure**
#### **Time Horizons & Temporal Dependencies**
- **Discrete time steps**: `t[1,type=int]` represents the current time step.
- **Unbounded horizon**: The agent is defined for an infinite time horizon (`ModelTimeHorizon=Unbounded`).
- **Dynamic components**:
  - The state `s` evolves over time via transitions (`B`).
  - Observations `o` are noisy and update the belief `s`.
  - Actions `u` are chosen based on the current policy `π`.
- **No lookahead**: The agent acts greedily (no future planning).

#### **Dynamic vs. Static Components**
| Component | Dynamic? | Role |
|-----------|----------|------|
| **State `s`** | Yes | Belief over hidden states |
| **Observation `o`** | Yes | Noisy sensor reading |
| **Action `u`** | Yes | Chosen by policy `π` |
| **Policy `π`** | Yes (updates via `G`) | Greedy action selection |
| **Matrices `A`, `B`, `C`, `D`, `E`** | No | Fixed model parameters |

---

### **Summary of Key Features**
1. **Bayesian Inference**: The agent maintains a belief distribution `s` over hidden states and updates it using Variational Free Energy (`F`).
2. **Active Inference**: The agent actively queries the environment (via actions) to maximize expected evidence.
3. **POMDP Structure**:
   - **Hidden states**: `s` (unknown to the agent).
   - **Observations**: `o` (noisy and deterministic).
   - **Actions**: `u` (discrete, no planning).
   - **Transition model**: `B` (deterministic).
   - **Reward/Preference**: `C` (log-preferences over observations).
4. **Greedy Policy**: The agent acts based on the current policy `π` (no lookahead).
5. **Fixed Parameters**: All model matrices (`A`, `B`, `C`, `D`, `E`) are hardcoded.

This agent is a **simple but effective** example of **Active Inference in POMDPs**, where the goal is to maximize expected evidence (or reward) by querying the environment optimally.