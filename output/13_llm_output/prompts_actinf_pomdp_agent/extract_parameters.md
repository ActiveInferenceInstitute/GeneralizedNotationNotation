# EXTRACT_PARAMETERS

Here is a **systematic breakdown** of the parameters in the provided **Active Inference POMDP Agent** GNN specification, organized by conceptual categories with rigorous mathematical and practical implications:

---

### **1. Model Matrices**
#### **(A) Likelihood Matrix**
- **Dimensions**: \( A \in \mathbb{R}^{3 \times 3} \) (observations × hidden states).
- **Structure**:
  - Rows index observations (0, 1, 2), columns index hidden states (0, 1, 2).
  - **Identity mapping**: Each hidden state deterministically produces a unique observation (e.g., state 0 → observation 0 with probability 0.9).
- **Interpretation**:
  - \( A_{o,s} = P(o|s) \): Conditional probability of observation \( o \) given hidden state \( s \).
  - **Active Inference Role**: Used in variational inference to compute posterior beliefs over hidden states.

#### **(B) Transition Matrix**
- **Dimensions**: \( B \in \mathbb{R}^{3 \times 3 \times 3} \) (next states × previous states × actions).
- **Structure**:
  - Each slice (action) is a \( 3 \times 3 \) matrix where rows = previous states, columns = next states.
  - **Deterministic transitions**: Each action moves the system to a fixed next state (e.g., action 0 → state 0 → state 1).
- **Interpretation**:
  - \( B_{s',s,a} = P(s'|s,a) \): Transition probability of next state \( s' \) given current state \( s \) and action \( a \).
  - **Active Inference Role**: Core of the POMDP dynamics; used in belief updating and policy inference.

#### **(C) Preference Vector**
- **Dimensions**: \( C \in \mathbb{R}^{3} \) (observations).
- **Structure**:
  - Log-preferences over observations (log-probabilities).
  - Example: \( C = (0.1, 0.1, 1.0) \): Observation 2 is most preferred (highest log-probability).
- **Interpretation**:
  - \( C_o = \log P(o) \): Reward signal for observation \( o \).
  - **Active Inference Role**: Encodes the agent’s intrinsic motivation; used in expected free energy minimization.

#### **(D) Prior Vector**
- **Dimensions**: \( D \in \mathbb{R}^{3} \) (hidden states).
- **Structure**:
  - Uniform prior: \( D = (0.333, 0.333, 0.333) \).
- **Interpretation**:
  - \( D_s = P(s) \): Prior probability of hidden state \( s \) at initialization.
  - **Active Inference Role**: Initial belief over hidden states; used in variational inference.

---

### **2. Precision Parameters**
*(Note: The GNN specification does not explicitly define precision parameters like \( \gamma \) or \( \alpha \). These are inferred from typical Active Inference conventions or may be omitted if fixed.)*

- **\( \gamma \) (Gamma)**: Precision parameter for variational inference.
  - **Role**: Controls the trade-off between data fidelity and model complexity in variational free energy minimization.
  - **Typical Values**: Often set to \( \gamma = 1 \) for discrete models (as in this example).
- **\( \alpha \) (Alpha)**: Learning rate for policy adaptation.
  - **Role**: Determines how quickly the agent updates its policy based on expected free energy.
  - **Typical Values**: Not specified; may be inferred from \( E \) (habit) or \( C \) (preferences).

---
### **3. Dimensional Parameters**
- **State Space**:
  - \( \text{num\_hidden\_states} = 3 \) (discrete).
  - \( s \in \{0,1,2\} \): Current hidden state distribution.
- **Observation Space**:
  - \( \text{num\_obs} = 3 \) (discrete).
  - \( o \in \{0,1,2\} \): Current observation.
- **Action Space**:
  - \( \text{num\_actions} = 3 \) (discrete).
  - \( u \in \{0,1,2\} \): Chosen action.

---

### **4. Temporal Parameters**
- **Time Horizon**:
  - \( \text{ModelTimeHorizon} = \text{Unbounded} \): Agent operates indefinitely.
  - **Discrete Time**: \( t \in \mathbb{N} \) (e.g., \( t = 0, 1, 2, \dots \)).
- **Update Frequencies**:
  - **Belief Update**: Triggered by observations (via \( A \) and \( B \)).
  - **Policy Update**: Triggered by expected free energy minimization (via \( C \) and \( E \)).
  - **No Deep Planning**: Only 1-step lookahead (as per the GNN footer).

---

### **5. Initial Conditions**
- **Prior Beliefs**:
  - \( D = (0.333, 0.333, 0.333) \): Uniform prior over hidden states.
- **Initial Parameter Values**:
  - \( A \), \( B \), \( C \), \( E \): Explicitly defined in the GNN specification.
- **Initialization Strategy**:
  - **Belief**: \( s = D \) (uniform distribution over hidden states).
  - **Policy**: \( \pi = E \) (uniform distribution over actions).

---

### **6. Configuration Summary**
#### **Parameter File Format Recommendations**
- **Structured JSON/YAML**: For readability and extensibility.
  ```yaml
  Model:
    A: [0.9, 0.05, 0.05; 0.05, 0.9, 0.05; 0.05, 0.05, 0.9]
    B: [ [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0] ]  # Action 0
    C: [0.1, 0.1, 1.0]
    D: [0.333, 0.333, 0.333]
    E: [0.333, 0.333, 0.333]
    Precision:
      gamma: 1.0
  ```

#### **Tunable vs. Fixed Parameters**
| Parameter       | Tunable? | Notes                                  |
|-----------------|----------|----------------------------------------|
| \( A \)         | No       | Hardcoded identity mapping.            |
| \( B \)         | No       | Hardcoded deterministic transitions.   |
| \( C \)         | Yes      | Log-preferences can be adjusted.       |
| \( D \)         | No       | Uniform prior fixed.                  |
| \( E \)         | No       | Uniform habit fixed.                  |
| \( \gamma \)    | Yes      | Precision parameter (default: 1.0).    |

#### **Sensitivity Analysis Priorities**
1. **Preference Vector (\( C \))**: Critical for intrinsic motivation; small changes can drastically alter policy.
2. **Transition Matrix (\( B \))**: Core dynamics; misalignment here breaks deterministic transitions.
3. **Precision (\( \gamma \))**: Affects belief updating fidelity; may need tuning for discrete models.

---
### **Key Active Inference Implications**
1. **Variational Free Energy (\( F \))**: Used to compute posterior beliefs over hidden states, balancing data fidelity and model complexity.
2. **Expected Free Energy (\( G \))**: Minimized to infer policies, aligning with preferences (\( C \)) and habits (\( E \)).
3. **Belief Propagation**: Coupled with \( A \) and \( B \) to update hidden state distributions after observations.
4. **Deterministic Dynamics**: The \( B \) matrix’s deterministic structure simplifies planning but limits flexibility in real-world POMDPs.