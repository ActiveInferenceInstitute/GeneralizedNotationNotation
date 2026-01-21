# EXTRACT_PARAMETERS

Here is a **systematic breakdown** of the parameters in the provided **Active Inference POMDP Agent GNN specification**, organized by conceptual categories with rigorous mathematical and semantic interpretations:

---

### **1. Model Matrices**
#### **(A) Likelihood Matrix**
- **Dimensions**: \( A \in \mathbb{R}^{3 \times 3} \) (observation outcomes × hidden states).
- **Structure**:
  - Rows index observations (0, 1, 2), columns index hidden states (0, 1, 2).
  - Deterministic mapping: Each hidden state \( s \) maps to a unique observation \( o \) via \( A_{o,s} \).
  - Given \( s \), \( o \) is sampled as \( o = \argmax_{o'} A_{o',s} \).
- **Interpretation**:
  - \( A_{o,s} \) = \( P(o|s) \) (likelihood of observation \( o \) given hidden state \( s \)).
  - **Initialization**: Identity mapping (e.g., \( A_{0,0} = 0.9 \), \( A_{1,1} = 0.9 \), etc.), implying observations are deterministic functions of hidden states.

#### **(B) Transition Matrix**
- **Dimensions**: \( B \in \mathbb{R}^{3 \times 3 \times 3} \) (next states × previous states × actions).
- **Structure**:
  - Each slice \( B_{a,:,:} \) (for action \( a \)) is a \( 3 \times 3 \) matrix where rows index previous states and columns index next states.
  - Deterministic transitions: \( s' = \argmax_{s'} B_{a,s,s'} \).
- **Interpretation**:
  - \( B_{a,s,s'} \) = \( P(s'|s,a) \) (transition probability from state \( s \) to \( s' \) via action \( a \)).
  - **Initialization**: Cyclic transitions (e.g., action 0 moves from state 0→1, action 1 from 1→2, action 2 from 2→0), forming a loop.
- **Key Property**: No stochasticity; actions deterministically update hidden states.

#### **(C) Preference Vector**
- **Dimensions**: \( C \in \mathbb{R}^3 \) (observation outcomes).
- **Structure**:
  - \( C_o \) = log-preference for observation \( o \).
- **Interpretation**:
  - \( C \) encodes the agent’s intrinsic motivation: higher \( C_o \) means observation \( o \) is more rewarding.
  - **Initialization**: \( C = (0.1, 0.1, 1.0) \), implying observation 2 is most preferred (log-probability = 1.0), others are weakly preferred.

#### **(D) Prior Vector**
- **Dimensions**: \( D \in \mathbb{R}^3 \) (hidden states).
- **Structure**:
  - \( D_s \) = prior probability of hidden state \( s \).
- **Interpretation**:
  - \( D \) encodes initial beliefs about hidden states.
  - **Initialization**: Uniform prior \( D = (0.333, 0.333, 0.333) \), implying no prior bias.

---

### **2. Precision Parameters**
*(Note: The GNN specification does not explicitly define precision parameters like \( \gamma \) or \( \alpha \). These are inferred from the variational free energy framework and typical Active Inference conventions.)*

- **\( \gamma \) (Precision Parameter)**:
  - In variational inference, \( \gamma \) scales the precision of the variational distribution over hidden states.
  - **Role**: Controls how tightly the variational distribution approximates the true posterior.
  - **Typical Values**: Often set via learning rates or hyperparameters (e.g., \( \gamma \propto \text{learning rate} \)).
  - **Missing in GNN**: Not explicitly defined here; may be inferred from \( F \) (variational free energy) updates.

- **\( \alpha \) (Adaptation Parameter)**:
  - In Active Inference, \( \alpha \) may refer to:
    - **Learning rate** for updating parameters (e.g., \( A, B, C \)).
    - **Temporal decay** in belief updates (e.g., \( \alpha = 1 - \text{decay rate} \)).
  - **Missing in GNN**: Not specified; could be inferred from \( G \) (expected free energy) updates.

- **Other Parameters**:
  - **Confidence Intervals**: Not explicitly modeled; variational free energy \( F \) implicitly encodes uncertainty.
  - **Regularization**: Not present; the model is unregularized.

---

### **3. Dimensional Parameters**
| Parameter               | Dimensions          | Description                                                                 |
|-------------------------|---------------------|-----------------------------------------------------------------------------|
| **Hidden States**       | \( s \in \{0,1,2\} \) | 3 discrete states.                                                        |
| **Observations**         | \( o \in \{0,1,2\} \) | 3 discrete observation outcomes.                                           |
| **Actions**              | \( u \in \{0,1,2\} \) | 3 discrete actions.                                                        |
| **Time Steps**           | \( t \in \mathbb{N} \) | Discrete time (unbounded horizon).                                         |
| **Policy Distribution**  | \( \pi \in \mathbb{R}^3 \) | Softmax distribution over actions (no planning).                          |

---

### **4. Temporal Parameters**
- **Time Horizon**:
  - **Model Definition**: Unbounded (\( \text{ModelTimeHorizon} = \text{Unbounded} \)).
  - **Simulation**: May be truncated to finite horizons (e.g., \( T = 10 \) steps).
- **Temporal Dependencies**:
  - **Markov Property**: Hidden states \( s \) and observations \( o \) are Markovian (no memory beyond immediate state/action).
  - **No Hierarchy**: No nested or hierarchical dependencies (e.g., no sub-policies).
- **Update Frequencies**:
  - **Belief Updates**: Occur after each observation \( o \) via variational free energy \( F \).
  - **Policy Updates**: Occur via expected free energy \( G \) (no deep planning).

---

### **5. Initial Conditions**
| Parameter       | Value               | Description                                                                 |
|-----------------|---------------------|-----------------------------------------------------------------------------|
| **Prior Beliefs** | \( D = (0.333, 0.333, 0.333) \) | Uniform initial beliefs over hidden states.                              |
| **Initial Policy** | \( E = (0.333, 0.333, 0.333) \) | Uniform habit (no bias toward any action).                                |
| **Initial Likelihoods** | \( A \) as identity mapping | Deterministic observation mapping.                                         |
| **Initial Transitions** | Cyclic transitions | \( B \) enforces deterministic loops.                                      |
| **Initial Preferences** | \( C = (0.1, 0.1, 1.0) \) | Observation 2 is most preferred.                                           |

---

### **6. Configuration Summary**
#### **Parameter File Format Recommendations**
- **YAML/JSON**: Preferred for readability and extensibility.
  ```yaml
  # Example YAML snippet:
  A:
    - [0.9, 0.05, 0.05]  # Observation 0
    - [0.05, 0.9, 0.05]  # Observation 1
    - [0.05, 0.05, 0.9]  # Observation 2
  B:
    - [ [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0] ]  # Action 0
    - ...
  C: [0.1, 0.1, 1.0]
  D: [0.333, 0.333, 0.333]
  E: [0.333, 0.333, 0.333]
  ```

#### **Tunable vs. Fixed Parameters**
| Category               | Tunable Parameters                          | Fixed Parameters                          |
|------------------------|---------------------------------------------|------------------------------------------|
| **Model Structure**    | None (fully specified)                      | All matrices \( A, B, C, D, E \)         |
| **Precision**          | \( \gamma \), \( \alpha \) (inferred)       | None                                     |
| **Initialization**     | None (fixed)                               | All initial values \( D, E, A, B, C \)   |
| **Temporal Dynamics**  | None (Markovian)                           | Unbounded horizon, deterministic \( B \) |

#### **Sensitivity Analysis Priorities**
1. **Precision Parameters**:
   - How \( \gamma \) affects variational approximation quality.
   - How \( \alpha \) affects learning dynamics (e.g., convergence speed).
2. **Initial Conditions**:
   - Impact of \( D \) (prior beliefs) on belief updates.
   - Impact of \( E \) (initial policy) on exploration vs. exploitation.
3. **Deterministic Transitions**:
   - Robustness to perturbations in \( B \) (e.g., stochasticizing \( B \)).
4. **Preference Vector \( C \)**:
   - Sensitivity to changes in \( C \) (e.g., \( C = (1.0, 0.1, 0.1) \) vs. \( (0.1, 0.1, 1.0) \)).

---
### **Key Takeaways**
- The model is **deterministic** (no stochasticity in \( A, B \)) and **Markovian** (no memory beyond immediate state/action).
- **Variational free energy \( F \)** implicitly encodes uncertainty, but precision parameters \( \gamma \) and \( \alpha \) are not explicitly defined.
- **Initial conditions** are fixed but could be tuned for exploration (e.g., non-uniform \( D \) or \( E \)).
- **Sensitivity to \( B \)**: The cyclic transitions may limit flexibility; stochasticizing \( B \) could improve robustness.