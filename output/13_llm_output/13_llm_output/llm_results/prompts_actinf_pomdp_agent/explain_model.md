# EXPLAIN_MODEL

Based on the document, here is a concise outline of what I will cover:

**Model Purpose:** This agent represents an active inference POMDP that models a discrete environment with three types of actions (decision to explore, exploration towards another location, and return), one mode of behavior (exploration through different locations), 2 states per action ($action_1$-forward policy) and 3 modes of behavior (control $=>x$, $<0.95$-influence), two actions are taken when exploring/returning.

**Core Components:**
  - **S**: (i,ii) Likelihood Matrix representing the probability distribution over action outcomes ($x\leq \frac{1}{3}$).
  - **B**: (j,k) Transition Matrix describing transition probabilities $(\xi_j$, $\mu_{ij})$. These represent actions chosen by the agent.
    - **g**: (l-v)(i=2n): Observation distribution over a single level of exploration; see also [1]
  - **H**: (a,b) Prior distribution representing prior probabilities $(\beta_j),$ for all actions $(x\leq \frac{1}{3}$).
    - A: (k-u)(i=2n): Observation distribution over a single level of exploration; see also [1]
  - **T**: (a,b) Belief vector representing initial belief prior ($v_{t}^*) for the current state.
    - T: (j+m) (actions taken), which is the probability of action $x$ chosen by agent $i$.
    - B(pi_): (k-s)(next actions/observations). See also [1]
  - **Q**: (n,e)(f^*) Probability vector representing preferences over observed beliefs.
    - Q: (l-v) (observation outcomes), which is the probability of observed belief $g_{\eta}(x)$ ($u$) chosen by agent $i$.
**Model Dynamics:**
  - **F(s_1, s_2n): Initial Policy** : A policy vector $A$, a sequence of action choices that are randomly assigned to each observation.
  - **G(θ)\rangle**: Probability distribution over actions $\{g_{\eta}(x)|x=t$). These represent the probability distributions over observations and beliefs, respectively.
  - **B(\xi_1) \otimes B(\xi_2)`: (i,j), a sequence of beliefs that are initialized uniformly over each observation ($h(s_1)\rangle$, $H(θ)$), represented by their joint probabilities in the last layer of belief vector.
**Active Inference Context:**
  - **G**: [3] A prior distribution representing initial guess policies and actions, represented as a sequence of beliefs (first 2 layers). These represent states chosen when exploring/returning with actions $x_1$-directed and $x_2$.
  - **B(θ)\rangle * G** : Probability distributions for the first layer and last layer.
The agent's initial policy is represented as a sequence of beliefs (first 3 layers).
These are initialized uniformly over each observation ($h_{\eta}(s)$, $H_{0}^{i}^\prime$), but in reality, it would be possible to do so with probabilistic graphical models and other types of explicit inference protocols.
It is not only a POMDP agent but also an active inference agent (and thus a generative model).
We can generate random actions for the agents by training or generating new observations/observations based on previous ones, i.e., by sampling from current policy / beliefs and updating it using belief update in each layer of belief vector; so no planning is performed here except when acting upon previously collected data to retrieve observed outcomes (see [1] above).
There are various types of actions available:
   - **F** (choices) 
       - **B(θ)\rangle * G**: Probability distribution over beliefs $G$ in the first layer and last layer. It represents all possible values for hypothesis states $\{x_i^*\}_{j}$, represented by their joint probabilities across layers. These are initialized uniformly on observations of each observation ($h_{\eta}(s)$, $H_{0}^{i}^\prime$), but in reality there could be many such beliefs associated with the same observed data points to achieve good generalization and prediction accuracy.
- **I** (choices) 
       - **F(θ)\rangle * B(**G**, i=2n): Probability distribution over actions $\{g_{\eta}(x_i^*)^{1/3} := \beta_t^\prime(\xi_j^*x_i)^*$). These represent the probability distributions of different choices.
- **E** (choices)
       - **G(θ)\rangle * B(**B**, i=2n): Probability distribution over actions $\{g_{\eta}(x_1^\prime x_2)^* := \beta_t^*(h_{\eta}(\xi_i))^{-1}$). These represent the probability distributions of different choices.
These are initialized uniformly on observations, but in reality there could be many such beliefs associated with the same observed data points to achieve good generalization and prediction accuracy.
There are various types of actions available:
   - **I** (choices)
       - **F(θ)\rangle * G(**B**, i=2n): Probability distribution over actions $\{g_{\eta}(x_i^*)^{1/3} := \beta_t^\prime(\xi_j^*x_i)^*(h_{\eta}(\xi_j))^{-1}$). These represent the probability distributions of different choices.
These are initialized uniformly on observations, but in reality there could be many such beliefs associated with the same observed data points to achieve good generalization and prediction accuracy.
There are various types of actions available:
   - **I** (choices)
       - **F(θ)\rangle * B(**G**, i=2n): Probability distribution over actions $\{g_{\eta}(x_1^\prime x_2)^* := \beta_t^*(h_{\eta}(\xi_j))^{-1}$). These represent the probability distributions of different choices.
These are initialized uniformly on observations, but in reality there could be many such beliefs associated with the same observed data points to achieve good generalization and prediction accuracy.