# EXPLAIN_MODEL

Here's a concise summary of the key points:

**Model Purpose:** This is an active inference model that represents a simple Markov chain with 4 hidden states and 6 observation symbols. It encodes the probability distribution over possible actions (actions = [action1, action2]) based on past observations (observations) and future predictions (forward and backward). The goal is to estimate the next state of the system using forward and backward updates in time.

**Core Components:**

1. **Hidden states**: 4 hidden states with Markovian dynamics:
   - `s_f0`: Initial state
   - `s_f1`: Forward history (observation)
   - `o_m0`: Forward history (action)
   - `u_c0`: Forward history (belief)

2. **Observations**: 6 observation symbols with fixed transition and emission matrices:
   - `a` for action A, etc.,
   - `b` for action B, etc.,
   - `d` for action D
3. **Forward History**: A stochastic initial state distribution (P(o_t|s) = P(o_{t+1}|s') * P(s|s', s'')) with probabilities of each observation:
   - `p_a`: Probability of action A, etc.,
   - `p_b`: Probability of action B, etc.
4. **Forward History**: A stochastic forward history (P(o_{t+1}|s') = P(o_{t+1}|s', s'')) with probabilities of each observation:
   - `p_a` and `p_b`: Probability of action A or B, respectively
5. **Backward History**: A stochastic backward history (P(o_{t-1}|s') = P(o_{t-1}|s', s')) with probabilities of each observation:
   - `p_a` and `p_b`: Probability of action A or B, respectively
6. **Forward and Backward Belief**: A belief distribution (B) that updates the beliefs based on past predictions (`u_c0`) and actions (`π_c0`, etc.) for each state:
   - `P(o_{t+1}|s')` and `P(o_{t-1}|s')`: Probability of action A or B, respectively
7. **Forward and Backward Belief**: A