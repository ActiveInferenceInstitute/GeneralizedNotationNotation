# EXPLAIN_MODEL

This GNN example represents a Markov Chain Monte Carlo (MCMC) algorithm for estimating the probability of observing a sequence of states and actions based on a set of observed observations. The model is composed of 4 hidden states (`s`) with transition matrices (`P(o)`), while 6 observation symbols (`O`).

The model's core components are:

1. **Hidden Markov Model Baseline**: A discrete Markov Chain (DMC) that models the evolution of a sequence of states and actions based on observed observations. The DMC consists of 4 hidden states (`s`) with transition matrices (`P(o)`).

2. **Random Walks**: A stochastic process that generates random samples from the DMC's state space, allowing for inference about future states and actions. Each step in the Markov Chain corresponds to a sequence of observations (actions) being observed.

3. **Forward Algorithm**: The algorithm iteratively updates the belief distribution (`B`) based on the observed outcomes (`o`), while backward algorithms update the beliefs (`F`) using the observed outcomes (`s_prime`, `O_m0`, and `O_m1`.

4. **State Posterior**: The model estimates the probability of observing a sequence of states and actions by updating the belief distribution based on the observed outcomes. This process is repeated for each observation, allowing for inference about future states and actions.

The key relationships between the hidden states (`s`) and observable states (`o`) are:

1. **Forward Algorithm**: The algorithm iteratively updates the belief distribution (`B`) based on the observed outcomes (`o`). This process is repeated for each observation, allowing for inference about future states and actions.

2. **Backward Algorithm**: The algorithm iteratively updates the beliefs (`F`) using the observed outcomes (`s_prime`, `O_m0`, and `O_m1`.

The model's predictions are based on a probabilistic graphical model that incorporates the learned beliefs (`B`) from previous steps, allowing for inference about future states and actions. The predictions can be used to make decisions in uncertain environments or predict outcomes with high confidence.

This GNN provides an example of how to implement Active Inference principles using a Markov Chain Monte Carlo (MCMC) algorithm. It demonstrates the use of a probabilistic graphical model, allowing for inference about future states and actions based on observed outcomes. The model's predictions are used as benchmarks in other Active Inference variants