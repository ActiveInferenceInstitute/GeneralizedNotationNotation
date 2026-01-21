# Active Inference Glossary

> **📋 Document Metadata**  
> **Type**: Reference | **Audience**: All | **Complexity**: Beginner  
> **Cross-References**: [FEP Foundations](fep_foundations.md) | [Active Inference Theory](active_inference_theory.md)

## Overview

This glossary provides definitions for key terms in Active Inference and the Free Energy Principle.

---

## A

### A Matrix
**Definition**: The likelihood matrix P(o|s) mapping hidden states to observations.  
**Dimensions**: [num_observations, num_states]  
**Also known as**: Observation model, likelihood mapping

### Action
**Definition**: The agent's influence on the environment, selected to minimize expected free energy.  
**Symbol**: a, π (policy)

### Active Inference
**Definition**: A process theory derived from the Free Energy Principle describing how agents perceive, act, and learn by minimizing (expected) free energy.

### Approximate Posterior
**Definition**: The variational distribution Q(s) used to approximate the true posterior P(s|o).  
**Symbol**: Q(s)

---

## B

### B Matrix
**Definition**: The transition matrix P(s'|s,a) describing state dynamics under actions.  
**Dimensions**: [num_states, num_states, num_actions]  
**Also known as**: Transition model, dynamics model

### Bayesian Inference
**Definition**: Statistical inference using Bayes' theorem to update beliefs given evidence.

### Belief
**Definition**: The agent's probability distribution over hidden states.  
**Symbol**: Q(s), b(s)

### Belief State
**Definition**: A probability distribution over hidden states representing the agent's uncertainty.

---

## C

### C Matrix/Vector
**Definition**: Log prior preferences over observations, driving goal-directed behavior.  
**Dimensions**: [num_observations] or [num_observations, T]  
**Also known as**: Preference vector, goal specification

### Complexity
**Definition**: The KL divergence between posterior and prior beliefs: D_KL(Q(s)||P(s)).  
**Role**: Regularization term in free energy

---

## D

### D Vector
**Definition**: Prior probability over initial states P(s₀).  
**Dimensions**: [num_states]  
**Also known as**: Initial state prior

---

## E

### E Vector
**Definition**: Prior probability over policies P(π).  
**Dimensions**: [num_policies]  
**Also known as**: Habit prior, policy prior

### Entropy
**Definition**: Measure of uncertainty in a distribution: H(P) = -Σ P(x) log P(x).  
**Symbol**: H

### Epistemic Value
**Definition**: Expected information gain about hidden states; drives exploration.  
**Formula**: I(s;o|π) = H(s|π) - H(s|o,π)

### Expected Free Energy (EFE)
**Definition**: The quantity minimized for action selection, balancing pragmatic and epistemic value.  
**Symbol**: G(π)  
**Formula**: G(π) = -Pragmatic + Epistemic

---

## F

### Factor Graph
**Definition**: Graphical model representing factorization of probability distributions.

### Free Energy Principle (FEP)
**Definition**: The proposal that self-organizing systems minimize variational free energy.

---

## G

### Generative Model
**Definition**: The agent's model of how observations are generated from hidden states.  
**Components**: A, B, C, D, E matrices

### GNN (Generalized Notation Notation)
**Definition**: A standardized language for specifying Active Inference generative models.

---

## I

### Inaccuracy
**Definition**: Negative expected log-likelihood: -E_Q[log P(o|s)].  
**Role**: Data fit term in free energy

### Information Gain
**Definition**: Mutual information between future states and observations.  
**Symbol**: I(s;o)

### Inference
**Definition**: The process of updating beliefs given observations.

---

## K

### KL Divergence
**Definition**: Measure of difference between distributions: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x)).  
**Symbol**: D_KL

---

## L

### Likelihood
**Definition**: The probability of observations given states: P(o|s).  
**Matrix**: A

### Log Preferences
**Definition**: The C vector expressing preferred observations in log-probability form.

---

## M

### Markov Blanket
**Definition**: Statistical boundary separating internal from external states.

### Message Passing
**Definition**: Algorithm for inference on graphical models via local message exchange.

### Model Evidence
**Definition**: The marginal likelihood P(o|m) of observations under model m.

---

## O

### Observation
**Definition**: Sensory data received by the agent.  
**Symbol**: o

---

## P

### Policy
**Definition**: A sequence of actions over time.  
**Symbol**: π

### POMDP
**Definition**: Partially Observable Markov Decision Process—mathematical framework for sequential decision-making under uncertainty.

### Posterior
**Definition**: The updated belief after observing data: P(s|o).

### Pragmatic Value
**Definition**: Expected alignment of predicted observations with preferences; drives goal-seeking.  
**Formula**: -D_KL(Q(o|π)||P(o))

### Precision
**Definition**: Inverse variance; confidence in beliefs or preferences.  
**Symbol**: γ, β, π (context-dependent)

### Prior
**Definition**: Belief before observing data: P(s).

---

## S

### Softmax
**Definition**: Function converting values to probabilities: σ(x)_i = exp(x_i)/Σ exp(x_j).

### State
**Definition**: Hidden variables generating observations.  
**Symbol**: s

### Surprise
**Definition**: Negative log probability of observations: -log P(o|m).  
**Also known as**: Surprisal, self-information

---

## T

### Transition
**Definition**: State dynamics under actions: P(s'|s,a).  
**Matrix**: B

---

## V

### Variational Free Energy (VFE)
**Definition**: Upper bound on surprise used for tractable inference.  
**Symbol**: F  
**Formula**: F = D_KL(Q(s)||P(s|o)) - log P(o|m)

### Variational Inference
**Definition**: Approximating intractable posteriors by minimizing KL divergence.

---

## Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| s | Hidden state |
| o | Observation |
| a | Action |
| π | Policy |
| Q(s) | Approximate posterior (beliefs) |
| P(s) | Prior over states |
| P(o\|s) | Likelihood (A matrix) |
| P(s'\|s,a) | Transition (B matrix) |
| C | Log preferences |
| F | Variational Free Energy |
| G | Expected Free Energy |
| γ | Precision |
| H | Entropy |
| D_KL | KL Divergence |

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards
