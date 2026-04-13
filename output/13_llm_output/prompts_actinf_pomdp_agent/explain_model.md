# EXPLAIN_MODEL

Here is a concise overview of the GNN Model:

**Model Purpose:** 
1. Represent real-world phenomena (e.g., decision making) and provide insights into their behavior.
2. Represent hidden states (s_f0, s_f1, etc.) using probability distributions.
3. Provide a framework for planning actions based on observed observations.
4. Implement Active Inference principles in the model:
- **State inference**: Use Variational Free Energy to update beliefs and estimate probabilities of observing new data.
- **Policy inference**: Use Expected Free Energy (E) to update policies and estimate expected values.
- **Action selection**: Use Action Selection from Policy Prior (A) to update actions based on observed observations.
5. **Model dynamics**: Implement Active Inference principles:
- **State evolution**: Update beliefs using Variational Free Energy (VFE), E, or A/B (Bayesian inference).
- **Action selection**: Use Action Selection from Policy Prior (A) to update actions based on observed observations.
6. **Practical implications**: 
1. **Estimate probabilities of observing new data**: Update beliefs using VFE and A/B.
2. **Make decisions**: Use A/B or Bayesian inference to make predictions about future outcomes.
3. **Predict future behavior**: Use A/B or Bayesian inference to predict the outcome of a given action.