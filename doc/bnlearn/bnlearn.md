# Comprehensive bnlearn Guide

`bnlearn` is a streamlined and highly configurable Python library for Causal Discovery, Structure Learning, Parameter Learning, and Inference in Bayesian Networks. Built on top of robust foundations like `pgmpy`, it significantly simplifies the pipeline for constructing, evaluating, and applying Probabilistic Graphical Models (PGMs) to complex datasets.

## 1. Structure Learning

Structure learning is the process of generating a Directed Acyclic Graph (DAG) that accurately represents the conditional dependencies within a dataset. `bnlearn` offers multiple algorithmic approaches to achieve this:

*   **Exhaustive Search (`ex`)**: Evaluates every possible DAG. Only feasible for very small networks (typically $N \le 5$ nodes) due to super-exponential complexity.
*   **Hill-Climb Search (`hc`)**: A greedy local search algorithm starting from an empty, full, or random graph, iteratively adding, removing, or reversing edges to optimize a scoring metric.
*   **Chow-Liu (`cl`)**: A tree-based algorithm that finds the maximum likelihood tree structure. Very fast, but restricted to tree-like structures where each node has at most one parent.
*   **Tree-augmented Naive Bayes (TAN)**: Extends Naive Bayes by allowing edges between features, forming a tree over the features rooted at the class variable. Useful for classification tasks.
*   **Constraint-based Search**: Algorithms like PC that use statistical independence tests to determine network structure rather than a scoring metric.

### Code Example:
```python
import bnlearn as bn
df = bn.import_example('sprinkler')

# Evaluate using Hill-Climb with BIC score
model_hc_bic = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')

# Evaluate using Chow-Liu starting from a root node
model_cl = bn.structure_learning.fit(df, methodtype='cl', root_node='Wet_Grass')
```

## 2. Parameter Learning

Once the structure (DAG) is defined, parameter learning calculates the Conditional Probability Distributions (CPDs) for each node based on the data. 

*   **Maximum Likelihood Estimation (MLE)**: Calculates probabilities directly from the frequencies observed in the data. Can struggle with sparse data if certain combinations are unobserved (assigning probability 0).
*   **Bayesian Parameter Estimation**: Uses a prior distribution (typically a Dirichlet prior, effectively acting as pseudo-counts) alongside the data, smoothing probabilities and avoiding zero-probability assignments for unseen combinations.

### Code Example:
```python
import bnlearn as bn
df = bn.import_example()
model_structure = bn.import_DAG('sprinkler', CPD=False) # Get empty graph
model_update = bn.parameter_learning.fit(model_structure, df, methodtype='bayes')
```

## 3. Inference

Inference calculates the marginal or conditional probabilities of specific variables (queries) given observed values for other variables (evidence). Using junction tree algorithms, `bnlearn` performs exact inference efficiently.

### Code Example:
```python
import bnlearn as bn
model = bn.import_DAG('sprinkler')

query = bn.inference.fit(
    model, 
    variables=['Rain'], 
    evidence={'Wet_Grass': 1, 'Cloudy': 0}
)
print(query.df)
```

## 4. Plotting and Visualization

`bnlearn` integrates seamlessly with `networkx` to plot DAGs. It also offers advanced Interactive Plotting (e.g., generating D3.js HTML graphs) and allows visual comparison of networks to identify structural differences instantly.

### Code Example:
```python
import bnlearn as bn
df = bn.import_example('sprinkler')
model = bn.structure_learning.fit(df)

# Standard Plot
bn.plot(model)

# Interactive Plot
bn.plot(model, interactive=True)
```

## 5. Other Functionalities

*   **Discretizing**: Continuous datasets must often be transformed into categorical bounds. `bnlearn.discretize()` handles automatic density partitioning or manual bounded discretizations.
*   **Synthetic Data Generation**: `bn.sampling()` leverages learned models to perform Forward or Gibbs sampling, enabling rich synthetic data generation that preserves conditional dependencies.
*   **Imputation**: `bn.knn_imputer()` to resolve missing fields securely prior to structural learning.

For complete API routes, see the official [BNLearn’s Documentation](https://erdogant.github.io/bnlearn/pages/html/index.html).
