# GNN Resource Estimation Report

Analyzed 1 files
Average Memory Usage: 0.23 KB
Average Inference Time: 95.82 units
Average Storage: 3.45 KB

## actinf_pomdp_agent.md
Path: ../input/gnn_files/actinf_pomdp_agent.md
Memory Estimate: 0.23 KB
Inference Estimate: 95.82 units
Storage Estimate: 3.45 KB

### Model Info
- variables_count: 12
- edges_count: 8
- time_spec: Dynamic
- equation_count: 4

### Complexity Metrics
- state_space_complexity: 5.9307
- graph_density: 0.0606
- avg_in_degree: 1.1429
- avg_out_degree: 1.0000
- max_in_degree: 2.0000
- max_out_degree: 1.0000
- cyclic_complexity: 0.0000
- temporal_complexity: 0.0000
- equation_complexity: 12.8750
- overall_complexity: 10.0000
- variable_count: 12.0000
- edge_count: 8.0000
- total_state_space_dim: 60.0000
- max_variable_dim: 27.0000

# Metric Definitions

## General Metrics
- **Memory Estimate (KB):** Estimated RAM required to hold the model's variables and data structures in memory. Calculated based on variable dimensions and data types (e.g., float: 4 bytes, int: 4 bytes).
- **Inference Estimate (units):** A relative, abstract measure of computational cost for a single inference pass. It is derived from factors like model type (Static, Dynamic, Hierarchical), the number and type of variables, the complexity of connections (edges), and the operations defined in equations. Higher values indicate a more computationally intensive model. These units are not tied to a specific hardware time (e.g., milliseconds) but allow for comparison between different GNN models.
- **Storage Estimate (KB):** Estimated disk space required to store the model file. This includes the memory footprint of the data plus overhead for the GNN textual representation, metadata, comments, and equations.

## Complexity Metrics (scores are generally relative; higher often means more complex)
- **state_space_complexity:** Logarithmic measure of the total dimensionality of all variables (sum of the product of dimensions for each variable). Represents the model's theoretical information capacity or the size of its state space.
- **graph_density:** Ratio of actual edges to the maximum possible edges in the model graph. A value of 0 indicates no connections, while 1 would mean a fully connected graph. Measures how interconnected the variables are.
- **avg_in_degree:** Average number of incoming connections (edges) per variable.
- **avg_out_degree:** Average number of outgoing connections (edges) per variable.
- **max_in_degree:** Maximum number of incoming connections for any single variable in the model.
- **max_out_degree:** Maximum number of outgoing connections for any single variable in the model.
- **cyclic_complexity:** A score indicating the presence and extent of cyclic patterns or feedback loops in the graph. Approximated based on the ratio of edges to variables; higher values suggest more complex recurrent interactions.
- **temporal_complexity:** Proportion of edges that involve time dependencies (e.g., connecting a variable at time `t` to one at `t+1`). Indicates the degree to which the model's behavior depends on past states or sequences.
- **equation_complexity:** A measure based on the average length, number, and types of mathematical operators (e.g., +, *, log, softmax) used in the model's equations. Higher values suggest more intricate mathematical relationships between variables.
- **overall_complexity:** A weighted composite score (typically scaled, e.g., 0-10) that combines state space size, graph structure (density, cyclicity), temporal aspects, and equation complexity to provide a single, holistic measure of the model's intricacy.
