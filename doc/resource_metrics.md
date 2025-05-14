# GNN Resource Estimation Metrics

This document describes the metrics used in the GNN Resource Estimation Report, typically found at `output/gnn_type_check/resource_estimates/resource_report.md`. These metrics help in understanding the computational requirements and complexity of GNN models.

## General Metrics

- **Memory Estimate (KB):** Estimated RAM required to hold the model's variables and data structures in memory. Calculated based on variable dimensions and data types (e.g., float: 4 bytes, int: 4 bytes).
    - *Calculation Details:* Sum of (product of dimensions * bytes_per_type) for all variables.

- **Inference Estimate (units):** A relative, abstract measure of computational cost for a single inference pass. It is derived from factors like model type (Static, Dynamic, Hierarchical), the number and type of variables, the complexity of connections (edges), and the operations defined in equations. Higher values indicate a more computationally intensive model. These units are not tied to a specific hardware time (e.g., milliseconds) but allow for comparison between different GNN models.
    - *Calculation Details:* Base cost from variable count and dimensions, scaled by model type (Dynamic/Hierarchical are more expensive) and average data type cost. Further influenced by equation complexity and graph density.

- **Storage Estimate (KB):** Estimated disk space required to store the model file. This includes the memory footprint of the data plus overhead for the GNN textual representation, metadata, comments, and equations.
    - *Calculation Details:* Primarily based on the raw GNN file size, potentially augmented by estimates of textual representation of parsed data if the raw file is not directly used.

## Complexity Metrics

*(Scores are generally relative; higher often means more complex. Some are normalized, others are direct counts or ratios.)*

- **state_space_complexity:** Logarithmic measure of the total dimensionality of all variables (sum of the product of dimensions for each variable). Represents the model's theoretical information capacity or the size of its state space.
    - *Calculation Details:* `log2(sum(Π(dims_i)))` for each variable `i`.

- **graph_density:** Ratio of actual edges to the maximum possible edges in the model graph (for `N` variables, max edges is `N*(N-1)` for a directed graph). A value of 0 indicates no connections, while 1 would mean a fully connected graph. Measures how interconnected the variables are.
    - *Calculation Details:* `num_edges / (num_variables * (num_variables - 1))` if `num_variables > 1`, else 0.

- **avg_in_degree:** Average number of incoming connections (edges) per variable.
    - *Calculation Details:* `total_in_degrees / num_variables`.

- **avg_out_degree:** Average number of outgoing connections (edges) per variable.
    - *Calculation Details:* `total_out_degrees / num_variables`.

- **max_in_degree:** Maximum number of incoming connections for any single variable in the model.

- **max_out_degree:** Maximum number of outgoing connections for any single variable in the model.

- **cyclic_complexity:** A score indicating the presence and extent of cyclic patterns or feedback loops in the graph. Approximated based on the ratio of edges to variables; higher values suggest more complex recurrent interactions. (This is a heuristic).
    - *Calculation Details:* Often `(num_edges - num_variables + num_connected_components)` if based on McCabe's, or a simpler heuristic.

- **temporal_complexity:** Proportion of edges that involve time dependencies (e.g., connecting a variable at time `t` to one at `t+1`). Indicates the degree to which the model's behavior depends on past states or sequences.
    - *Calculation Details:* `num_temporal_edges / total_edges`.

- **equation_complexity:** A measure based on the average length, number, and types of mathematical operators (e.g., +, *, log, softmax) used in the model's equations. Higher values suggest more intricate mathematical relationships between variables.
    - *Calculation Details:* Sum of (operator_count * operator_cost_factor) across all equations, possibly normalized by number of equations or variables.

- **overall_complexity:** A weighted composite score (typically scaled, e.g., 0-10) that combines state space size, graph structure (density, cyclicity), temporal aspects, and equation complexity to provide a single, holistic measure of the model's intricacy.
    - *Calculation Details:* A weighted sum of normalized versions of other complexity metrics. Weights and normalization method are specific to the estimator's implementation.

- **variable_count:** Total number of unique variables defined in the `StateSpaceBlock`.

- **edge_count:** Total number of connections (edges) defined in the `Connections` section.

- **total_state_space_dim:** The sum of the product of dimensions for each variable. This is the raw (non-logarithmic) total state space size.
    - *Calculation Details:* `sum(Π(dims_i))` for each variable `i`.

- **max_variable_dim:** The largest single dimension size found across all variables.

*(For precise calculation formulas, refer to the implementation in `src/gnn_type_checker/resource_estimator.py`.)* 