# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

You can use the `pymppo` library to generate a sparse graph of all possible actions for an agent in a given ontology. The following code generates a sparse graph with 16 nodes and 30 edges:
```python
import pymppo as pmp
from pymppo import Graph, Agent

# Create the graph structure
graph = pmp.Graph()
graph.add_nodes("A", num_nodes=4) # Add A to the graph
graph.add_edges(
    ("A", "B"),
    ("A", "C") + ["D"])  # Add A and B as edges, then add C as a new edge
)
```
Next, you can use the `agent` class from `pymppo`:
```python
agent = pmp.Agent(graph)
```
The following code generates an action graph for the agent:
```python
action_graphs = [
    {"name": "A", "actions": ["B"], "nodes": ["C"]},
    {"name": "A", "actions": ["D"},
    {"name": "B", "actions": []}
]
```
Then, you can use the `agent` class to generate a sparse graph:
```python
graph = pmp.Graph()
for node in action_graphs[0]:
    agent(node)
```
Finally, you can use the `generate_action_graph()` function from `pymppo`:
```python
gen_actions = pmp.GeneratedActionSet([{"name": "A", "nodes": ["C"]}, {"name": "B"}])
```