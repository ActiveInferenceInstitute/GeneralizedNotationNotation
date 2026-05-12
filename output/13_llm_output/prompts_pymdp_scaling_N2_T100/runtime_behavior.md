# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

```python
import json
from pymdp import *
from pymdp.models.inference import InferenceModelAnnotation as INF_MODELANNOTATION
from pymdp.models.inference import InferenceContext, InferenceContextType
from pymdp.utils import *
from pymdp.util import *
import json
import math
import numpy as np


def generate_ontology(
    graph: dict = {},
    num_hidden_states=2,
    num_actions=3,
    num_timesteps=100,
    num_layers=5,
    num_nodes=64,
    num_classes=100,
    num_labels=[
        "A",
        "B",
        "C"
    ],
) -> dict:

    """Generate a topology for the graph.

    Args:
        graph (dict): A dictionary of dictionaries representing nodes and edges.

        num_hidden_states: The number of hidden states in the graph.
            This is used to generate the inference model annotations.

        num_actions: The number of actions in the graph.
            This is used to generate the inference model annotations.

    Returns:
        dict: A dictionary containing the topology for the graph.
    """

    # Generate a list of nodes and edges
    nodes = []
    edges = []
    for node, edge in graph.items():
        if num_hidden_states == 2:
            nodes.append(node)
        elif num_actions == 3:
            edges.append((edge[0], edge[1]))

    # Generate a list of labels and transitions
    labels = []
    for label, transition in graph.items():

        # Create an inference context with the node and label
        inf_context = InferenceContext(node=label)
        inf_context.update({
            "nodes": nodes,
            "edges": edges,
            "labels": labels
        })

    return {"nodes": nodes, "edges": edges}


def generate_ontology_with_inferences(
    graph: dict = {},
    num_hidden_states=2,
    num_actions=3,
    num_layers=5,
    num_classes=100,
    num_labels=[
        "A",
        "B",
        "C"
    ],
) -> dict:

    """Generate a topology for the graph.

    Args:
        graph (dict): A dictionary of dictionaries