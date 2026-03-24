# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

```python
import numpy as np
from typing import List, Dict, Tuple

def gnn_representation(
    graph: Dict[str, Dict[str, Dict]],
    num_positions: int = 3,
    num_velocities: int = 2,
    num_visual_obs: int = 6,
    num_proprio_obs: int = 4,
    num_actions: int = 2
) -> Tuple[Dict[str, Dict], Dict]:
    """GNN Representation of a factor graph."""

    def gnn(
        node1: str,
        node2: str,
        action: List[Tuple[int, float]],
        prob: List[float] = [],
        log_probs: List[List[float]] = []  # Log probabilities for each node
    ):

        """GNN Representation of a graph."""

        if isinstance(node1, str):
            return gnn_representation(graph.get(node1), num_positions)

        if isinstance(node2, str):
            return gnn_representation(graph.get(node2), num_positions)

        if isinstance(action, List[Tuple[int, float]]):
            return gnn_representation(
                graph.get(node1 + node2),
                action,
                prob,
                log_probs,
                [],
                [],
                num_actions = num_actions,
                num_visitors = num_visitors,
                num_proprio_obs = num_proprio_obs
            )

        if isinstance(prob, List[Tuple[int, float]]):
            return gnn_representation(graph.get(node1 + node2), prob)

    def gnn_representation(
        graph: Dict[str, Dict],
        num_positions: int = 3
    ):
        """GNN Representation of a factor graph."""

        if isinstance(num_positions, int):
            return {}

        for node in graph.keys():

            # Create a node with the same number as its neighbors and label it as "visited"
            visited = set()
            visited.add(node)
            gnn(
                node1: str,
                node2: str,
                action: List[Tuple[int, float]],
                prob: List[float],
                log_probs: List[List[float]] = [],
                num_visitors=num_visitors,
                num_proprio_