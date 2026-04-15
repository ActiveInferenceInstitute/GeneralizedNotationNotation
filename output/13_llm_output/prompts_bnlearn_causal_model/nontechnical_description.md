# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

```python
import json
from typing import List, Dict

def generate_bio(node: dict) -> List[Dict]:
    """Generate BioInfo object from a graph."""

    bio = {}
    for node in node.keys():
        if isinstance(node, dict):
            bio += generate_bio([dict(node)] + [generate_bio([{}])for _ in range(len(node))])

        elif isinstance(node, list) and len(node) > 0:
            bio = generate_bio([{
                node[key]: generate_bio([{}] for _ in range(len(node))] + [generate_bio([{}])for _ in range(len(node)))
            })

    return bio
```