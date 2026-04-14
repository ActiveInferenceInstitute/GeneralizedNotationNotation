# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

```python
import numpy as np
from scipy import stats

# GNN Representation
GNN = GNNVersionAndFlags(
    level_1=np.array([
        (0.85, 0.05, 0.05, 0.05),
        (0.05, 0.85, 0.05, 0.05)
    ], dtype=[dtype]
)
GNN = GNNVersionAndFlags(level_2=np.array([
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.9, 0.1),
        (0.0, 0.0, 0.1, 0.9)
    ], dtype=[dtype]
])
GNN = GNNVersionAndFlags(level_2=np.array([
        (0.95, 0.1),
        (0.1, 0.9),
        (0.0, 0.0),
        (0.0, 0.0)
    ], dtype=[dtype]
])
GNN = GNNVersionAndFlags(level_2=np.array([
            (0.85, 0.1),
            (0.05, 0.9),
            (0.05, 0.1),
        ]
    ), dtype=[dtype]
)
GNN = GNNVersionAndFlags(level_2=np.array([
                (0.85, 0.1),
                (0.05, 0.9),
            ]
    ], dtype=[dtype]
])
```