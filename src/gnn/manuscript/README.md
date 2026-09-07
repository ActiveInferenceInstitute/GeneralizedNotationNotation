# gnn.manuscript

Deterministic `{{...}}` token production for the GNN manuscript. The
producer reads the repository snapshot at the current commit and emits
`output/data/manuscript_variables.json`; the render pipeline hydrates the
manuscript from that map.

```python
from gnn.manuscript import generate_variables
```

Private helpers used by the behavior tests live in
`gnn.manuscript.variables`; see `AGENTS.md` for the module contract.
