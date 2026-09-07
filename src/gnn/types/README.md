# gnn.types

Shared dataclass types for the GNN pipeline: the parsed-document model
(`ParsedGNN`, `GNNVariable`, `GNNConnection`), validation and round-trip
results, and re-exports of `GNNFormat` / `GNNInternalRepresentation` from
`gnn.parsers.common`.

```python
from gnn.types import ParsedGNN, ValidationLevel
```

Importing `gnn.types` is stdlib-light; see `AGENTS.md` for the module
contract and `definitions.py` for the definitions.
