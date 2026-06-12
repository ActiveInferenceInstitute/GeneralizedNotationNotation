# Type System Examples — Specification

## Example Requirements

Each example must:
- Be syntactically valid in its host language (Scala, Python, Julia)
- Demonstrate at least one dimension constraint (e.g., `A: Matrix[num_obs, num_states]`)
- Include comments mapping to the GNN type checker's validation rules
- Be parseable by the corresponding GNN format parser

## Supported Example Formats

- `.scala` — Dotty/Scala 3 type-level dimension encoding
- `.py` — Python dataclass with NumPy shape annotations
- `.jl` — Julia struct with parametric type bounds
