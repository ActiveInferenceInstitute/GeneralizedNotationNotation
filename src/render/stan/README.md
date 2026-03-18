# Stan Renderer

`src/render/stan/` generates **Stan model code** from parsed GNN variables and connections.

## Usage

```python
from render.stan import render_stan

stan_code = render_stan(
    variables=parsed_variables,
    connections=parsed_connections,
    model_name="MyModel",
)
```

## Output

The API returns Stan code as a string. The parent render step is responsible for writing it to a `.stan` file and arranging output directories.

## Notes

This backend uses simple structural heuristics and does not attempt full Active Inference semantics.

