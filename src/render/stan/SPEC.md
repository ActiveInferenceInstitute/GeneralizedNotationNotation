# Stan Render Backend Specification

## Overview

The Stan backend converts parsed GNN variables and connections into a **syntactically valid Stan model**.

## Public API

`render.stan` must export:

- `render_stan(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]], model_name: str = "gnn_model") -> str`

## Inputs

- `variables`: list of dicts with keys such as `name`, `dimensions`, `dtype`
- `connections`: list of dicts with keys such as `source`, `target`, `directed`
- `model_name`: used in output comments

## Output

- a Stan program string containing `data {}`, `parameters {}`, and `model {}`

## Non-goals

- full Active Inference semantics
- automatic inference configuration or execution

