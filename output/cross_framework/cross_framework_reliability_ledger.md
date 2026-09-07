# GNN Cross-Framework Reliability Ledger

- Schema: gnn_cross_framework_reliability_ledger_v1
- Families: 9
- Strict: true
- Frameworks: pymdp, rxinfer, jax, numpyro, pytorch, activeinference_jl, discopy

| Family | Status | Comparison | Compared Frameworks | Required Failures |
| --- | --- | --- | --- | ---: |
| basics | passed | skipped | pymdp | 0 |
| discrete | passed | skipped | pymdp | 0 |
| continuous | passed | passed | jax, numpyro, rxinfer | 0 |
| hierarchical | passed | passed | jax, pymdp, rxinfer | 0 |
| multiagent | passed | skipped | rxinfer | 0 |
| precision | passed | skipped | pymdp | 0 |
| structured | passed | skipped | pymdp | 0 |
| gridworld | passed | passed | activeinference_jl, pymdp, rxinfer | 0 |
| scaling-study | passed | skipped | pymdp | 0 |

## Framework Profiles

### basics

| Framework | Profile | Status | Reason | Metrics |
| --- | --- | --- | --- | --- |
| pymdp | required | passed |  | available |
| rxinfer | unsupported | unsupported | not declared compatible for this model family | missing |
| jax | unsupported | unsupported | not declared compatible for this model family | missing |
| numpyro | unsupported | unsupported | not declared compatible for this model family | missing |
| pytorch | unsupported | unsupported | not declared compatible for this model family | missing |
| activeinference_jl | unsupported | unsupported | not declared compatible for this model family | missing |
| discopy | unsupported | unsupported | not declared compatible for this model family | missing |

### discrete

| Framework | Profile | Status | Reason | Metrics |
| --- | --- | --- | --- | --- |
| pymdp | required | passed |  | available |
| rxinfer | unsupported | unsupported | not declared compatible for this model family | missing |
| jax | unsupported | unsupported | not declared compatible for this model family | missing |
| numpyro | unsupported | unsupported | not declared compatible for this model family | missing |
| pytorch | unsupported | unsupported | not declared compatible for this model family | missing |
| activeinference_jl | unsupported | unsupported | not declared compatible for this model family | missing |
| discopy | unsupported | unsupported | not declared compatible for this model family | missing |

### continuous

| Framework | Profile | Status | Reason | Metrics |
| --- | --- | --- | --- | --- |
| pymdp | unsupported | unsupported | not declared compatible for this model family | missing |
| rxinfer | required | passed |  | available |
| jax | required | passed |  | available |
| numpyro | required | passed |  | available |
| pytorch | unsupported | unsupported | not declared compatible for this model family | missing |
| activeinference_jl | unsupported | unsupported | not declared compatible for this model family | missing |
| discopy | unsupported | unsupported | not declared compatible for this model family | missing |

### hierarchical

| Framework | Profile | Status | Reason | Metrics |
| --- | --- | --- | --- | --- |
| pymdp | required | passed |  | available |
| rxinfer | required | passed |  | available |
| jax | required | passed |  | available |
| numpyro | unsupported | unsupported | not declared compatible for this model family | missing |
| pytorch | unsupported | unsupported | not declared compatible for this model family | missing |
| activeinference_jl | unsupported | unsupported | not declared compatible for this model family | missing |
| discopy | unsupported | unsupported | not declared compatible for this model family | missing |

### multiagent

| Framework | Profile | Status | Reason | Metrics |
| --- | --- | --- | --- | --- |
| pymdp | unsupported | unsupported | not declared compatible for this model family | missing |
| rxinfer | required | passed |  | available |
| jax | unsupported | unsupported | not declared compatible for this model family | missing |
| numpyro | unsupported | unsupported | not declared compatible for this model family | missing |
| pytorch | unsupported | unsupported | not declared compatible for this model family | missing |
| activeinference_jl | unsupported | unsupported | not declared compatible for this model family | missing |
| discopy | unsupported | unsupported | not declared compatible for this model family | missing |

### precision

| Framework | Profile | Status | Reason | Metrics |
| --- | --- | --- | --- | --- |
| pymdp | required | passed |  | available |
| rxinfer | unsupported | unsupported | not declared compatible for this model family | missing |
| jax | unsupported | unsupported | not declared compatible for this model family | missing |
| numpyro | unsupported | unsupported | not declared compatible for this model family | missing |
| pytorch | unsupported | unsupported | not declared compatible for this model family | missing |
| activeinference_jl | unsupported | unsupported | not declared compatible for this model family | missing |
| discopy | unsupported | unsupported | not declared compatible for this model family | missing |

### structured

| Framework | Profile | Status | Reason | Metrics |
| --- | --- | --- | --- | --- |
| pymdp | required | passed |  | available |
| rxinfer | unsupported | unsupported | not declared compatible for this model family | missing |
| jax | unsupported | unsupported | not declared compatible for this model family | missing |
| numpyro | unsupported | unsupported | not declared compatible for this model family | missing |
| pytorch | unsupported | unsupported | not declared compatible for this model family | missing |
| activeinference_jl | unsupported | unsupported | not declared compatible for this model family | missing |
| discopy | unsupported | unsupported | not declared compatible for this model family | missing |

### gridworld

| Framework | Profile | Status | Reason | Metrics |
| --- | --- | --- | --- | --- |
| pymdp | required | passed |  | available |
| rxinfer | required | passed |  | available |
| jax | unsupported | unsupported | not declared compatible for this model family | missing |
| numpyro | unsupported | unsupported | not declared compatible for this model family | missing |
| pytorch | unsupported | unsupported | not declared compatible for this model family | missing |
| activeinference_jl | required | passed |  | available |
| discopy | unsupported | unsupported | not declared compatible for this model family | missing |

### scaling-study

| Framework | Profile | Status | Reason | Metrics |
| --- | --- | --- | --- | --- |
| pymdp | required | passed |  | available |
| rxinfer | unsupported | unsupported | not declared compatible for this model family | missing |
| jax | unsupported | unsupported | not declared compatible for this model family | missing |
| numpyro | unsupported | unsupported | not declared compatible for this model family | missing |
| pytorch | unsupported | unsupported | not declared compatible for this model family | missing |
| activeinference_jl | unsupported | unsupported | not declared compatible for this model family | missing |
| discopy | unsupported | unsupported | not declared compatible for this model family | missing |
