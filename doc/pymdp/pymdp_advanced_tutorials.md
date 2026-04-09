# pymdp 1.0.0 Advanced Tutorials

> **Scope:** This document contains *illustrative* patterns for building
> advanced active-inference agents on top of pymdp 1.0.0 (JAX-first). These
> examples are not pipeline contracts — the canonical pipeline entry points
> live in `src/execute/pymdp/simple_simulation.py` and
> `src/execute/pymdp/pymdp_simulation.py`.
>
> All examples use the real 1.0.0 API (batched `list[jax.Array]` models,
> `infer_states(empirical_prior=…, return_info=True)`, `infer_policies(qs)`,
> `sample_action(q_pi, rng_key=…)`, `update_empirical_prior(action, qs)`).

## Table of Contents

1. [Multi-factor Agents with Learning](#multi-factor-agents-with-learning)
2. [Multi-modal Sensory Integration](#multi-modal-sensory-integration)
3. [Batched Rollouts](#batched-rollouts-batch_size--1)
4. [Preference Shaping](#preference-shaping)
5. [Reproducible Rollouts with JAX PRNG Keys](#reproducible-rollouts-with-jax-prng-keys)
6. [Pointers to Upstream Features](#pointers-to-upstream-features)

---

## Multi-factor Agents with Learning

pymdp 1.0.0 supports Dirichlet updates on A/B via `learn_A`, `learn_B`, and
`pA` / `pB` concentration parameters (arrays mirroring the A/B shapes).

```python
import jax.numpy as jnp
import jax.random as jr
from pymdp.agent import Agent

num_states = [3, 2]
num_obs = [3]
num_controls = [3, 1]  # factor 1 is passive

A = [jnp.ones((1, 3, 3, 2)) / 3.0]   # (batch, num_obs[0], Ns0, Ns1)
B = [
    jnp.ones((1, 3, 3, 3)) / 3.0,    # controllable factor 0
    jnp.eye(2)[None, :, :, None],    # passive factor 1, 1 action
]
C = [jnp.zeros((1, 3))]
D = [jnp.array([[1/3, 1/3, 1/3]]), jnp.array([[0.5, 0.5]])]

pA = [jnp.full_like(A[0], 1.0)]      # symmetric Dirichlet concentration
pB = [jnp.full_like(B[0], 1.0), jnp.full_like(B[1], 1.0)]

agent = Agent(
    A=A, B=B, C=C, D=D,
    pA=pA, pB=pB,
    num_controls=num_controls,
    control_fac_idx=[0],  # factor 1 has num_controls == 1 and must NOT be listed
    policy_len=1,
    batch_size=1,
    learn_A=True,
    learn_B=True,
)
```

Calls to `agent.infer_parameters(...)` (see upstream docs) update the
Dirichlet posteriors after each rollout segment.

## Multi-modal Sensory Integration

Multiple modalities → multiple entries in the `A` / `C` lists and multiple
observation arrays per step.

```python
A = [
    jnp.ones((1, 4, 3)) / 4.0,   # modality 0: 4 outcomes × 3 states
    jnp.ones((1, 2, 3)) / 2.0,   # modality 1: 2 outcomes × 3 states
]
C = [jnp.zeros((1, 4)), jnp.array([[0.0, 1.0]])]
B = [jnp.ones((1, 3, 3, 2)) / 3.0]
D = [jnp.array([[1/3, 1/3, 1/3]])]

agent = Agent(A=A, B=B, C=C, D=D, num_controls=[2], control_fac_idx=[0], batch_size=1)

# Per-step observation list: one 1-element jnp.int32 array per modality
obs = [jnp.array([2], dtype=jnp.int32), jnp.array([0], dtype=jnp.int32)]
qs, info = agent.infer_states(obs, empirical_prior=agent.D, return_info=True)
```

## Batched Rollouts (`batch_size > 1`)

pymdp 1.0.0 can vectorise identical agents across a batch dimension. Build A,
B, C, D with the chosen batch size and duplicate observations across the
batch:

```python
batch = 4
A = [jnp.broadcast_to(A_single[None, ...], (batch, *A_single.shape))]
B = [jnp.broadcast_to(B_single[None, ...], (batch, *B_single.shape))]
C = [jnp.broadcast_to(C_single[None, ...], (batch, *C_single.shape))]
D = [jnp.broadcast_to(D_single[None, ...], (batch, *D_single.shape))]

agent = Agent(A=A, B=B, C=C, D=D, num_controls=[2], control_fac_idx=[0], batch_size=batch)

# Observation arrays now have size `batch`.
obs = [jnp.array([0, 1, 2, 0], dtype=jnp.int32)]
qs, info = agent.infer_states(obs, empirical_prior=agent.D, return_info=True)

key = jr.PRNGKey(0)
action = agent.sample_action(q_pi=..., rng_key=jr.split(key, batch + 1)[1:])
# action.shape == (batch, num_factors)
```

The pipeline's `run_simple_pymdp_simulation` pins `batch_size=1` by default
but the underlying `_build_pymdp_agent` accepts any batch size you pass via
`model_parameters.batch_size` in the GNN spec.

## Preference Shaping

pymdp 1.0.0 treats `C[m]` as the log-prior over observations in modality `m`.
Positive values = preferred outcomes; negative values = aversive outcomes.

```python
C = [jnp.array([[-1.0, 0.0, 2.0]])]   # shape (batch, num_obs[0])
```

Re-normalise expected-free-energy precision with `gamma` if you change the
scale of preferences (pipeline default `gamma=16.0`).

## Reproducible Rollouts with JAX PRNG Keys

pymdp 1.0.0 never touches `numpy.random`; every stochastic choice comes from
an explicit `jax.random.PRNGKey`. To reproduce a rollout exactly, carry a
single root key and split it at each step:

```python
key = jr.PRNGKey(1234)
for t in range(T):
    key, subkey = jr.split(key)
    action_keys = jr.split(subkey, agent.batch_size + 1)
    action = agent.sample_action(q_pi, rng_key=action_keys[1:])
```

The pipeline additionally seeds the numpy generator used for fake-environment
transitions from `gnn_spec["model_parameters"]["random_seed"]`, so a single
seed reproduces both the agent's samples and the environment transitions
(see `test_pymdp_seeded_reproducibility_contract`).

## Pointers to Upstream Features

Features available upstream that this repository does **not** wrap:

- `Agent.infer_parameters(...)` — Dirichlet learning updates
- `pymdp.planning.*` — MCTS-style planning via `mctx`
- `pymdp.control.generate_I_matrix` — inductive-inference pruning
- `pymdp.envs.*` — built-in JAX environments for agent training

See the upstream documentation and `src/tests/test_pymdp_1_0_0_upstream_api.py`
for the exact surface currently asserted by this repository.
