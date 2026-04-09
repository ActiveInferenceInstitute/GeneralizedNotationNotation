# pymdp 1.0.0 Performance Guide

> **Scope:** Practical performance notes for pymdp 1.0.0 (JAX-first) in the
> context of this repository's render/execute pipeline. The numbers are
> representative, not binding — they reflect what you should expect on a
> reasonably modern CPU. GPU figures depend on your CUDA/Metal stack.

## Table of Contents

1. [JIT Compilation: Why the First Step is Slow](#jit-compilation-why-the-first-step-is-slow)
2. [Batched Rollouts](#batched-rollouts)
3. [Memory Footprint of List-of-Array Models](#memory-footprint-of-list-of-array-models)
4. [Avoiding Python-Loop Overhead](#avoiding-python-loop-overhead)
5. [Dtype Choice (float32 vs float64)](#dtype-choice-float32-vs-float64)
6. [Benchmarking the Pipeline](#benchmarking-the-pipeline)
7. [Known Pitfalls](#known-pitfalls)

---

## JIT Compilation: Why the First Step is Slow

pymdp 1.0.0's `Agent` is an `equinox.Module`. Every public method
(`infer_states`, `infer_policies`, `sample_action`) is traced on its first
call and compiled to XLA. That gives a noticeable warm-up hit:

| Operation              | Cold (first call) | Warm (steady state) |
|------------------------|-------------------|---------------------|
| `infer_states`         | 100–500 ms        | 1–5 ms              |
| `infer_policies`       | 50–200 ms         | 0.5–3 ms            |
| `sample_action`        | 20–100 ms         | 0.1–1 ms            |
| `update_empirical_prior` | 10–50 ms        | 0.1–0.5 ms          |

Implications:

- Do **not** benchmark a single step — it's almost entirely compilation.
- For short rollouts (e.g. 5-step unit tests), the compilation cost dominates
  the measured wall time. `test_pymdp_contracts.py::test_render_execute_contract_pymdp`
  takes ~1.6 s because of JIT warm-up; the actual rollout takes <50 ms.

## Batched Rollouts

Vectorising `batch_size > 1` across independent agents is the single biggest
speed-up available. pymdp 1.0.0 is designed for it: all public methods
operate on leading-batch-dim tensors and `sample_action` accepts a batch of
PRNG keys.

Rule of thumb:

- `batch_size = 1`: ~1× throughput (baseline)
- `batch_size = 16`: ~6–10× throughput vs running 16 separate agents serially
- `batch_size = 128`: saturates on a typical laptop CPU

The pipeline's `run_simple_pymdp_simulation` defaults to `batch_size=1`
because most GNN POMDPs are single-agent. Override via the GNN spec:

```json
"model_parameters": {
  "num_timesteps": 20,
  "batch_size": 32
}
```

`_build_pymdp_agent` then broadcasts A/B/C/D along the batch axis and the
rollout runs all 32 agents in lock-step.

## Memory Footprint of List-of-Array Models

The JAX arrays carry a leading batch dim, so memory use scales linearly with
`batch_size`:

| Parameter                | Single agent (No=5, Ns=5, Nu=5) | batch=64 |
|--------------------------|---------------------------------|----------|
| `A[0]` `(1, 5, 5)`       | 100 bytes                       | 6.4 KB   |
| `B[0]` `(1, 5, 5, 5)`    | 500 bytes                       | 32 KB    |
| `C[0]` `(1, 5)`          | 20 bytes                        | 1.3 KB   |
| `D[0]` `(1, 5)`          | 20 bytes                        | 1.3 KB   |
| `qs` history (T=100)     | ~2 KB                           | ~130 KB  |

Larger problems (e.g. Ns=50, Nu=10) increase `B[0]` by ~500× but the per-step
cost is still JIT-dominated, not memory-dominated.

## Avoiding Python-Loop Overhead

The pipeline rollout is a plain Python loop over timesteps. That's fine for
up to ~200 steps; past that, consider using a JAX `lax.scan` over a compiled
step function.

Sketch (not shipped in the pipeline; illustrative):

```python
import jax
import jax.numpy as jnp
from pymdp.agent import Agent

@jax.jit
def step_fn(carry, inputs):
    prior, key = carry
    obs, = inputs
    qs, _ = agent.infer_states([obs], empirical_prior=prior, return_info=True)
    q_pi, _ = agent.infer_policies(qs)
    key, sub = jax.random.split(key)
    action = agent.sample_action(q_pi, rng_key=jax.random.split(sub, 2)[1:])
    new_prior = agent.update_empirical_prior(action, qs)
    return (new_prior, key), (qs, q_pi, action)

(_, _), (qs_hist, q_pi_hist, act_hist) = jax.lax.scan(step_fn, (agent.D, key), (obs_seq,))
```

This removes the Python overhead at each step and fuses the whole rollout
into a single XLA kernel. Expect 3–10× speed-up on rollouts of 100+ steps.

## Dtype Choice (float32 vs float64)

pymdp 1.0.0 defaults to `float32`. The pipeline follows suit by casting GNN
matrices with `jnp.asarray(..., dtype=jnp.float32)`. `float64` is possible
(set `JAX_ENABLE_X64=1`) but memory doubles and CPU throughput drops roughly
2× with no accuracy gain for typical discrete POMDPs.

## Benchmarking the Pipeline

The repository ships a runnable smoke benchmark that you can adapt:

```bash
uv run pytest src/tests/test_pymdp_contracts.py::test_pymdp_seeded_reproducibility_contract \
    --durations=5 -v
```

This runs the same GNN spec twice under a fixed seed, verifies determinism,
and prints per-test wall times in the slowest-durations report.

For a larger workload, use the ActInf POMDP end-to-end test:

```bash
uv run pytest src/tests/test_pymdp_contracts.py::test_actinf_pomdp_render_execute_analyze_e2e \
    --durations=5 -v -m "integration and slow"
```

## Known Pitfalls

- **Stale wheel after version bump.** Some `uv pip install` flows can leave a
  stale `pymdp/` directory while updating the `*.dist-info`. If
  `importlib.metadata.version("inferactively-pymdp")` disagrees with the
  behaviour (e.g. reports 1.0.0 but `Agent.update_empirical_prior` is
  missing), delete `.venv/lib/python*/site-packages/pymdp/` and re-install.
- **`num_controls[f] > 1` assertion.** pymdp 1.0.0's `Agent._validate`
  requires every factor listed in `control_fac_idx` to have `num_controls > 1`.
  Passive HMMs must omit `control_fac_idx` (the pipeline handles this
  automatically for `num_actions == 1`).
- **JAX array as static field warning.** You may see
  `UserWarning: A JAX array is being set as static!` from equinox. This is
  harmless for our use — it's triggered by pymdp's internal `Policies` helper.
