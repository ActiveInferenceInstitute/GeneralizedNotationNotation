# Render Frameworks

> Step 11 generates simulation code for multiple Active Inference frameworks.
> Run: `uv run python src/11_render.py --target-dir input/gnn_files`

## Supported Frameworks

| Framework | Language | Deps | Status |
|-----------|----------|------|--------|
| **PyMDP** | Python | `inferactively-pymdp` | Optional |
| **JAX** | Python | `jax`, `jaxlib`, `optax` (no Flax) | Recommended |
| **RxInfer.jl** | Julia | Julia + RxInfer | Optional |
| **ActiveInference.jl** | Julia | Julia + ActiveInference | Optional |
| **DisCoPy** | Python | `discopy` | Optional |

**Execution fallback order** (Step 12): JAX → PyMDP → ActiveInference.jl → RxInfer.jl → DisCoPy

---

## JAX ⭐ (Pure JAX — No Flax Required)

```python
"""Pure JAX Active Inference — no Flax dependency."""
import jax
import jax.numpy as jnp
from jax import random, jit

# Model matrices (extracted from GNN)
A_MATRIX = jnp.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
B_MATRIX = jnp.array([[[0.9, 0.1], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]])
C_VECTOR = jnp.array([0.0, 1.0, 0.0])
D_VECTOR = jnp.array([0.5, 0.5])

@jit
def belief_update(prior, obs, A):
    posterior = prior * A[obs, :]
    return posterior / jnp.sum(posterior)

@jit
def expected_free_energy(qs, A, B, C, action):
    qs_next = B[:, :, action] @ qs
    qo = A @ qs_next
    H_qo = -jnp.sum(qo * jnp.log(qo + 1e-10))   # Epistemic
    pragmatic = jnp.sum(qo * C)                    # Pragmatic
    return -pragmatic + H_qo
```

**Key**: Generated code uses only `jax`, `jax.numpy`, `optax` — never `flax` or `flax.linen`.

---

## PyMDP

```python
"""PyMDP Active Inference."""
try:
    from pymdp.agent import Agent
    from pymdp import utils
    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False

# POMDP matrices
A = [[0.9, 0.1], [0.1, 0.9]]  # Likelihood
B = [[[0.9, 0.1], [0.1, 0.9]]]  # Transition
C = [0.0, 1.0]  # Preferences
D = [0.5, 0.5]  # Initial prior

def run_simulation(n_steps=10):
    if not PYMDP_AVAILABLE:
        raise ImportError("Install: uv pip install inferactively-pymdp")
    agent = Agent(A=A, B=B, C=C, D=D)
    obs = agent.reset()
    for t in range(n_steps):
        qs = agent.infer_states(obs)
        action = agent.sample_action()
        print(f"Step {t}: obs={obs}, action={action}")
```

---

## RxInfer.jl (Julia)

```julia
# Generated Julia code
using RxInfer, Distributions, LinearAlgebra, Random

const A_MATRIX = [0.9 0.1; 0.1 0.9; 0.5 0.5]
const B_MATRIX = cat([0.9 0.1; 0.1 0.9], [0.5 0.5; 0.5 0.5], dims=3)

@model function active_inference_model(n_timesteps)
    s = randomvar(n_timesteps)
    o = randomvar(n_timesteps)
    s[1] ~ Categorical(D_VECTOR)
    for t in 2:n_timesteps
        s[t] ~ Categorical(B_MATRIX[:, s[t-1], 1])
        o[t] ~ Categorical(A_MATRIX[:, s[t]])
    end
    return s, o
end
```

**Setup**: `julia src/execute/rxinfer/setup_environment.jl --verbose`

---

## Matrix Extraction from GNN

All frameworks use the same extraction:

```python
def extract_matrices(gnn_model: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract POMDP matrices from parsed GNN model."""
    init = gnn_model.get("initialparameterization", {})
    return {
        "A": np.array(init.get("A", np.eye(3))),         # Likelihood
        "B": np.array(init.get("B", np.eye(3))),         # Transition
        "C": np.array(init.get("C", np.zeros(3))),        # Preferences
        "D": np.array(init.get("D", np.ones(3) / 3)),    # Initial prior
    }
```

---

## Output Structure

```
output/11_render_output/
└── model_name/
    ├── pymdp/model_name_pymdp.py
    ├── jax/model_name_jax.py
    ├── rxinfer/model_name_rxinfer.jl
    ├── activeinference_jl/model_name_activeinference.jl
    └── discopy/model_name_discopy.py
```

---

**Last Updated**: March 2026 | **Status**: Production Standard
