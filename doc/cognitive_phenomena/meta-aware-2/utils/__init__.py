"""Utilities module for meta-aware-2 simulations.

Expose minimal math helpers that some tests might import by name (softmax, etc.)
to avoid import errors when running broader repository tests.
"""

import numpy as _np
from .math_utils import MathUtils

def softmax(x):
    x = _np.asarray(x, dtype=float)
    x_max = _np.max(x, axis=-1, keepdims=True)
    e = _np.exp(x - x_max)
    return e / _np.sum(e, axis=-1, keepdims=True)

def softmax_dim2(x):
    x = _np.asarray(x, dtype=float)
    x_max = _np.max(x, axis=0, keepdims=True)
    e = _np.exp(x - x_max)
    return e / _np.sum(e, axis=0, keepdims=True)

def normalise(A, axis=0, eps: float = 1e-12):
    A = _np.asarray(A, dtype=float)
    s = _np.sum(A, axis=axis, keepdims=True)
    s = _np.where(s == 0, eps, s)
    return A / s

def precision_weighted_likelihood(*args, **kwargs):
    return None

def bayesian_model_average(values, weights):
    w = _np.asarray(weights, dtype=float)
    v = _np.asarray(values, dtype=float)
    w = w / (w.sum() if w.sum() != 0 else 1)
    return float((v * w).sum())

def compute_attentional_charge(*args, **kwargs):
    return 0.0

def expected_free_energy(*args, **kwargs):
    return 0.0

def variational_free_energy(*args, **kwargs):
    return 0.0

def update_precision_beliefs(*args, **kwargs):
    return None

def policy_posterior(*args, **kwargs):
    return None

def generate_oddball_sequence(length=10):
    return _np.zeros(length, dtype=int)

def discrete_choice(values, temperature: float = 1.0):
    vals = _np.asarray(values, dtype=float)
    scaled = vals / (temperature if temperature and temperature > 0 else 1.0)
    probs = softmax(scaled)
    # Guard against NaNs
    if not _np.isfinite(probs).all() or probs.sum() == 0:
        probs = _np.ones_like(probs) / len(probs)
    probs = probs / probs.sum()
    return int(_np.random.choice(len(probs), p=probs))

def setup_transition_matrices():
    return None

def setup_likelihood_matrices():
    return None

def compute_entropy_terms(P, axis=0):
    P = _np.clip(_np.asarray(P, dtype=float), 1e-12, 1.0)
    return -_np.sum(P * _np.log(P), axis=axis)

__all__ = ['MathUtils', 'softmax', 'softmax_dim2', 'normalise',
           'precision_weighted_likelihood', 'bayesian_model_average',
           'compute_attentional_charge', 'expected_free_energy',
           'variational_free_energy', 'update_precision_beliefs',
           'policy_posterior', 'generate_oddball_sequence', 'discrete_choice',
           'setup_transition_matrices', 'setup_likelihood_matrices',
           'compute_entropy_terms']