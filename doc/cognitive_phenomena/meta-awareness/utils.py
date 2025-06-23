# -*- coding: utf-8 -*-
"""
Utility functions for Sandved-Smith et al. (2021) computational phenomenology model.
Mathematical operations for active inference with hierarchical precision control.
"""

import numpy as np
from typing import Union, Tuple

def softmax(X: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert log probabilities to normalized probabilities.
    
    Args:
        X: Log probabilities (1D array or list)
        
    Returns:
        Normalized probability distribution
    """
    X = np.asarray(X)
    # Add small epsilon for numerical stability
    exp_X = np.exp(X) + 1e-5
    norm = np.sum(exp_X)
    return exp_X / norm

def softmax_dim2(X: np.ndarray) -> np.ndarray:
    """
    Convert matrix of log probabilities to matrix of normalized probabilities.
    Normalizes along axis 0 (columns).
    
    Args:
        X: Matrix of log probabilities
        
    Returns:
        Matrix of normalized probabilities
    """
    X = np.asarray(X)
    exp_X = np.exp(X) + 1e-5
    norm = np.sum(exp_X, axis=0)
    return exp_X / norm

def normalise(X: np.ndarray) -> np.ndarray:
    """
    Normalize a matrix of probabilities along columns (axis 0).
    
    Args:
        X: Matrix of probabilities
        
    Returns:
        Column-normalized matrix
    """
    X = np.asarray(X)
    return X / np.sum(X, axis=0)

def precision_weighted_likelihood(A: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply precision weighting to likelihood matrix.
    
    Args:
        A: Likelihood matrix
        gamma: Precision parameter (inverse variance)
        
    Returns:
        Precision-weighted likelihood matrix
    """
    return softmax_dim2(np.log(A) * gamma)

def bayesian_model_average(beta_values: np.ndarray, state_beliefs: np.ndarray, 
                          likelihood_matrix: np.ndarray) -> float:
    """
    Compute Bayesian model average for precision beliefs.
    
    Args:
        beta_values: Array of inverse precision parameters
        state_beliefs: Current state belief distribution
        likelihood_matrix: Likelihood mapping matrix
        
    Returns:
        Model-averaged precision estimate
    """
    weighted_beliefs = np.inner(likelihood_matrix, state_beliefs)
    return np.sum(beta_values * weighted_beliefs)

def compute_attentional_charge(O_bar: np.ndarray, A_bar: np.ndarray, 
                              X_bar: np.ndarray, A: np.ndarray) -> float:
    """
    Compute 'attentional charge' - the inverse precision updating term.
    Based on "Uncertainty, epistemics and active inference" (Parr & Friston).
    
    Args:
        O_bar: Observation posterior beliefs
        A_bar: Precision-weighted likelihood matrix
        X_bar: State posterior beliefs
        A: Original likelihood matrix
        
    Returns:
        Attentional charge value
    """
    charge = 0.0
    n_obs, n_states = A.shape
    
    for i in range(n_obs):  # Loop over outcomes
        for j in range(n_states):  # Loop over states
            charge += (O_bar[i] - A_bar[i, j]) * X_bar[j] * np.log(A[i, j])
    
    return charge

def expected_free_energy(O_pred: np.ndarray, C: np.ndarray, 
                        X_pred: np.ndarray, H: np.ndarray) -> float:
    """
    Compute expected free energy for a policy.
    G = E[o*(ln(o) - C) - x*H]
    
    Args:
        O_pred: Predicted observations under policy
        C: Prior preferences over observations
        X_pred: Predicted states under policy
        H: Entropy terms
        
    Returns:
        Expected free energy value
    """
    epistemic_term = np.sum(O_pred * (np.log(O_pred + 1e-16) - C))
    pragmatic_term = -np.sum(X_pred * H)
    return epistemic_term + pragmatic_term

def variational_free_energy(X_bar: np.ndarray, X_pred: np.ndarray, 
                           A: np.ndarray, obs_idx: int) -> float:
    """
    Compute variational free energy term for policy evaluation.
    
    Args:
        X_bar: Posterior state beliefs
        X_pred: Predicted state beliefs
        A: Likelihood matrix
        obs_idx: Index of observed outcome
        
    Returns:
        Variational free energy value
    """
    return np.sum(X_bar * (np.log(X_bar + 1e-16) - 
                           np.log(A[obs_idx, :] + 1e-16) - 
                           np.log(X_pred + 1e-16)))

def update_precision_beliefs(beta_prior: float, charge: float, 
                           beta_bounds: Tuple[float, float]) -> float:
    """
    Update inverse precision beliefs based on prediction error.
    
    Args:
        beta_prior: Prior inverse precision
        charge: Computed charge (prediction error signal)
        beta_bounds: (min_beta, max_beta) bounds for stability
        
    Returns:
        Updated inverse precision
    """
    min_beta, max_beta = beta_bounds
    
    # Clamp charge to prevent numerical instability
    if charge > min_beta:
        charge = min_beta - 1e-5
    
    beta_posterior = beta_prior - charge
    
    # Ensure bounds
    beta_posterior = np.clip(beta_posterior, min_beta, max_beta)
    
    return beta_posterior

def policy_posterior(log_prior: np.ndarray, expected_free_energy: np.ndarray, 
                    variational_free_energy: np.ndarray = None, 
                    gamma_G: float = 1.0) -> np.ndarray:
    """
    Compute posterior beliefs over policies.
    
    Args:
        log_prior: Log prior beliefs over policies
        expected_free_energy: Expected free energy for each policy
        variational_free_energy: Variational free energy terms (optional)
        gamma_G: Precision parameter for policy selection
        
    Returns:
        Posterior policy distribution
    """
    log_posterior = log_prior - gamma_G * expected_free_energy
    
    if variational_free_energy is not None:
        log_posterior -= variational_free_energy
    
    return softmax(log_posterior)

def discrete_choice(probabilities: np.ndarray) -> int:
    """
    Make discrete choice based on probability distribution.
    
    Args:
        probabilities: Probability distribution over choices
        
    Returns:
        Selected choice index
    """
    return np.argmax(probabilities)

def generate_oddball_sequence(T: int, oddball_times: list = None) -> np.ndarray:
    """
    Generate oddball stimulus sequence.
    
    Args:
        T: Total time steps
        oddball_times: List of time points for oddball stimuli (if None, uses default)
        
    Returns:
        Binary sequence (0=standard, 1=oddball)
    """
    sequence = np.zeros(T)
    
    if oddball_times is None:
        # Default: oddball at 1/5, 2/5, 3/5, 4/5 of trial
        oddball_times = [int(T/5), int(2*T/5), int(3*T/5), int(4*T/5)]
    
    for t in oddball_times:
        if 0 <= t < T:
            sequence[t] = 1
    
    return sequence

def setup_transition_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Set up transition matrices for the three-level model.
    
    Returns:
        Tuple of (B1, B2a, B2b, B3) transition matrices
    """
    # Level 1: Perceptual transitions
    B1 = np.array([[0.8, 0.2], 
                   [0.2, 0.8]])
    
    # Level 2: Attentional transitions
    B2a = np.array([[0.8, 0.0],  # Stay policy
                    [0.2, 1.0]])
    
    B2b = np.array([[0.0, 1.0],  # Switch policy
                    [1.0, 0.0]])
    
    # Level 3: Meta-awareness transitions
    B3 = np.array([[0.9, 0.1], 
                   [0.1, 0.9]])
    
    return B1, B2a, B2b, B3

def setup_likelihood_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Set up likelihood matrices for the three-level model.
    
    Returns:
        Tuple of (A1, A2, A3) likelihood matrices
    """
    # Level 1: Perceptual likelihood
    A1 = np.array([[0.75, 0.25], 
                   [0.25, 0.75]])
    
    # Level 2: Attentional likelihood
    A2 = np.array([[0.65, 0.35], 
                   [0.35, 0.65]])
    
    # Level 3: Meta-awareness likelihood
    A3 = np.array([[0.9, 0.1], 
                   [0.1, 0.9]])
    
    return A1, A2, A3

def compute_entropy_terms(A: np.ndarray) -> np.ndarray:
    """
    Compute entropy terms for expected free energy calculation.
    
    Args:
        A: Likelihood matrix
        
    Returns:
        Entropy terms for each state
    """
    H = np.zeros(A.shape[1])
    for j in range(A.shape[1]):
        H[j] = np.sum(A[:, j] * np.log(A[:, j] + 1e-16))
    return H 