"""
Geometric Deep Learning Specification for GNN (Generalized Notation Notation)

This module provides geometric deep learning representations for Active Inference models,
incorporating Riemannian manifolds, geometric algebra, and equivariant transformations.

Research Foundation:
- RiemannFormer: A Framework for Attention in Curved Spaces
- Geometric Meta-Learning via Coupled Ricci Flow
- Geometric Algebra Transformer (GATr)
- Clifford Group Equivariant Neural Networks (CGENN)

Author: @docxology
Date: 2025-06-21
License: MIT
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

# ================================
# RIEMANNIAN MANIFOLD OPERATIONS
# ================================

class RiemannianManifold(ABC):
    """Abstract base class for Riemannian manifolds in Active Inference."""
    
    @abstractmethod
    def metric_tensor(self, point: jnp.ndarray) -> jnp.ndarray:
        """Compute the metric tensor at a given point."""
        pass
    
    @abstractmethod
    def christoffel_symbols(self, point: jnp.ndarray) -> jnp.ndarray:
        """Compute Christoffel symbols for covariant derivatives."""
        pass
    
    @abstractmethod
    def exponential_map(self, point: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
        """Exponential map from tangent space to manifold."""
        pass
    
    @abstractmethod
    def logarithmic_map(self, point1: jnp.ndarray, point2: jnp.ndarray) -> jnp.ndarray:
        """Logarithmic map from manifold to tangent space."""
        pass

class ProbabilityManifold(RiemannianManifold):
    """Riemannian manifold structure for probability distributions in Active Inference."""
    
    def __init__(self, dim: int):
        self.dim = dim
    
    def metric_tensor(self, point: jnp.ndarray) -> jnp.ndarray:
        """Fisher information metric for probability distributions."""
        # Ensure point is a valid probability distribution
        point = jnp.clip(point, 1e-8, 1.0)
        point = point / jnp.sum(point)
        
        # Fisher information metric: G_ij = 1/p_i * δ_ij
        return jnp.diag(1.0 / point)
    
    def christoffel_symbols(self, point: jnp.ndarray) -> jnp.ndarray:
        """Christoffel symbols for the Fisher information metric."""
        point = jnp.clip(point, 1e-8, 1.0)
        point = point / jnp.sum(point)
        
        gamma = jnp.zeros((self.dim, self.dim, self.dim))
        # For the Fisher metric: Γ^k_ij = -1/(2*p_k) * δ_ij for i=j=k, 0 otherwise
        for i in range(self.dim):
            gamma = gamma.at[i, i, i].set(-1.0 / (2.0 * point[i]))
        
        return gamma
    
    def exponential_map(self, point: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
        """Exponential map using natural gradient flow."""
        point = jnp.clip(point, 1e-8, 1.0)
        point = point / jnp.sum(point)
        
        # Natural gradient step
        natural_gradient = tangent / point
        updated = point * jnp.exp(natural_gradient)
        return updated / jnp.sum(updated)
    
    def logarithmic_map(self, point1: jnp.ndarray, point2: jnp.ndarray) -> jnp.ndarray:
        """Logarithmic map using log-ratio transformation."""
        point1 = jnp.clip(point1, 1e-8, 1.0)
        point2 = jnp.clip(point2, 1e-8, 1.0)
        point1 = point1 / jnp.sum(point1)
        point2 = point2 / jnp.sum(point2)
        
        return point1 * (jnp.log(point2) - jnp.log(point1))

# ================================
# GEOMETRIC ALGEBRA OPERATIONS
# ================================

@dataclass
class CliffordAlgebra:
    """Clifford algebra for geometric transformations in Active Inference."""
    
    dim: int
    signature: Tuple[int, int]  # (positive, negative) signature
    
    def __post_init__(self):
        self.total_dim = 2 ** self.dim
        self._basis_elements = self._generate_basis()
    
    def _generate_basis(self) -> List[Tuple[int, ...]]:
        """Generate basis elements for the Clifford algebra."""
        basis = []
        for i in range(self.total_dim):
            element = []
            for j in range(self.dim):
                if i & (1 << j):
                    element.append(j)
            basis.append(tuple(element))
        return basis
    
    def geometric_product(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Compute the geometric product of two multivectors."""
        result = jnp.zeros(self.total_dim)
        
        for i, basis_i in enumerate(self._basis_elements):
            for j, basis_j in enumerate(self._basis_elements):
                coeff = a[i] * b[j]
                if coeff != 0:
                    # Compute the product of basis elements
                    product_basis, sign = self._basis_product(basis_i, basis_j)
                    k = self._basis_elements.index(product_basis)
                    result = result.at[k].add(sign * coeff)
        
        return result
    
    def _basis_product(self, basis1: Tuple[int, ...], basis2: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
        """Compute the product of two basis elements."""
        combined = list(basis1) + list(basis2)
        sign = 1
        
        # Remove pairs (anticommutation)
        i = 0
        while i < len(combined):
            j = i + 1
            while j < len(combined):
                if combined[i] == combined[j]:
                    # Check signature for sign
                    if combined[i] < self.signature[0]:  # Positive signature
                        sign *= 1
                    else:  # Negative signature
                        sign *= -1
                    
                    # Remove the pair
                    combined.pop(j)
                    combined.pop(i)
                    i -= 1
                    break
                else:
                    # Count swaps for sign
                    if combined[i] > combined[j]:
                        combined[i], combined[j] = combined[j], combined[i]
                        sign *= -1
                    j += 1
            i += 1
        
        return tuple(sorted(combined)), sign
    
    def rotation(self, angle: float, plane_indices: Tuple[int, int]) -> jnp.ndarray:
        """Generate a rotation multivector."""
        rotor = jnp.zeros(self.total_dim)
        
        # Scalar part
        rotor = rotor.at[0].set(jnp.cos(angle / 2))
        
        # Bivector part
        i, j = plane_indices
        bivector_index = self._get_bivector_index(i, j)
        rotor = rotor.at[bivector_index].set(jnp.sin(angle / 2))
        
        return rotor
    
    def _get_bivector_index(self, i: int, j: int) -> int:
        """Get the index of a bivector basis element."""
        if i > j:
            i, j = j, i
        basis_element = tuple(sorted([i, j]))
        return self._basis_elements.index(basis_element)

# ================================
# EQUIVARIANT NEURAL NETWORKS
# ================================

class EquivariantLinear(nn.Module):
    """Equivariant linear layer for geometric transformations."""
    
    features: int
    group_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, group_element: jnp.ndarray) -> jnp.ndarray:
        """Apply equivariant linear transformation."""
        # Standard linear transformation
        W = self.param('weight', nn.initializers.normal(), (x.shape[-1], self.features))
        
        # Group-equivariant weights
        G = self.param('group_weight', nn.initializers.normal(), (self.group_dim, self.features, self.features))
        
        # Apply linear transformation
        y = jnp.dot(x, W)
        
        # Apply group action
        group_transform = jnp.einsum('i,ijk->jk', group_element, G)
        y = jnp.dot(y, group_transform)
        
        return y

class GeometricAttention(nn.Module):
    """Geometric attention mechanism on Riemannian manifolds."""
    
    d_model: int
    num_heads: int
    manifold: RiemannianManifold
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
        """Apply geometric attention."""
        batch_size, seq_len, _ = x.shape
        head_dim = self.d_model // self.num_heads
        
        # Query, Key, Value projections
        q = nn.Dense(self.d_model)(x)
        k = nn.Dense(self.d_model)(x)
        v = nn.Dense(self.d_model)(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim)
        
        # Compute geometric distances
        distances = self._compute_geodesic_distances(positions)
        
        # Apply attention with geometric weighting
        attention_scores = jnp.einsum('bihd,bjhd->bhij', q, k) / jnp.sqrt(head_dim)
        attention_scores = attention_scores - distances[None, None, :, :]
        
        attention_weights = nn.softmax(attention_scores)
        
        # Apply attention to values
        output = jnp.einsum('bhij,bjhd->bihd', attention_weights, v)
        output = output.reshape(batch_size, seq_len, self.d_model)
        
        return nn.Dense(self.d_model)(output)
    
    def _compute_geodesic_distances(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Compute geodesic distances between positions on the manifold."""
        seq_len = positions.shape[0]
        distances = jnp.zeros((seq_len, seq_len))
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    # Use logarithmic map to compute distance
                    tangent_vec = self.manifold.logarithmic_map(positions[i], positions[j])
                    metric = self.manifold.metric_tensor(positions[i])
                    distance = jnp.sqrt(jnp.dot(tangent_vec, jnp.dot(metric, tangent_vec)))
                    distances = distances.at[i, j].set(distance)
        
        return distances

# ================================
# ACTIVE INFERENCE GEOMETRIC MODEL
# ================================

class GeometricActiveInferenceModel(nn.Module):
    """Geometric Active Inference model with Riemannian structure."""
    
    state_dim: int
    obs_dim: int
    action_dim: int
    hidden_dim: int
    manifold: RiemannianManifold
    clifford_algebra: CliffordAlgebra
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Forward pass of the geometric Active Inference model."""
        
        # Encode observations on the manifold
        obs_encoder = nn.Dense(self.hidden_dim)
        obs_features = obs_encoder(observations)
        
        # Map to probability manifold
        belief_params = nn.Dense(self.state_dim)(obs_features)
        beliefs = nn.softmax(belief_params)
        
        # Geometric attention over beliefs
        geometric_attn = GeometricAttention(
            d_model=self.hidden_dim,
            num_heads=4,
            manifold=self.manifold
        )
        
        # Create position embeddings for beliefs
        positions = self._create_position_embeddings(beliefs)
        
        # Apply geometric attention
        attended_features = geometric_attn(
            obs_features[None, :, :], 
            positions
        )[0]
        
        # Predict next state distribution (A matrix)
        state_predictor = nn.Dense(self.state_dim * self.obs_dim)
        A_matrix_flat = state_predictor(attended_features)
        A_matrix = A_matrix_flat.reshape(self.obs_dim, self.state_dim)
        A_matrix = nn.softmax(A_matrix, axis=-1)
        
        # Predict state transitions (B matrix)
        transition_predictor = nn.Dense(self.state_dim * self.state_dim * self.action_dim)
        B_matrix_flat = transition_predictor(attended_features)
        B_matrix = B_matrix_flat.reshape(self.action_dim, self.state_dim, self.state_dim)
        B_matrix = nn.softmax(B_matrix, axis=-1)
        
        # Compute preferences (C vector) using geometric operations
        preference_encoder = nn.Dense(self.hidden_dim)
        pref_features = preference_encoder(attended_features)
        C_vector = nn.Dense(self.obs_dim)(pref_features)
        
        # Precision parameters using Clifford algebra
        precision_features = self._apply_clifford_operations(attended_features)
        gamma = nn.sigmoid(nn.Dense(1)(precision_features))
        
        return {
            'beliefs': beliefs,
            'A_matrix': A_matrix,
            'B_matrix': B_matrix,
            'C_vector': C_vector,
            'precision': gamma,
            'attended_features': attended_features,
            'positions': positions
        }
    
    def _create_position_embeddings(self, beliefs: jnp.ndarray) -> jnp.ndarray:
        """Create position embeddings for beliefs on the manifold."""
        # Use belief parameters as coordinates on the probability manifold
        return beliefs
    
    def _apply_clifford_operations(self, features: jnp.ndarray) -> jnp.ndarray:
        """Apply Clifford algebra operations for geometric transformations."""
        # Create multivector representation
        multivector_dim = self.clifford_algebra.total_dim
        mv_encoder = nn.Dense(multivector_dim)
        multivector = mv_encoder(features)
        
        # Apply geometric product with learned rotation
        rotation_params = self.param('rotation_params', 
                                   nn.initializers.uniform(), 
                                   (multivector_dim,))
        
        # Geometric product in Clifford algebra
        result = self.clifford_algebra.geometric_product(multivector, rotation_params)
        
        # Project back to feature space
        projector = nn.Dense(features.shape[-1])
        return projector(result)
    
    def compute_expected_free_energy(self, 
                                   beliefs: jnp.ndarray,
                                   A_matrix: jnp.ndarray,
                                   B_matrix: jnp.ndarray,
                                   C_vector: jnp.ndarray,
                                   actions: jnp.ndarray) -> jnp.ndarray:
        """Compute Expected Free Energy using geometric operations."""
        
        # Predict future state using geometric flow
        future_beliefs = self._geometric_belief_update(beliefs, B_matrix, actions)
        
        # Predicted observations
        predicted_obs = jnp.dot(A_matrix, future_beliefs)
        
        # Epistemic value (information gain)
        epistemic_value = self._compute_information_gain(beliefs, future_beliefs)
        
        # Pragmatic value (preference satisfaction)
        pragmatic_value = jnp.dot(predicted_obs, C_vector)
        
        # Expected Free Energy
        efe = -pragmatic_value - epistemic_value
        
        return efe
    
    def _geometric_belief_update(self, 
                               beliefs: jnp.ndarray,
                               B_matrix: jnp.ndarray,
                               actions: jnp.ndarray) -> jnp.ndarray:
        """Update beliefs using geometric flow on the manifold."""
        
        # Standard belief update
        action_idx = jnp.argmax(actions)
        transition_matrix = B_matrix[action_idx]
        new_beliefs = jnp.dot(transition_matrix.T, beliefs)
        
        # Apply geometric flow using manifold structure
        tangent_vector = self.manifold.logarithmic_map(beliefs, new_beliefs)
        geometric_beliefs = self.manifold.exponential_map(beliefs, tangent_vector)
        
        return geometric_beliefs
    
    def _compute_information_gain(self, 
                                current_beliefs: jnp.ndarray,
                                future_beliefs: jnp.ndarray) -> jnp.ndarray:
        """Compute information gain using geometric distance."""
        
        # KL divergence approximation using Fisher information metric
        metric = self.manifold.metric_tensor(current_beliefs)
        diff = future_beliefs - current_beliefs
        
        # Geometric distance as information measure
        info_gain = jnp.sqrt(jnp.dot(diff, jnp.dot(metric, diff)))
        
        return info_gain

# ================================
# RICCI FLOW OPTIMIZATION
# ================================

class RicciFlowOptimizer:
    """Ricci flow-based optimization for geometric Active Inference."""
    
    def __init__(self, manifold: RiemannianManifold, flow_rate: float = 0.01):
        self.manifold = manifold
        self.flow_rate = flow_rate
    
    def ricci_tensor(self, point: jnp.ndarray) -> jnp.ndarray:
        """Compute Ricci tensor at a given point."""
        # Simplified Ricci tensor computation
        # In practice, this would involve Christoffel symbols and curvature
        metric = self.manifold.metric_tensor(point)
        dim = metric.shape[0]
        
        # Approximate Ricci tensor using metric determinant
        det_metric = jnp.linalg.det(metric)
        ricci = -0.5 * jnp.log(det_metric) * jnp.eye(dim)
        
        return ricci
    
    def flow_step(self, metric: jnp.ndarray) -> jnp.ndarray:
        """Perform one step of Ricci flow."""
        # Extract point from metric (simplified)
        point = jnp.diag(metric)
        
        # Compute Ricci tensor
        ricci = self.ricci_tensor(point)
        
        # Ricci flow equation: ∂g/∂t = -2 Ric(g)
        metric_update = metric - 2 * self.flow_rate * ricci
        
        # Ensure positive definiteness
        eigenvals, eigenvecs = jnp.linalg.eigh(metric_update)
        eigenvals = jnp.maximum(eigenvals, 1e-8)
        metric_update = eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
        
        return metric_update
    
    def optimize_geometry(self, 
                         initial_metric: jnp.ndarray, 
                         loss_fn: Callable,
                         num_steps: int = 100) -> Tuple[jnp.ndarray, List[float]]:
        """Optimize geometry using Ricci flow to minimize loss."""
        
        current_metric = initial_metric
        losses = []
        
        for step in range(num_steps):
            # Compute loss
            loss = loss_fn(current_metric)
            losses.append(float(loss))
            
            # Ricci flow step
            current_metric = self.flow_step(current_metric)
            
            # Optional: add loss-dependent perturbation
            if step % 10 == 0:
                gradient = grad(loss_fn)(current_metric)
                current_metric = current_metric - 0.001 * gradient
        
        return current_metric, losses

# ================================
# UTILITY FUNCTIONS
# ================================

def create_geometric_model(config: Dict[str, Any]) -> GeometricActiveInferenceModel:
    """Create a geometric Active Inference model with specified configuration."""
    
    # Create probability manifold
    manifold = ProbabilityManifold(dim=config['state_dim'])
    
    # Create Clifford algebra
    clifford_algebra = CliffordAlgebra(
        dim=config.get('clifford_dim', 4),
        signature=config.get('clifford_signature', (3, 1))
    )
    
    # Create model
    model = GeometricActiveInferenceModel(
        state_dim=config['state_dim'],
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
        manifold=manifold,
        clifford_algebra=clifford_algebra
    )
    
    return model

def geometric_loss_function(predictions: Dict[str, jnp.ndarray],
                          targets: Dict[str, jnp.ndarray],
                          manifold: RiemannianManifold) -> jnp.ndarray:
    """Compute geometric loss incorporating manifold structure."""
    
    # Standard prediction losses
    belief_loss = jnp.mean((predictions['beliefs'] - targets['beliefs'])**2)
    
    # Geometric distance loss using manifold structure
    if 'target_positions' in targets:
        geometric_loss = 0.0
        for i in range(len(predictions['positions'])):
            pred_pos = predictions['positions'][i]
            target_pos = targets['target_positions'][i]
            
            # Use manifold distance
            tangent_vec = manifold.logarithmic_map(pred_pos, target_pos)
            metric = manifold.metric_tensor(pred_pos)
            distance = jnp.sqrt(jnp.dot(tangent_vec, jnp.dot(metric, tangent_vec)))
            geometric_loss += distance
        
        geometric_loss = geometric_loss / len(predictions['positions'])
    else:
        geometric_loss = 0.0
    
    # Information-theoretic loss
    info_loss = -jnp.sum(predictions['beliefs'] * jnp.log(predictions['beliefs'] + 1e-8))
    
    total_loss = belief_loss + 0.1 * geometric_loss + 0.01 * info_loss
    
    return total_loss

# ================================
# EXAMPLE USAGE AND TESTING
# ================================

def example_geometric_active_inference():
    """Example usage of geometric Active Inference model."""
    
    # Configuration
    config = {
        'state_dim': 8,
        'obs_dim': 6,
        'action_dim': 4,
        'hidden_dim': 64,
        'clifford_dim': 4,
        'clifford_signature': (3, 1)
    }
    
    # Create model
    model = create_geometric_model(config)
    
    # Initialize parameters
    key = random.PRNGKey(42)
    dummy_obs = jnp.ones((config['obs_dim'],))
    dummy_actions = jnp.ones((config['action_dim'],))
    
    params = model.init(key, dummy_obs, dummy_actions)
    
    # Forward pass
    outputs = model.apply(params, dummy_obs, dummy_actions)
    
    print("Geometric Active Inference Model Output:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
    
    # Example Ricci flow optimization
    manifold = ProbabilityManifold(dim=config['state_dim'])
    optimizer = RicciFlowOptimizer(manifold)
    
    initial_metric = jnp.eye(config['state_dim'])
    
    def dummy_loss(metric):
        return jnp.trace(metric)
    
    optimized_metric, losses = optimizer.optimize_geometry(
        initial_metric, dummy_loss, num_steps=50
    )
    
    print(f"\nRicci Flow Optimization:")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Optimized metric shape: {optimized_metric.shape}")
    
    return model, params, outputs

if __name__ == "__main__":
    # Run example
    model, params, outputs = example_geometric_active_inference()
    
    print("\nGeometric Deep Learning Specification for GNN - Ready!")
    print("Features implemented:")
    print("✓ Riemannian manifold operations")
    print("✓ Geometric algebra (Clifford algebras)")
    print("✓ Equivariant neural networks")
    print("✓ Geometric attention mechanisms")
    print("✓ Ricci flow optimization")
    print("✓ Active Inference integration") 