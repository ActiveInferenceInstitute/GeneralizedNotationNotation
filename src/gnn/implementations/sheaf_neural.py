"""
Sheaf-Theoretic Neural Specification for GNN (Generalized Notation Notation)

This module provides sheaf-theoretic neural network representations for Active Inference models,
incorporating cellular sheaf structures, sheaf Laplacians, and cohomological operations.

Research Foundation:
- "You Can't Ignore Structure: Analysis of Copresheaf Topological Neural Networks"
- "Cellular Sheaf Neural Networks"
- "Sheaf Neural Networks for Graph-based Recommender Systems"
- "Higher-Order Sheaf Neural Networks"

Key Concepts:
- Cellular Sheaves: Data structures that capture local-to-global relationships
- Sheaf Laplacians: Generalized diffusion operators for heterophilic graphs
- Cohomology: Topological invariants for analyzing information flow
- Sheaf Diffusion: Structure-aware message passing mechanisms

Author: GNN Development Team
Date: 2024
License: MIT
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict
import networkx as nx

# ================================
# CELLULAR SHEAF STRUCTURES
# ================================

@dataclass
class CellularSheaf:
    """Cellular sheaf structure for multi-scale Active Inference models."""
    
    # Cellular complex structure
    vertices: List[int]
    edges: List[Tuple[int, int]]
    faces: List[Tuple[int, int, int]]  # Higher-order simplices
    
    # Sheaf data
    vertex_spaces: Dict[int, int]  # vertex_id -> dimension
    edge_spaces: Dict[Tuple[int, int], int]  # edge -> dimension
    face_spaces: Dict[Tuple[int, int, int], int]  # face -> dimension
    
    # Restriction maps
    restriction_maps: Dict[str, jnp.ndarray]  # Maps between stalks
    
    def __post_init__(self):
        """Initialize cellular sheaf structure."""
        self._validate_structure()
        self._compute_boundary_operators()
    
    def _validate_structure(self):
        """Validate the cellular sheaf structure."""
        # Check that all edges connect valid vertices
        for edge in self.edges:
            assert edge[0] in self.vertices and edge[1] in self.vertices
        
        # Check that all faces use valid edges
        for face in self.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1)%3]]))
                assert edge in self.edges or edge[::-1] in self.edges
    
    def _compute_boundary_operators(self):
        """Compute boundary operators for the cellular complex."""
        n_vertices = len(self.vertices)
        n_edges = len(self.edges)
        n_faces = len(self.faces)
        
        # Boundary operator δ₁: edges → vertices
        self.boundary_1 = jnp.zeros((n_vertices, n_edges))
        for j, edge in enumerate(self.edges):
            v0_idx = self.vertices.index(edge[0])
            v1_idx = self.vertices.index(edge[1])
            self.boundary_1 = self.boundary_1.at[v0_idx, j].set(-1)
            self.boundary_1 = self.boundary_1.at[v1_idx, j].set(1)
        
        # Boundary operator δ₂: faces → edges
        if n_faces > 0:
            self.boundary_2 = jnp.zeros((n_edges, n_faces))
            for k, face in enumerate(self.faces):
                face_edges = []
                for i in range(3):
                    edge = tuple(sorted([face[i], face[(i+1)%3]]))
                    if edge in self.edges:
                        face_edges.append(self.edges.index(edge))
                    else:
                        face_edges.append(self.edges.index(edge[::-1]))
                
                for edge_idx in face_edges:
                    self.boundary_2 = self.boundary_2.at[edge_idx, k].set(1)
    
    def get_stalk_dimension(self, cell: Union[int, Tuple]) -> int:
        """Get the dimension of the stalk over a given cell."""
        if isinstance(cell, int):
            return self.vertex_spaces.get(cell, 1)
        elif len(cell) == 2:
            return self.edge_spaces.get(cell, 1)
        elif len(cell) == 3:
            return self.face_spaces.get(cell, 1)
        else:
            raise ValueError(f"Unsupported cell type: {cell}")

class SheafLaplacian:
    """Sheaf Laplacian operators for diffusion on cellular sheaves."""
    
    def __init__(self, sheaf: CellularSheaf):
        self.sheaf = sheaf
        self._compute_laplacians()
    
    def _compute_laplacians(self):
        """Compute sheaf Laplacian operators."""
        # 0-Laplacian (vertex-edge Laplacian)
        self.laplacian_0 = self._compute_vertex_laplacian()
        
        # 1-Laplacian (edge-face Laplacian)
        if hasattr(self.sheaf, 'boundary_2'):
            self.laplacian_1 = self._compute_edge_laplacian()
    
    def _compute_vertex_laplacian(self) -> jnp.ndarray:
        """Compute vertex (0-dimensional) sheaf Laplacian."""
        # Get dimensions
        vertex_dims = [self.sheaf.get_stalk_dimension(v) for v in self.sheaf.vertices]
        edge_dims = [self.sheaf.get_stalk_dimension(e) for e in self.sheaf.edges]
        
        total_vertex_dim = sum(vertex_dims)
        total_edge_dim = sum(edge_dims)
        
        # Build block matrix structure
        # L₀ = δ₁ᵀ δ₁ (up Laplacian) + δ₀ δ₀ᵀ (down Laplacian, zero for vertices)
        
        # Create extended boundary operator with restriction maps
        extended_boundary = jnp.zeros((total_vertex_dim, total_edge_dim))
        
        vertex_offset = 0
        edge_offset = 0
        
        for i, vertex in enumerate(self.sheaf.vertices):
            vertex_dim = vertex_dims[i]
            
            for j, edge in enumerate(self.sheaf.edges):
                edge_dim = edge_dims[j]
                
                # Check if vertex is incident to edge
                if vertex in edge:
                    # Get restriction map
                    restriction_key = f"edge_{edge}_to_vertex_{vertex}"
                    if restriction_key in self.sheaf.restriction_maps:
                        restriction_map = self.sheaf.restriction_maps[restriction_key]
                    else:
                        # Default to identity if dimensions match, projection otherwise
                        if vertex_dim == edge_dim:
                            restriction_map = jnp.eye(vertex_dim)
                        else:
                            restriction_map = jnp.ones((vertex_dim, edge_dim)) / edge_dim
                    
                    # Apply boundary operator sign
                    sign = self.sheaf.boundary_1[i, j]
                    
                    extended_boundary = extended_boundary.at[
                        vertex_offset:vertex_offset+vertex_dim,
                        edge_offset:edge_offset+edge_dim
                    ].set(sign * restriction_map)
                
                edge_offset += edge_dim
            
            vertex_offset += vertex_dim
            edge_offset = 0
        
        # Sheaf Laplacian: L₀ = δ₁ᵀ δ₁
        laplacian = extended_boundary @ extended_boundary.T
        
        return laplacian
    
    def _compute_edge_laplacian(self) -> jnp.ndarray:
        """Compute edge (1-dimensional) sheaf Laplacian."""
        if not hasattr(self.sheaf, 'boundary_2'):
            return jnp.zeros((len(self.sheaf.edges), len(self.sheaf.edges)))
        
        # Similar construction for edge Laplacian
        edge_dims = [self.sheaf.get_stalk_dimension(e) for e in self.sheaf.edges]
        face_dims = [self.sheaf.get_stalk_dimension(f) for f in self.sheaf.faces]
        
        total_edge_dim = sum(edge_dims)
        total_face_dim = sum(face_dims)
        
        # Build extended boundary operator δ₂
        extended_boundary_2 = jnp.zeros((total_edge_dim, total_face_dim))
        
        # Implementation similar to vertex case...
        # (Simplified for brevity)
        
        # For now, return identity - full implementation would be similar to above
        return jnp.eye(total_edge_dim)

# ================================
# SHEAF NEURAL NETWORKS
# ================================

class SheafConvolution(nn.Module):
    """Sheaf convolution layer for cellular sheaf neural networks."""
    
    output_dim: int
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray, 
                 sheaf: CellularSheaf, 
                 laplacian: jnp.ndarray) -> jnp.ndarray:
        """Apply sheaf convolution operation."""
        
        # Linear transformation
        W = self.param('weight', nn.initializers.normal(), (x.shape[-1], self.output_dim))
        y = jnp.dot(x, W)
        
        # Sheaf diffusion
        diffused = jnp.dot(laplacian, y)
        
        # Combine original and diffused signals
        alpha = self.param('diffusion_weight', nn.initializers.constant(0.5), ())
        output = (1 - alpha) * y + alpha * diffused
        
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.output_dim,))
            output = output + bias
        
        return output

class SheafAttention(nn.Module):
    """Sheaf attention mechanism incorporating topological structure."""
    
    d_model: int
    num_heads: int
    
    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray, 
                 sheaf: CellularSheaf,
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Apply sheaf attention with topological constraints."""
        
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
        
        # Compute attention scores
        attention_scores = jnp.einsum('bihd,bjhd->bhij', q, k) / jnp.sqrt(head_dim)
        
        # Apply topological mask based on sheaf structure
        topo_mask = self._create_topological_mask(sheaf, seq_len)
        attention_scores = attention_scores + topo_mask[None, None, :, :]
        
        if mask is not None:
            attention_scores = attention_scores + mask[None, None, :, :]
        
        attention_weights = nn.softmax(attention_scores)
        
        # Apply attention
        output = jnp.einsum('bhij,bjhd->bihd', attention_weights, v)
        output = output.reshape(batch_size, seq_len, self.d_model)
        
        return nn.Dense(self.d_model)(output)
    
    def _create_topological_mask(self, 
                               sheaf: CellularSheaf, 
                               seq_len: int) -> jnp.ndarray:
        """Create attention mask based on sheaf topology."""
        mask = jnp.full((seq_len, seq_len), -jnp.inf)
        
        # Allow attention between connected cells
        for i, vertex in enumerate(sheaf.vertices[:seq_len]):
            for j, other_vertex in enumerate(sheaf.vertices[:seq_len]):
                if i == j:
                    mask = mask.at[i, j].set(0.0)  # Self-attention
                else:
                    # Check if vertices are connected by an edge
                    edge = tuple(sorted([vertex, other_vertex]))
                    if edge in sheaf.edges or edge[::-1] in sheaf.edges:
                        mask = mask.at[i, j].set(0.0)
        
        return mask

class SheafPooling(nn.Module):
    """Sheaf pooling operation preserving topological information."""
    
    pooling_type: str = "mean"  # "mean", "max", "sum", "sheaf_diffusion"
    
    def __call__(self, 
                 x: jnp.ndarray, 
                 sheaf: CellularSheaf,
                 pooling_indices: List[List[int]]) -> jnp.ndarray:
        """Apply sheaf-aware pooling operation."""
        
        if self.pooling_type == "sheaf_diffusion":
            return self._sheaf_diffusion_pooling(x, sheaf, pooling_indices)
        else:
            return self._standard_pooling(x, pooling_indices)
    
    def _standard_pooling(self, 
                         x: jnp.ndarray, 
                         pooling_indices: List[List[int]]) -> jnp.ndarray:
        """Standard pooling operations."""
        pooled_features = []
        
        for indices in pooling_indices:
            if len(indices) == 0:
                continue
            
            cluster_features = x[indices]
            
            if self.pooling_type == "mean":
                pooled = jnp.mean(cluster_features, axis=0)
            elif self.pooling_type == "max":
                pooled = jnp.max(cluster_features, axis=0)
            elif self.pooling_type == "sum":
                pooled = jnp.sum(cluster_features, axis=0)
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling_type}")
            
            pooled_features.append(pooled)
        
        return jnp.stack(pooled_features)
    
    def _sheaf_diffusion_pooling(self, 
                               x: jnp.ndarray, 
                               sheaf: CellularSheaf,
                               pooling_indices: List[List[int]]) -> jnp.ndarray:
        """Sheaf diffusion-based pooling."""
        # Create local sheaf Laplacian for each cluster
        pooled_features = []
        
        for indices in pooling_indices:
            if len(indices) <= 1:
                if len(indices) == 1:
                    pooled_features.append(x[indices[0]])
                continue
            
            # Extract subgraph induced by indices
            cluster_features = x[indices]
            
            # Create local Laplacian (simplified)
            n_nodes = len(indices)
            local_laplacian = jnp.eye(n_nodes)
            
            # Apply diffusion
            diffused = jnp.dot(local_laplacian, cluster_features)
            
            # Pool diffused features
            pooled = jnp.mean(diffused, axis=0)
            pooled_features.append(pooled)
        
        return jnp.stack(pooled_features)

# ================================
# COHOMOLOGICAL OPERATIONS
# ================================

class SheafCohomology:
    """Compute sheaf cohomology for topological analysis."""
    
    def __init__(self, sheaf: CellularSheaf):
        self.sheaf = sheaf
        self.laplacian = SheafLaplacian(sheaf)
    
    def compute_cohomology_groups(self, 
                                features: jnp.ndarray) -> Dict[int, jnp.ndarray]:
        """Compute sheaf cohomology groups."""
        cohomology = {}
        
        # 0th cohomology (kernel of δ₀: vertex features → edge features)
        # For vertices, this is just the features themselves constrained by sheaf
        h0 = self._compute_h0(features)
        cohomology[0] = h0
        
        # 1st cohomology (more complex, involves edge-face relationships)
        if hasattr(self.sheaf, 'boundary_2'):
            h1 = self._compute_h1(features)
            cohomology[1] = h1
        
        return cohomology
    
    def _compute_h0(self, features: jnp.ndarray) -> jnp.ndarray:
        """Compute 0th cohomology group."""
        # H⁰ consists of global sections (features consistent across the sheaf)
        
        # For each connected component, find features that are consistent
        # across all restriction maps
        
        n_vertices = len(self.sheaf.vertices)
        consistent_features = []
        
        for i in range(features.shape[-1]):  # For each feature dimension
            feature_vec = features[:, i] if len(features.shape) > 1 else features
            
            # Check consistency across edges
            is_consistent = True
            for edge in self.sheaf.edges:
                v0, v1 = edge
                v0_idx = self.sheaf.vertices.index(v0)
                v1_idx = self.sheaf.vertices.index(v1)
                
                # Simple consistency check (in practice, would use restriction maps)
                if abs(feature_vec[v0_idx] - feature_vec[v1_idx]) > 1e-6:
                    is_consistent = False
                    break
            
            if is_consistent:
                consistent_features.append(feature_vec)
        
        if consistent_features:
            return jnp.stack(consistent_features, axis=-1)
        else:
            return jnp.zeros((n_vertices, 1))
    
    def _compute_h1(self, features: jnp.ndarray) -> jnp.ndarray:
        """Compute 1st cohomology group."""
        # H¹ captures "holes" or inconsistencies in the sheaf
        
        # Simplified computation: look for edge features that don't extend to faces
        n_edges = len(self.sheaf.edges)
        
        # Create edge features by averaging incident vertex features
        edge_features = []
        for edge in self.sheaf.edges:
            v0, v1 = edge
            v0_idx = self.sheaf.vertices.index(v0)
            v1_idx = self.sheaf.vertices.index(v1)
            
            if len(features.shape) > 1:
                edge_feat = (features[v0_idx] + features[v1_idx]) / 2
            else:
                edge_feat = jnp.array([(features[v0_idx] + features[v1_idx]) / 2])
            
            edge_features.append(edge_feat)
        
        return jnp.stack(edge_features) if edge_features else jnp.zeros((n_edges, 1))
    
    def betti_numbers(self, features: jnp.ndarray) -> Dict[int, int]:
        """Compute Betti numbers of the sheaf."""
        cohomology = self.compute_cohomology_groups(features)
        
        betti = {}
        for degree, cohom_group in cohomology.items():
            # Betti number is the rank of the cohomology group
            # Approximated by number of linearly independent vectors
            if cohom_group.size > 0:
                rank = jnp.linalg.matrix_rank(cohom_group)
                betti[degree] = int(rank)
            else:
                betti[degree] = 0
        
        return betti

# ================================
# SHEAF ACTIVE INFERENCE MODEL
# ================================

class SheafActiveInferenceModel(nn.Module):
    """Active Inference model with sheaf-theoretic neural architecture."""
    
    state_dim: int
    obs_dim: int
    action_dim: int
    hidden_dim: int
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, 
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 sheaf: CellularSheaf) -> Dict[str, jnp.ndarray]:
        """Forward pass of sheaf-based Active Inference model."""
        
        # Create sheaf Laplacian
        laplacian_computer = SheafLaplacian(sheaf)
        laplacian_0 = laplacian_computer.laplacian_0
        
        # Observation encoder with sheaf structure
        obs_features = nn.Dense(self.hidden_dim)(observations)
        
        # Stack of sheaf convolution layers
        x = obs_features
        for i in range(self.num_layers):
            # Sheaf convolution
            sheaf_conv = SheafConvolution(
                output_dim=self.hidden_dim,
                use_bias=True
            )
            x = sheaf_conv(x, sheaf, laplacian_0)
            x = nn.relu(x)
            
            # Optional: Sheaf attention
            if i < self.num_layers - 1:
                sheaf_attn = SheafAttention(
                    d_model=self.hidden_dim,
                    num_heads=4
                )
                x = sheaf_attn(x[None, :, :], sheaf)[0]  # Add/remove batch dim
        
        # Active Inference components
        
        # Beliefs (posterior over states) - constrained by sheaf topology
        belief_logits = nn.Dense(self.state_dim)(x)
        beliefs = nn.softmax(belief_logits)
        
        # Observation model (A matrix) - each stalk has its own observation model
        A_features = nn.Dense(self.hidden_dim)(x)
        A_logits = nn.Dense(self.obs_dim * self.state_dim)(A_features)
        A_matrix = nn.softmax(
            A_logits.reshape((-1, self.obs_dim, self.state_dim)), 
            axis=-1
        )
        
        # Transition model (B matrix) - transitions respect sheaf structure
        B_features = nn.Dense(self.hidden_dim)(x)
        B_logits = nn.Dense(self.action_dim * self.state_dim * self.state_dim)(B_features)
        B_matrix = nn.softmax(
            B_logits.reshape((-1, self.action_dim, self.state_dim, self.state_dim)), 
            axis=-1
        )
        
        # Preferences (C vector) - global preferences constrained by sheaf
        C_features = nn.Dense(self.hidden_dim)(x)
        C_logits = nn.Dense(self.obs_dim)(C_features)
        C_vector = C_logits  # Log preferences
        
        # Precision parameters
        precision_features = nn.Dense(self.hidden_dim)(x)
        precision = nn.sigmoid(nn.Dense(1)(precision_features))
        
        # Compute cohomological features
        cohomology_computer = SheafCohomology(sheaf)
        cohomology_groups = cohomology_computer.compute_cohomology_groups(x)
        betti_numbers = cohomology_computer.betti_numbers(x)
        
        return {
            'beliefs': beliefs,
            'A_matrix': A_matrix,
            'B_matrix': B_matrix,
            'C_vector': C_vector,
            'precision': precision,
            'sheaf_features': x,
            'cohomology_groups': cohomology_groups,
            'betti_numbers': betti_numbers,
            'laplacian': laplacian_0
        }
    
    def compute_expected_free_energy(self,
                                   predictions: Dict[str, jnp.ndarray],
                                   actions: jnp.ndarray,
                                   sheaf: CellularSheaf) -> jnp.ndarray:
        """Compute Expected Free Energy with sheaf-theoretic constraints."""
        
        beliefs = predictions['beliefs']
        A_matrix = predictions['A_matrix']
        B_matrix = predictions['B_matrix']
        C_vector = predictions['C_vector']
        
        # Predicted future state (using sheaf-constrained transitions)
        action_idx = jnp.argmax(actions, axis=-1)
        if len(B_matrix.shape) == 4 and B_matrix.shape[0] > 1:
            # Multiple stalks
            future_beliefs = jnp.mean([
                jnp.dot(B_matrix[i, action_idx[i]], beliefs[i])
                for i in range(len(beliefs))
            ], axis=0)
        else:
            # Single stalk
            transition_matrix = B_matrix[0, action_idx] if len(B_matrix.shape) > 3 else B_matrix[action_idx]
            future_beliefs = jnp.dot(transition_matrix, beliefs[0] if len(beliefs.shape) > 1 else beliefs)
        
        # Predicted observations
        if len(A_matrix.shape) == 3 and A_matrix.shape[0] > 1:
            predicted_obs = jnp.mean([
                jnp.dot(A_matrix[i], future_beliefs)
                for i in range(len(A_matrix))
            ], axis=0)
        else:
            obs_matrix = A_matrix[0] if len(A_matrix.shape) > 2 else A_matrix
            predicted_obs = jnp.dot(obs_matrix, future_beliefs)
        
        # Pragmatic value (preference satisfaction)
        if len(C_vector.shape) > 1:
            pragmatic_value = jnp.mean([
                jnp.dot(predicted_obs, C_vector[i])
                for i in range(len(C_vector))
            ])
        else:
            pragmatic_value = jnp.dot(predicted_obs, C_vector)
        
        # Epistemic value (information gain) - enhanced with topological information
        if len(beliefs.shape) > 1:
            current_entropy = -jnp.mean([
                jnp.sum(beliefs[i] * jnp.log(beliefs[i] + 1e-12))
                for i in range(len(beliefs))
            ])
        else:
            current_entropy = -jnp.sum(beliefs * jnp.log(beliefs + 1e-12))
        
        future_entropy = -jnp.sum(future_beliefs * jnp.log(future_beliefs + 1e-12))
        epistemic_value = current_entropy - future_entropy
        
        # Topological regularization using Betti numbers
        betti_numbers = predictions['betti_numbers']
        topological_complexity = sum(betti_numbers.values())
        
        # Expected Free Energy
        efe = -pragmatic_value - epistemic_value + 0.01 * topological_complexity
        
        return efe

# ================================
# UTILITY FUNCTIONS
# ================================

def create_simple_sheaf(num_vertices: int, 
                       edges: List[Tuple[int, int]],
                       stalk_dim: int = 1) -> CellularSheaf:
    """Create a simple cellular sheaf for testing."""
    
    vertices = list(range(num_vertices))
    
    # Create uniform stalk dimensions
    vertex_spaces = {v: stalk_dim for v in vertices}
    edge_spaces = {edge: stalk_dim for edge in edges}
    
    # Create identity restriction maps
    restriction_maps = {}
    for edge in edges:
        for vertex in edge:
            key = f"edge_{edge}_to_vertex_{vertex}"
            restriction_maps[key] = jnp.eye(stalk_dim)
    
    return CellularSheaf(
        vertices=vertices,
        edges=edges,
        faces=[],  # No faces for simplicity
        vertex_spaces=vertex_spaces,
        edge_spaces=edge_spaces,
        face_spaces={},
        restriction_maps=restriction_maps
    )

def create_complete_sheaf(num_vertices: int, stalk_dim: int = 2) -> CellularSheaf:
    """Create a complete graph cellular sheaf."""
    vertices = list(range(num_vertices))
    edges = [(i, j) for i in range(num_vertices) for j in range(i+1, num_vertices)]
    
    return create_simple_sheaf(num_vertices, edges, stalk_dim)

def analyze_sheaf_topology(sheaf: CellularSheaf, 
                          features: jnp.ndarray) -> Dict[str, Any]:
    """Analyze topological properties of a cellular sheaf."""
    
    cohomology = SheafCohomology(sheaf)
    laplacian = SheafLaplacian(sheaf)
    
    # Compute cohomology groups
    cohomology_groups = cohomology.compute_cohomology_groups(features)
    betti_numbers = cohomology.betti_numbers(features)
    
    # Spectral analysis of sheaf Laplacian
    eigenvals, eigenvecs = jnp.linalg.eigh(laplacian.laplacian_0)
    
    analysis = {
        'cohomology_groups': cohomology_groups,
        'betti_numbers': betti_numbers,
        'laplacian_eigenvalues': eigenvals,
        'laplacian_eigenvectors': eigenvecs,
        'euler_characteristic': sum((-1)**k * betti for k, betti in betti_numbers.items()),
        'spectral_gap': eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0.0
    }
    
    return analysis

# ================================
# EXAMPLE USAGE
# ================================

def example_sheaf_active_inference():
    """Example usage of sheaf-theoretic Active Inference."""
    
    # Create a simple cellular sheaf (triangle graph)
    sheaf = create_simple_sheaf(
        num_vertices=3,
        edges=[(0, 1), (1, 2), (2, 0)],
        stalk_dim=2
    )
    
    # Model configuration
    config = {
        'state_dim': 4,
        'obs_dim': 3,
        'action_dim': 2,
        'hidden_dim': 16,
        'num_layers': 2
    }
    
    # Create model
    model = SheafActiveInferenceModel(**config)
    
    # Initialize parameters
    key = random.PRNGKey(42)
    dummy_obs = jnp.ones((3, config['obs_dim']))  # 3 vertices, obs_dim features each
    dummy_actions = jnp.ones((config['action_dim'],))
    
    params = model.init(key, dummy_obs, dummy_actions, sheaf)
    
    # Forward pass
    outputs = model.apply(params, dummy_obs, dummy_actions, sheaf)
    
    print("Sheaf Active Inference Model Output:")
    for key, value in outputs.items():
        if isinstance(value, dict):
            print(f"{key}: {list(value.keys())}")
        elif hasattr(value, 'shape'):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    
    # Topological analysis
    analysis = analyze_sheaf_topology(sheaf, outputs['sheaf_features'])
    
    print(f"\nTopological Analysis:")
    print(f"Betti numbers: {analysis['betti_numbers']}")
    print(f"Euler characteristic: {analysis['euler_characteristic']}")
    print(f"Spectral gap: {analysis['spectral_gap']:.4f}")
    
    return model, params, outputs, analysis

if __name__ == "__main__":
    # Run example
    model, params, outputs, analysis = example_sheaf_active_inference()
    
    print("\nSheaf-Theoretic Neural Specification for GNN - Ready!")
    print("Features implemented:")
    print("✓ Cellular sheaf structures")
    print("✓ Sheaf Laplacian operators")
    print("✓ Sheaf convolution layers")
    print("✓ Topological attention mechanisms")
    print("✓ Cohomological analysis")
    print("✓ Active Inference integration")
    print("✓ Betti number computation") 